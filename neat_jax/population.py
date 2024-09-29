from __future__ import annotations

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from neat_jax.config import GenomeConfig, MutationConfig, SelectionConfig
from neat_jax.genome import Genome
from neat_jax.species import SpeciesData, init_species_data
from neat_jax.utils import apply_where, mask_data, round_to_integers


@dataclass(frozen=True)
class Population:
    batched_genome: Genome
    prev_batched_genome: Genome
    champion: Genome
    next_innovation_id: jnp.int32
    generation: jnp.int32
    species_data: SpeciesData
    # store in population for mutation scaling
    add_node_prob: float
    add_connection_prob: float
    disable_node_prob: float
    disable_connection_prob: float
    mutate_weight_prob: float
    mutate_weight_std: float
    mutate_activation_prob: float
    mutate_bias_prob: float
    mutate_bias_std: float

    def speciate(self, selection_config: SelectionConfig) -> Population:
        population = self.replace(species_data=self.species_data.reset_counts())
        c_conn, c_weight = selection_config.compatibility_coefficients
        prev = population.prev_batched_genome
        new = population.batched_genome
        compatibility_fn = partial(
            Genome.compatibility_distance, c_conn=c_conn, c_weight=c_weight
        )
        compatibility_fn_over_prev = jax.vmap(
            compatibility_fn, in_axes=(0, None), out_axes=0
        )
        compatibility_fn_over_new = jax.vmap(
            compatibility_fn_over_prev, in_axes=(None, 0), out_axes=1
        )

        distances_to_prev = compatibility_fn_over_new(new, prev)

        def assign_species(idx, population):
            new = population.batched_genome

            species_data = population.species_data
            new_i = jax.tree_map(lambda x: x[idx], new)
            compatibility_fn = partial(
                new_i.compatibility_distance, c_conn=c_conn, c_weight=c_weight
            )
            dist_to_prev = distances_to_prev[idx]
            dist_to_new = jax.vmap(compatibility_fn)(new)
            seen_in_new_mask = jnp.arange(new.batch_size) < idx

            # sum distances to each species in prev
            match_species_prev = (
                species_data.species_id[None, :] == prev.species_id[:, None]
            )
            species_distances = jnp.where(
                match_species_prev & species_data.mask, dist_to_prev[:, None], 0
            ).sum(axis=0)
            species_counts = jnp.where(
                match_species_prev & species_data.mask, 1, 0
            ).sum(axis=0)
            # sum distances to each species in new
            match_species_new = (
                species_data.species_id[None, :] == new.species_id[:, None]
            )
            species_distances += jnp.where(
                match_species_new & seen_in_new_mask[:, None] & species_data.mask,
                dist_to_new[:, None],
                0,
            ).sum(axis=0)
            species_counts += jnp.where(
                match_species_new & seen_in_new_mask[:, None] & species_data.mask, 1, 0
            ).sum(axis=0)

            mean_distances = jnp.where(
                species_counts > 0, species_distances / species_counts, jnp.inf
            )

            species_data, assigned_species = species_data.assign_species(
                mean_distances,
                new_i.species_id,
                new_i.fitness,
                selection_config.speciation_threshold,
                max_transfer_age=selection_config.max_transfer_age,
            )

            new = new.replace(species_id=new.species_id.at[idx].set(assigned_species))
            return population.replace(batched_genome=new, species_data=species_data)

        population = jax.lax.fori_loop(0, prev.batch_size, assign_species, population)

        return population.replace(
            species_data=population.species_data.remove_extinct_species()
            .next_gen(selection_config.fitness_ema_period)
            .remove_stagnant_species(selection_config.stagnation_fn)
        )

    def mutate(
        self,
        rng: chex.PRNGKey,
        elite_mask: chex.Array,
        mutation_config: MutationConfig,
        genome_config: GenomeConfig,
    ) -> Population:
        """Mutate the population"""
        batched_genome = self.batched_genome
        batch_size = batched_genome.batch_size
        next_innovation_id = self.next_innovation_id
        not_elites_mask = ~elite_mask

        # disable connections
        disable_conn_prob = self.disable_connection_prob
        rng, rng_disable_conn, rng_mask = jax.random.split(rng, 3)
        disable_conn_mask = jax.random.bernoulli(
            rng_mask, disable_conn_prob, (batch_size,)
        )
        disable_conn_mask &= not_elites_mask
        batched_genome = apply_where(
            batched_genome,
            Genome.disable_random_connection,
            disable_conn_mask,
            jax.random.split(rng_disable_conn, batch_size),
        )

        # disable nodes
        disable_node_prob = self.disable_node_prob
        rng, rng_disable_node, rng_mask = jax.random.split(rng, 3)
        disable_node_mask = jax.random.bernoulli(
            rng_mask, disable_node_prob, (batch_size,)
        )
        disable_node_mask &= not_elites_mask
        batched_genome = apply_where(
            batched_genome,
            Genome.disable_random_node,
            disable_node_mask,
            jax.random.split(rng_disable_node, batch_size),
        )

        # add nodes
        add_node_prob = self.add_node_prob
        rng, rng_add_node, rng_mask = jax.random.split(rng, 3)
        add_node_mask = jax.random.bernoulli(rng_mask, add_node_prob, (batch_size,))
        add_node_mask &= not_elites_mask
        # allocate innovation ids
        new_ids_needed = add_node_mask.sum()
        innovation_ids = (
            add_node_mask.repeat(2).cumsum() + next_innovation_id - 1
        ).reshape(batch_size, 2)
        next_innovation_id += new_ids_needed * 2

        ids_1 = innovation_ids[:, 0]
        ids_2 = innovation_ids[:, 1]

        batched_genome = apply_where(
            batched_genome,
            Genome.add_random_node,
            add_node_mask,
            jax.random.split(rng_add_node, batch_size),
            ids_1,
            ids_2,
            num_activation_fns=genome_config.num_activation_functions,
        )

        # add connections
        add_conn_prob = self.add_connection_prob
        rng, rng_add_conn, rng_mask = jax.random.split(rng, 3)
        add_conn_mask = jax.random.bernoulli(rng_mask, add_conn_prob, (batch_size,))
        add_conn_mask &= not_elites_mask
        # allocate innovation ids
        new_ids_needed = add_conn_mask.sum()
        innovation_ids = add_conn_mask.cumsum() + next_innovation_id - 1
        next_innovation_id += new_ids_needed

        batched_genome = apply_where(
            batched_genome,
            Genome.add_random_connection,
            add_conn_mask,
            jax.random.split(rng_add_conn, batch_size),
            innovation_ids,
            weight_mean=genome_config.init_weight_mean,
            weight_std=genome_config.init_weight_std,
        )

        # mutate nodes
        mutate_act_prob = self.mutate_activation_prob
        mutate_bias_prob = self.mutate_bias_prob
        rng, rng_mutate_node = jax.random.split(rng, 2)
        batched_genome = apply_where(
            batched_genome,
            Genome.mutate_nodes,
            not_elites_mask,
            jax.random.split(rng_mutate_node, batch_size),
            mutate_act_prob=mutate_act_prob,
            mutate_bias_prob=mutate_bias_prob,
            num_activation_fns=genome_config.num_activation_functions,
            bias_noise_std=self.mutate_bias_std,
            allow_output_activation_mutation=mutation_config.allow_output_act_mutation,
            allow_input_activation_mutation=mutation_config.allow_input_act_mutation,
            allow_output_bias_mutation=mutation_config.allow_output_bias_mutation,
            allow_input_bias_mutation=mutation_config.allow_input_bias_mutation,
        )

        # mutate connections
        mutate_conn_prob = self.mutate_weight_prob
        rng, rng_mutate_conn = jax.random.split(rng, 2)
        batched_genome = apply_where(
            batched_genome,
            Genome.mutate_connections,
            not_elites_mask,
            jax.random.split(rng_mutate_conn, batch_size),
            mutation_prob=mutate_conn_prob,
            noise_std=self.mutate_weight_std,
        )
        return self.replace(
            batched_genome=batched_genome, next_innovation_id=next_innovation_id
        )  # pylint: disable=no-member

    def crossover(
        self,
        rng: chex.PRNGKey,
        elite_mask: chex.Array,
        parent_idxs_1: chex.Array,
        parent_idxs_2: chex.Array,
    ) -> Population:
        parents_1 = jax.tree_map(lambda x: x[parent_idxs_1], self.batched_genome)
        parents_2 = jax.tree_map(lambda x: x[parent_idxs_2], self.batched_genome)
        crossover_mask = (parents_1.fitness > parents_2.fitness) | elite_mask
        apply_mask = partial(mask_data, mask=crossover_mask)
        superior_parents = jax.tree_map(apply_mask, parents_1, parents_2)
        inferior_parents = jax.tree_map(apply_mask, parents_2, parents_1)
        # perform crossover
        crossover_keys = jax.random.split(rng, self.batched_genome.batch_size)
        # apply crossover to non-elite members
        offspring = apply_where(
            superior_parents,
            Genome.crossover,
            (~elite_mask),
            crossover_keys,
            inferior_parents,
        )
        return self.replace(
            prev_batched_genome=self.batched_genome, batched_genome=offspring
        )  # pylint: disable=no-member

    def allocate_offspring(
        self, selection_config: SelectionConfig
    ) -> Tuple[chex.Array, chex.Array]:
        species_mask = self.species_data.mask
        ema_min = self.species_data.ema_fitness.min(where=species_mask, initial=jnp.inf)
        ema_max = self.species_data.ema_fitness.max(
            where=species_mask, initial=-jnp.inf
        )
        norm_fitness = (self.species_data.ema_fitness - ema_min) / (
            ema_max - ema_min + jnp.finfo(jnp.float32).eps
        )
        offspring_proportion = jax.nn.softmax(
            norm_fitness, where=species_mask, initial=0.0
        )
        # add any leftover evenly to masked species (this happens when every species is warming up)
        offspring_counts_float = offspring_proportion * selection_config.population_size
        offspring_counts_rounded = round_to_integers(
            offspring_counts_float, selection_config.population_size
        )
        elite_counts = jnp.minimum(
            jnp.floor(selection_config.elitism * self.species_data.species_size).astype(
                jnp.int32
            ),
            offspring_counts_rounded,
        )
        offspring_counts = offspring_counts_rounded - elite_counts
        species_2x = jnp.tile(
            jnp.arange(selection_config.maximum_species, dtype=jnp.int32), 2
        )
        elite_and_offspring = jnp.concatenate([elite_counts, offspring_counts])
        new_pop_species_idxs = jnp.repeat(
            species_2x,
            elite_and_offspring,
            total_repeat_length=selection_config.population_size,
        )
        elite_mask = jnp.arange(self.batched_genome.batch_size) < elite_counts.sum()

        return new_pop_species_idxs, elite_mask

    def select_parents(
        self,
        rng: chex.PRNGKey,
        offspring_species_idxs: chex.Array,
        elite_mask: chex.Array,
        selection_config: SelectionConfig,
    ) -> Tuple[chex.Array, chex.Array]:
        offspring_species = self.species_data.species_id[offspring_species_idxs]
        sorted_species = jnp.argsort(
            jnp.where(
                self.batched_genome.species_id[None, ...]
                == offspring_species[..., None],
                self.batched_genome.fitness,
                -jnp.inf,
            ),
            axis=1,
            descending=True,
        )

        mask = (
            jnp.arange(self.batched_genome.batch_size)
            < ((1 - selection_config.cutoff_pct) * self.species_data.species_size)[
                ..., None
            ]
        )

        probs = (
            mask[offspring_species_idxs]
            * self.species_data.species_size[offspring_species_idxs]
        ) / (
            self.species_data.species_size[offspring_species_idxs]
            * mask[offspring_species_idxs]
        ).sum(axis=-1)
        rng1, rng2 = jax.random.split(rng)
        rng1 = jax.random.split(rng1, self.batched_genome.batch_size)
        rng2 = jax.random.split(rng2, self.batched_genome.batch_size)

        choice_fn = partial(
            jax.random.choice, shape=(selection_config.selection_tournament_size,)
        )
        parents_1_pool = jax.vmap(choice_fn)(rng1, sorted_species, p=probs)
        parents_2_pool = jax.vmap(choice_fn)(rng2, sorted_species, p=probs)

        parents_1 = parents_1_pool[
            jnp.arange(self.batched_genome.batch_size),
            jnp.argmax(self.batched_genome.fitness[parents_1_pool], axis=-1),
        ]
        parents_2 = parents_2_pool[
            jnp.arange(self.batched_genome.batch_size),
            jnp.argmax(self.batched_genome.fitness[parents_2_pool], axis=-1),
        ]

        # assign elite parents (these will have no offspring and will just be copied to the next generation)
        elite_idxs = jnp.zeros_like(parents_1, dtype=jnp.int32)
        species_d_mask = (
            (offspring_species_idxs == jnp.roll(offspring_species_idxs, shift=1))
            .at[0]
            .set(False)
        )
        elite_idxs = jax.lax.fori_loop(
            0,
            self.batched_genome.batch_size,
            lambda i, e: e.at[i].set(
                jnp.where(species_d_mask[i] & elite_mask[i], e[i - 1] + 1, 0)
            ),
            elite_idxs,
        )

        parents_1 = jnp.where(
            elite_mask,
            sorted_species[jnp.arange(self.batched_genome.batch_size), elite_idxs],
            parents_1,
        )
        return parents_1, parents_2

    def create_next_generation(
        self,
        rng: chex.PRNGKey,
        selection_config: SelectionConfig,
        mutation_config: MutationConfig,
        genome_config: GenomeConfig,
    ) -> Population:
        # speciate current generation
        population = self.speciate(selection_config)
        # allocate new population members to species
        offspring_species_idxs, elite_mask = population.allocate_offspring(
            selection_config
        )
        # select parents
        parents_rng, rng = jax.random.split(rng)
        parent_idxs_1, parent_idxs_2 = population.select_parents(
            parents_rng, offspring_species_idxs, elite_mask, selection_config
        )
        # crossover
        crossover_rng, rng = jax.random.split(rng)
        population = population.crossover(
            crossover_rng, elite_mask, parent_idxs_1, parent_idxs_2
        )
        # mutate
        mutate_rng, rng = jax.random.split(rng)
        population = population.mutate(
            mutate_rng, elite_mask, mutation_config, genome_config
        )
        return population


def _init_population(
    batched_genome: Genome,
    mutation_config: MutationConfig,
    selection_config: SelectionConfig,
    next_innovation_id: int,
    generation: int = 0,
    next_species_id: int = 0,
) -> Population:
    return Population(
        batched_genome=batched_genome,
        prev_batched_genome=batched_genome,
        champion=jax.tree_map(lambda x: x[0], batched_genome),
        next_innovation_id=jnp.int32(next_innovation_id),
        generation=jnp.int32(generation),
        species_data=init_species_data(
            selection_config.maximum_species, next_species_id
        ),
        add_node_prob=mutation_config.add_node_prob,
        add_connection_prob=mutation_config.add_connection_prob,
        disable_node_prob=mutation_config.disable_node_prob,
        disable_connection_prob=mutation_config.disable_connection_prob,
        mutate_weight_prob=mutation_config.mutate_weight_prob,
        mutate_weight_std=mutation_config.mutate_weight_std,
        mutate_activation_prob=mutation_config.mutate_activation_prob,
        mutate_bias_prob=mutation_config.mutate_bias_prob,
        mutate_bias_std=mutation_config.mutate_bias_std,
    )
