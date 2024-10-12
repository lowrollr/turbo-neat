import concurrent.futures
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

import wandb
from neat_jax.config import NEATConfig
from neat_jax.genome import Genome, apply, init_genome
from neat_jax.logging import log_to_wandb, render_and_log_episode
from neat_jax.population import Population, _init_population
from neat_jax.utils import is_printable, mask_data
from neat_jax.visualize import visualize_genome_as_nn

FitnessFn = Callable[[chex.PRNGKey, Genome, Dict], chex.Array]


class NEAT:
    """NEAT algorithm"""

    def __init__(
        self,
        config: NEATConfig,
        fitness_fn: FitnessFn,
        h2h_test_fn: Optional[FitnessFn] = None,
        baseline_test_fn: Optional[FitnessFn] = None,
        wandb_project: Optional[str] = None,
    ):
        self.config = config
        self.activation_selector = config.genome_config.activation_selector
        if wandb_project is not None:
            self.wandb_run = wandb.init(project=wandb_project, config=config.__dict__)
        else:
            self.wandb_run = None
        self.evaluate_population = jax.jit(
            partial(self.evaluate, fitness_fn=fitness_fn)
        )
        if h2h_test_fn is not None:
            self.test_champion = jax.jit(
                partial(self.test_against_champion, h2h_fn=h2h_test_fn)
            )
        else:
            self.test_champion = None
        self.evolve = jax.jit(self.evolve_one_generation)
        if baseline_test_fn is not None:
            self.test_baseline = jax.jit(
                partial(self.test_against_baseline, test_fn=baseline_test_fn)
            )
        else:
            self.test_baseline = None

    def init_population(
        self, rng: chex.PRNGKey, generation: int = 0, initial_innovation_id: int = 0
    ) -> Population:
        """Initialize a population of genomes"""
        batched_genome = init_genome(
            rng,
            batch_size=self.config.selection_config.population_size,
            capacity=self.config.genome_config.initial_capacity,
            input_size=self.config.genome_config.input_size,
            output_size=self.config.genome_config.output_size,
            input_activation_ids=self.config.genome_config.input_activation_ids,
            output_activation_ids=self.config.genome_config.output_activation_ids,
            weight_mean=self.config.genome_config.init_weight_mean,
            weight_std=self.config.genome_config.init_weight_std,
            mode=self.config.genome_config.init_mode,
            initial_innovation_id=initial_innovation_id,
        )

        next_innovation_id = batched_genome.graph.innovation_ids.max() + 1

        return _init_population(
            batched_genome,
            self.config.mutation_config,
            self.config.selection_config,
            next_innovation_id,
            generation,
        )

    def resize_genome(self, population: Population) -> Population:
        """
        Expand the capacity of the genome. This will cause jitted functions to be recompiled.
        So it should be called sparingly!
        """
        strategy = self.config.genome_config.capacity_growth_strategy
        if strategy == "linear":
            new_capacity = (
                population.batched_genome.capacity
                + self.config.genome_config.initial_capacity
            )
        elif strategy == "exponential":
            new_capacity = population.batched_genome.capacity * 2
        elif strategy == "constant":
            return population

        amount = new_capacity - population.batched_genome.capacity
        extended_genome = apply(
            population.batched_genome, Genome.extend_capacity, amount=amount
        )
        extended_prev_genome = apply(
            population.prev_batched_genome, Genome.extend_capacity, amount=amount
        )
        extended_champion = population.champion.extend_capacity(amount=amount)
        return population.replace(
            batched_genome=extended_genome,
            prev_batched_genome=extended_prev_genome,
            champion=extended_champion,
        )

    def evaluate(
        self, rng: chex.PRNGKey, population: Population, fitness_fn: FitnessFn
    ) -> Population:
        """Evaluate the fitness of a genome"""
        population = population.replace(
            batched_genome=population.batched_genome.prepare_for_inference()
        )
        genome = population.batched_genome
        # Evaluate fitness of population
        fitness_rng, rng = jax.random.split(rng)
        fitnesses, _ = fitness_fn(
            rng=fitness_rng, genome=genome, activation_selector=self.activation_selector
        )
        # store raw fitnesses in genome
        genome = genome.replace(fitness=fitnesses)
        return population.replace(batched_genome=genome)

    def evolve_one_generation(
        self, rng: chex.PRNGKey, population: Population
    ) -> Population:
        """Evolve the population by one generation"""
        # Prepare genomes for crossover
        population = population.replace(
            batched_genome=population.batched_genome.prepare_for_crossover()
        )
        # create next generation
        selection_rng, rng = jax.random.split(rng)
        population = population.create_next_generation(
            selection_rng,
            selection_config=self.config.selection_config,
            mutation_config=self.config.mutation_config,
            genome_config=self.config.genome_config,
        )
        return population.replace(generation=population.generation + 1)

    def test_against_baseline(
        self, rng: chex.PRNGKey, population: Population, test_fn: FitnessFn
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Evaluate the fitness of the most fit member of the population on a test set"""
        genomes: Genome = population.batched_genome
        most_fit_genome_idx = jnp.argmax(genomes.fitness)
        # copy most fit genome to all idxs
        dup_genome = jax.tree_map(
            lambda x: jnp.broadcast_to(x[most_fit_genome_idx], x.shape), genomes
        )
        fitnesses, data = test_fn(
            rng=rng, genome=dup_genome, activation_selector=self.activation_selector
        )
        return fitnesses.mean(), data

    def test_against_champion(
        self, rng: chex.PRNGKey, population: Population, h2h_fn: FitnessFn
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        genome: Genome = population.batched_genome
        champion = population.champion
        challenger_idx = jnp.argmax(genome.fitness)
        challenger = jax.tree_map(lambda x: x[challenger_idx], genome)
        b_champion = jax.tree_map(
            lambda x, y: jnp.broadcast_to(x, y.shape), champion, genome
        )
        b_challenger = jax.tree_map(
            lambda x, y: jnp.broadcast_to(x, y.shape), challenger, genome
        )
        fitnesses, _ = h2h_fn(
            rng=rng,
            genome_1=b_champion,
            genome_2=b_challenger,
            activation_selector=self.activation_selector,
        )
        champion_fitness = fitnesses[0]
        challenger_fitness = fitnesses[1]
        new_champion = challenger_fitness > champion_fitness
        mask_fn = partial(mask_data, mask=new_champion)
        population = population.replace(
            champion=jax.tree_map(mask_fn, challenger, champion)
        )
        return population, challenger_fitness, new_champion

    def run(
        self,
        seed: int,
        num_generations: int,
        population: Optional[Population] = None,
        render_fn: Optional[Callable] = None,
    ):
        """Run the NEAT algorithm"""
        rng = jax.random.PRNGKey(seed)
        if population is None:
            rng_init, rng = jax.random.split(rng, 2)
            population = self.init_population(rng_init)

        prev_stats = dict()
        species_stats = population.species_data.get_species_stats(prev_stats=prev_stats)
        best_fitness = float("-inf")

        # this just functions as a logging job queue
        logger = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        for g in range(num_generations):
            results = {}
            rng, gen_rng, eval_rng, test_rng = jax.random.split(rng, 4)
            # evaluate population
            population = self.evaluate_population(eval_rng, population)
            # adjust mutation noise
            num_unique_species = len(
                jnp.unique(population.batched_genome.species_id).tolist()
            )
            # store results
            results.update(
                {
                    "generation": population.generation,
                    "mean_fitness": population.batched_genome.fitness.mean(),
                    "max_fitness": population.batched_genome.fitness.max(),
                    "min_fitness": population.batched_genome.fitness.min(),
                    "mean_hidden_nodes": population.batched_genome.get_hidden_node_counts(
                        use_condensed=False
                    ).mean(),
                    "mean_condensed_hidden_nodes": population.batched_genome.get_hidden_node_counts(
                        use_condensed=True
                    ).mean(),
                    "mean_connections": population.batched_genome.num_enabled_connections.mean(),
                    "mean_condensed_connections": population.batched_genome.condensed_size.mean(),
                    "num_species": num_unique_species,
                }
            )
            new_species_stats = population.species_data.get_species_stats(
                prev_stats=prev_stats
            )
            species_stats, prev_stats = new_species_stats, species_stats
            prev_stats = population.species_data.fill_prev_stats(
                prev_stats=prev_stats, cur_stats=species_stats
            )

            if self.test_champion:
                # test against champion
                test_rng, rng = jax.random.split(rng)
                population, challenger_fitness, improved = self.test_champion(
                    test_rng, population
                )

                results.update(
                    {
                        "challenger_fitness": challenger_fitness,
                        "improved": int(improved),
                    }
                )
            else:
                max_fitness = population.batched_genome.fitness.max().item()
                max_idx = jnp.argmax(population.batched_genome.fitness)
                improved = max_fitness > best_fitness
                best_fitness = max(max_fitness, best_fitness)
                if improved:
                    population = population.replace(
                        champion=jax.tree_map(
                            lambda x: x[max_idx], population.batched_genome
                        )
                    )

            # test against baseline and render video
            render_data = None
            if improved:
                # create nn visualization
                visualize_genome_as_nn(
                    population.champion,
                    self.config.genome_config.activation_map,
                    filename="neural_network",
                    input_labels=self.config.genome_config.input_labels,
                    output_labels=self.config.genome_config.output_labels,
                )

                if self.test_baseline:
                    if self.wandb_run is not None:
                        results["nn"] = wandb.Image("neural_network.png")
                    test_rng, rng = jax.random.split(rng)
                    test_fitness, render_data = self.test_baseline(test_rng, population)
                    results["fitness_against_baseline"] = test_fitness

            # evolve population
            population = self.evolve(gen_rng, population)

            if (
                self.wandb_run is not None
                and render_fn is not None
                and render_data is not None
            ):
                # spin off a thread to render and log because rendering is expensive and i/o bound
                logger.submit(
                    render_and_log_episode,
                    self.wandb_run,
                    prev_stats,
                    results,
                    jax.device_put(render_data, jax.devices("cpu")[0]),
                    render_fn,
                    g,
                )
            elif self.wandb_run is not None:
                logger.submit(
                    log_to_wandb,
                    self.wandb_run,
                    prev_stats,
                    results,
                    g,
                )

            results.update(**species_stats)
            print({k: f"{v:.2f}" for k, v in results.items() if is_printable(v)})

            # check if genome needs more space allocated
            if population.batched_genome.is_almost_full().any():
                population = self.resize_genome(population)
                print(f"Resized genome to {population.batched_genome.capacity} nodes")

        logger.shutdown(wait=True)
        # return the population when done
        return population
