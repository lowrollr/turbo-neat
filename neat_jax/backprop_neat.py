from functools import partial
from typing import Callable, Dict, Optional, Tuple

import chex
import jax
import wandb

from neat_jax.config import NEATConfig
from neat_jax.genome import Genome
from neat_jax.neat import NEAT, FitnessFn
from neat_jax.population import Population

# returns grads and fitness (-loss)
BackpropFn = Callable[[chex.PRNGKey, Genome, Dict], Tuple[chex.ArrayTree, chex.Array]]


class BackpropNEAT(NEAT):
    """NEAT algorithm with backpropagation for network parameters"""

    def __init__(
        self,
        config: NEATConfig,
        backprop_fn: BackpropFn,
        test_fn: Optional[FitnessFn] = None,
        wandb_project: Optional[str] = None,
    ):
        self.config = config
        self.activation_selector = config.genome_config.activation_selector
        if wandb_project is not None:
            self.wandb_run = wandb.init(project=wandb_project, config=config.__dict__)
        else:
            self.wandb_run = None

        self.evaluate_population = jax.jit(
            partial(self.evaluate, fitness_fn=backprop_fn)
        )
        self.evolve = jax.jit(partial(self.evolve_one_generation))
        if test_fn is not None:
            self.test_baseline = jax.jit(
                partial(self.test_against_baseline, test_fn=test_fn)
            )
        else:
            self.test_baseline = None

        self.test_champion = None

        if config.mutation_config.mutate_weight_prob > 0:
            print(
                "Mutate weight probability is greater than 0. This is not supported by backprop NEAT. Setting mutate_weight_prob to 0."
            )
            self.config.mutation_config.mutate_weight_prob = 0
        if config.mutation_config.mutate_bias_prob > 0:
            print(
                "Mutate bias probability is greater than 0. This is not supported by backprop NEAT. Setting mutate_bias_prob to 0."
            )
            self.config.mutation_config.mutate_bias_prob = 0
        if config.mutation_config.learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0 for backprop NEAT")

    def one_generation_backprop(
        self, rng: chex.PRNGKey, population: Population, backprop_fn: FitnessFn
    ) -> Population:
        """Evolve the population by one generation"""
        # Prepare genomes for crossover
        population = population.replace(
            batched_genome=population.batched_genome.prepare_for_crossover()
        )
        # create next generation
        selection_rng, rng = jax.random.split(rng)
        population = jax.lax.cond(
            population.generation == 0,
            lambda _: population,
            lambda _: population.create_next_generation(
                selection_rng,
                selection_config=self.config.selection_config,
                mutation_config=self.config.mutation_config,
                genome_config=self.config.genome_config,
            ),
            operand=None,
        )
        # topologically sort genomes to prepare for inference
        population = population.replace(
            batched_genome=population.batched_genome.prepare_for_inference()
        )
        genome = population.batched_genome
        # Evaluate fitness of population
        fitness_rng, rng = jax.random.split(rng)
        fitnesses, grads = backprop_fn(
            rng=fitness_rng, genome=genome, activation_selector=self.activation_selector
        )
        # assumed to be weight grads, bias grads
        wgrads, bgrads = grads[0], grads[1]
        weights = (
            genome.graph.weights - self.config.mutation_config.learning_rate * wgrads
        )
        biases = genome.node_biases - self.config.mutation_config.learning_rate * bgrads
        genome = genome.replace(
            graph=genome.graph.replace(weights=weights),
            node_biases=biases,
            fitness=fitnesses,
        )

        population = population.replace(
            batched_genome=genome, generation=population.generation + 1
        )

        return population
