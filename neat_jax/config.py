from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from neat_jax.activations import (
    ActivationFn,
    ActivationSelector,
    make_activation_selector_fn,
)


@dataclass
class MutationConfig:
    add_connection_prob: float
    disable_connection_prob: float
    add_node_prob: float
    disable_node_prob: float
    mutate_weight_prob: float
    mutate_weight_std: float
    mutate_activation_prob: float
    mutate_bias_prob: float
    mutate_bias_std: float
    allow_output_act_mutation: bool = False
    allow_output_bias_mutation: bool = False
    allow_input_act_mutation: bool = False
    allow_input_bias_mutation: bool = False
    learning_rate: float = 0.0  # only used for backprop neat


@dataclass
class SelectionConfig:
    population_size: int
    cutoff_pct: float
    compatibility_coefficients: Tuple[float, float]
    stagnation_fn: Callable
    offspring_temperature: float = 1.0
    fitness_ema_period: int = 10
    selection_tournament_size: int = 3
    speciation_threshold: float = 3.0
    elitism: float = 0.1
    max_transfer_age: Optional[int] = None
    maximum_species: int = 10
    min_species_size: int = 10
    species_warmup_threshold: int = 5


@dataclass
class GenomeConfig:
    input_size: int
    output_size: int
    initial_capacity: int
    init_weight_mean: float
    init_weight_std: float
    activation_fns: List[ActivationFn]
    input_activation_ids: List[int] = field(default_factory=list)
    output_activation_ids: List[int] = field(default_factory=list)
    input_labels: List[str] = field(default_factory=list)
    output_labels: List[str] = field(default_factory=list)
    capacity_growth_strategy: str = "linear"
    init_mode: str = "partial"

    @property
    def activation_map(self) -> Dict:
        return {i: f.__name__ for i, f in enumerate(self.activation_fns)}

    @property
    def num_activation_functions(self) -> int:
        return len(self.activation_fns)

    @property
    def activation_selector(self) -> ActivationSelector:
        return make_activation_selector_fn(self.activation_fns)


@dataclass
class NEATConfig:
    # mutation
    mutation_config: MutationConfig
    # genome
    genome_config: GenomeConfig
    # selection
    selection_config: SelectionConfig

    @property
    def __dict__(self) -> Dict:
        return {
            "mutation_config": self.mutation_config.__dict__,
            "selection_config": self.selection_config.__dict__,
            "genome_config": self.genome_config.__dict__,
        }
