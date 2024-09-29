from typing import Dict, List, Optional

import graphviz
import jax.numpy as jnp

from neat_jax.activations import visualization_color_mapping
from neat_jax.genome import Genome


def visualize_genome_as_nn(
    genome: Genome,
    activation_map: Dict,
    input_labels: Optional[List[str]] = None,
    output_labels: Optional[List[str]] = None,
    filename="neural_network",
) -> graphviz.Digraph:
    """Visualize a genome as a neural network using graphviz
    assumes that the graph is in topological order
    """

    dot = graphviz.Digraph("neural network", comment="")
    dot.attr(rankdir="LR", size="10,!", ratio="fill", bgcolor="transparent")
    inputs = genome.input_idxs.tolist()
    outputs = genome.output_idxs.tolist()

    weight_min = jnp.abs(genome.graph.weights).min().item()
    weight_max = jnp.abs(genome.graph.weights).max().item()

    nodes = set()
    for i in range(genome.condensed_size.item()):
        from_node = genome.graph.from_nodes[i].item()
        to_node = genome.graph.to_nodes[i].item()
        weight = genome.graph.weights[i].item()
        nodes.add(from_node)
        nodes.add(to_node)
        if weight > 0:
            color = "green"
        else:
            color = "red"
        dot.edge(
            str(from_node),
            str(to_node),
            color=color,
            penwidth=str(
                1 + 5 * (jnp.abs(weight) - weight_min) / (weight_max - weight_min)
            ),
        )

    with dot.subgraph() as sub:
        sub.attr(rank="same")
        for i, s in enumerate(inputs):
            sub.node(
                str(s),
                label=f"{input_labels[i]}\n{activation_map[genome.node_activation_ids[s].item()]}",
                style="filled",
                fillcolor="blue",
                fontcolor="white",
                fontname="Arial",
                fontsize="15",
            )

    with dot.subgraph() as sub:
        sub.attr(rank="same")
        for i, s in enumerate(outputs):
            sub.node(
                str(s),
                label=f"{output_labels[i]}\n{activation_map[genome.node_activation_ids[s].item()]}",
                style="filled",
                fillcolor="red",
                fontcolor="white",
                fontname="Arial",
                fontsize="15",
            )

    inputs = set(inputs)
    outputs = set(outputs)

    for i in nodes:
        if i in inputs or i in outputs:
            continue
        activation = activation_map[genome.node_activation_ids[i].item()]
        dot.node(
            str(i),
            label=f"{activation}",
            style="filled",
            fillcolor=visualization_color_mapping[activation],
            fontcolor="white",
            fontname="Arial",
            fontsize="15",
        )

    dot.render(filename, format="png", cleanup=True)
