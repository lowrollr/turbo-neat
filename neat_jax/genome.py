

from __future__ import annotations

from functools import partial
from typing import List

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from neat_jax.activations import ActivationSelector
from neat_jax.utils import apply


@dataclass(frozen=True)
class TopsortState:
    reachable: chex.Array
    cur_nodes: chex.Array
    node_rank: chex.Array
    cur_rank: int
    # see Genome.topsort_and_condense

@dataclass(frozen=True)
class ValidConnectionState:
    visit_mask: chex.Array
    added_new: chex.Array
    # see Genome.get_valid_connections


@dataclass(frozen=True)
class Graph:
    # all of size (C,)
    from_nodes: chex.Array
    to_nodes: chex.Array
    weights: chex.Array
    innovation_ids: chex.Array
    enabled_mask: chex.Array


    @property
    def capacity(self) -> int:
        if self.from_nodes.ndim == 1:
            return self.from_nodes.shape[0]
        return self.from_nodes.shape[1]
    
    
def init_empty_graph(capacity: int) -> Graph:
    return Graph(
        from_nodes=jnp.zeros(capacity, dtype=jnp.int32),
        to_nodes=jnp.zeros(capacity, dtype=jnp.int32),
        weights=jnp.zeros(capacity, dtype=jnp.float32),
        innovation_ids=jnp.zeros(capacity, dtype=jnp.int32),
        enabled_mask=jnp.zeros(capacity, dtype=jnp.bool_),
    )


@dataclass(frozen=True)
class Genome:
    graph: Graph
    input_idxs: chex.Array
    output_idxs: chex.Array
    node_mask: chex.Array
    node_activation_ids: chex.Array
    node_biases: chex.Array
    next_conn_idx: jnp.int32
    next_node_idx: jnp.int32
    num_initial_connections: jnp.int32
    condensed_size: jnp.int32
    fitness: jnp.int32
    species_id: jnp.int32
    
    @property
    def capacity(self) -> int:
        return self.graph.capacity
    
    @property
    def is_single(self) -> bool:
        return self.graph.from_nodes.ndim == 1

    @property
    def batch_size(self) -> int:
        # check if batched
        if self.is_single:
            return 1
        return self.graph.from_nodes.shape[0]
    
    @property
    def input_size(self) -> int:
        if self.is_single:
            return self.input_idxs.shape[0]
        return self.input_idxs.shape[1]
    
    @property
    def output_size(self) -> int:
        if self.is_single:
            return self.output_idxs.shape[0]
        return self.output_idxs.shape[1]
    
    @property
    def input_mask(self) -> chex.Array:
        return jnp.zeros(self.capacity, dtype=jnp.bool_).at[self.input_idxs].set(True)
    
    @property
    def output_mask(self) -> chex.Array:
        return jnp.zeros(self.capacity, dtype=jnp.bool_).at[self.output_idxs].set(True)
    
    @property
    def initialized_conn_mask(self) -> chex.Array:
        """Mask initialized connections."""
        if self.is_single:
            return jnp.arange(self.capacity) < self.next_conn_idx
        else:
            return jnp.arange(self.capacity) < self.next_conn_idx[..., None]
    
    @property
    def condensed_node_mask(self) -> chex.Array:
        """assumes topologically ordered"""
        if self.is_single:
            conn_mask = jnp.arange(self.capacity) < self.condensed_size
            node_mask = jnp.zeros_like(self.node_mask)
            node_mask = node_mask.at[self.graph.to_nodes].max(conn_mask)
        else:
            conn_mask = jnp.arange(self.capacity) < self.condensed_size[..., None]
            node_mask = jnp.zeros_like(self.node_mask)
            node_mask = node_mask.at[jnp.arange(self.batch_size)[...,None], self.graph.to_nodes].max(conn_mask)
        
        return node_mask
    
    @property
    def num_enabled_connections(self) -> chex.Array:
        return jnp.sum(self.graph.enabled_mask, axis=-1)
    
        
    @property
    def redundance(self) -> chex.Array:
        """Get redundance of genome: number of enabled connections not present in the condensed graph."""
        return jnp.sum(self.graph.enabled_mask, axis=-1) - self.condensed_size
    
    
    def is_almost_full(self, space_needed: int = 3) -> jnp.bool_:
        """Check if genome is almost full. 
        Genome can at most grow by 3: (2 for add_node, 1 for add_connection)"""
        return (self.capacity - self.next_conn_idx) < space_needed
    

    def get_initialized_innovation_ids(self, no_id_val = None) -> chex.Array:
        return jnp.where(
            self.initialized_conn_mask,
            self.graph.innovation_ids,
            no_id_val if no_id_val is not None else jnp.iinfo(self.graph.innovation_ids).max
        )
    

    def sort_graph_by_innovation(self) -> Genome:
        mask = self.initialized_conn_mask
        innovation_ids = self.get_initialized_innovation_ids(mask)
        sorted_indices = jnp.lexsort((innovation_ids, mask.astype(jnp.int32) * -1))
        sorted_graph = jax.tree_util.tree_map(lambda x: x[sorted_indices], self.graph)
        return self.replace(graph=sorted_graph) # pylint: disable=no-member
    

    def highest_common_node(self, other: Genome) -> chex.Array:
        """Get highest common node between two genomes (by common innovation ids)
        * assumes both genomes are ordered by innovation
        * returns -1 if no common nodes
        """
        common_innovations = self.graph.innovation_ids == other.graph.innovation_ids
        return jnp.where(common_innovations, other.graph.to_nodes, -1).max()
    

    # using activation_selector like this is weird...
    def forward(self, x: chex.Array, activation_selector: ActivationSelector, diff_mode=False) -> chex.Array:
        """Pass data through the neural network implemented by the genome.

        * x can be of arbitrary shape, but the last dimension must match the input size
        * assumes edges are sorted topology-wise
        * we generally expect many calls to forward w/ the same graph, so topsort is not called here
        assumes input nodes are contained within the first N=input size contiguous nodes
        """
        input_shape = x.shape[:-1]
        graph = self.graph
        # broadcast to allow arbitrary input shapes
        # as long as they are ..., input_size
        # init node values to their biases
        node_values = jnp.broadcast_to(self.node_biases, input_shape + (self.node_biases.shape[-1],))
        # set input node values
        node_values = node_values.at[..., self.input_idxs].set(x)

        # propagate values through the graph!
        def propagate(index: int, node_values: chex.Array) -> chex.Array:
            from_node = graph.from_nodes[index]
            to_node = graph.to_nodes[index]
            weight = graph.weights[index]
            enabled = graph.enabled_mask[index]
            activated_value = enabled * activation_selector(self.node_activation_ids[from_node], node_values[...,from_node]) * weight
            node_values = node_values.at[...,to_node].add(activated_value)
            return node_values
       
        # # TODO: try scan+unroll? 
        if diff_mode:
            node_values = jax.lax.fori_loop(0, self.capacity, propagate, node_values)
        else:
            node_values = jax.lax.fori_loop(0, self.condensed_size, propagate, node_values)

        # apply output activation functions
        output = node_values[..., self.output_idxs]
        output_activation_ids = self.node_activation_ids[self.output_idxs]
        # need to put last dimension (contains an individual samples output) first
        output_activations = jax.vmap(activation_selector)(output_activation_ids, jnp.moveaxis(output, -1, 0))
        # restore dimension order
        return jnp.moveaxis(output_activations, 0, -1)
    

    def init_connection(self, rng: chex.PRNGKey, from_node: int, to_node: int, innovation_id: int,
                        weight_mean: float, weight_std: float) -> Genome:
        """Initialize a connection between two nodes."""
        weight = jax.random.normal(rng, shape=(), dtype=jnp.float32) * weight_std + weight_mean
        return self.replace( # pylint: disable=no-member
            graph = self.graph.replace(
                from_nodes = self.graph.from_nodes.at[self.next_conn_idx].set(from_node),
                to_nodes = self.graph.to_nodes.at[self.next_conn_idx].set(to_node),
                weights = self.graph.weights.at[self.next_conn_idx].set(weight),
                innovation_ids = self.graph.innovation_ids.at[self.next_conn_idx].set(innovation_id),
                enabled_mask = self.graph.enabled_mask.at[self.next_conn_idx].set(True),
            ),
            next_conn_idx = self.next_conn_idx + 1
        )
    
    
    def update_connection(self, rng: chex.PRNGKey, innovation_id: int, weight_mean: float, weight_std: float) -> Genome:
        weight = jax.random.normal(rng, shape=(), dtype=jnp.float32) * weight_std + weight_mean
        idx = jnp.argmax(self.graph.innovation_ids == innovation_id)
        return self.replace( # pylint: disable=no-member
            graph = self.graph.replace(
                weights = self.graph.weights.at[idx].set(weight),
                enabled_mask = self.graph.enabled_mask.at[idx].set(True),
            )
        )


    def fully_connect(self, rng: chex.PRNGKey, innovation_id: int, weight_mean: float, weight_std: float) -> Genome:
        """Fully connect all input nodes to all output nodes."""
        num_connections = self.input_size * self.output_size
        keys = jax.random.split(rng, num_connections)
        return jax.lax.fori_loop(
            0, num_connections,
            lambda i, s: s.init_connection(
                keys[i],
                from_node=i % self.input_size,
                to_node=self.input_size + (i // self.input_size),
                innovation_id=innovation_id + i,
                weight_mean=weight_mean,
                weight_std=weight_std),
            self
        ).replace(condensed_size=num_connections, num_initial_connections=num_connections)
        
    
    def partially_connect(self, rng: chex.PRNGKey, innovation_id: int, weight_mean: float, weight_std: float) -> Genome:
        """Partially connect all input nodes to all output nodes."""
        conn_key, node_key = jax.random.split(rng, 2)
        to_nodes = jax.random.randint(node_key, shape=(self.input_size,), minval=self.input_size, maxval=self.input_size + self.output_size)
        conn_keys = jax.random.split(conn_key, self.input_size)
        genome = self
        return jax.lax.fori_loop(
            0, self.input_size,
            lambda i, s: s.init_connection(
                conn_keys[i],
                from_node=i,
                to_node=to_nodes[i],
                innovation_id=innovation_id + i,
                weight_mean=weight_mean,
                weight_std=weight_std),
            genome).replace(condensed_size=self.input_size, num_initial_connections=self.input_size)
    

    def extend_capacity(self, amount: int) -> Genome:
        """Extend the capacity of the genome.
        * this will cause JIT recompilation, so it should be called sparingly
        """
        padded_graph = jax.tree_util.tree_map(
            lambda x: jnp.pad(x, (0,amount), constant_values=0),
            self.graph
        )
        padded_node_mask = jnp.pad(self.node_mask, (0,amount), constant_values=False)
        padded_act_ids = jnp.pad(self.node_activation_ids, (0,amount), constant_values=0)
        padded_biases = jnp.pad(self.node_biases, (0,amount), constant_values=0.0)
        return self.replace( # pylint: disable=no-member
            graph=padded_graph,
            node_mask=padded_node_mask,
            node_activation_ids=padded_act_ids,
            node_biases=padded_biases
        )
    

    def topsort_and_condense(self) -> Genome:
        """ perform topological sort treating input nodes as sources
        * assumes graph is DAG
        * nodes not on a path from input to output are assigned max rank
        * assigns graph.condensed_size = number of edges in the topological order
        """

        graph = self.graph
        init_reachable = jnp.zeros(self.capacity, dtype=jnp.bool_).at[self.input_idxs].set(True)
        topsort_state = TopsortState(
            reachable = init_reachable,
            cur_nodes = init_reachable,
            node_rank = jnp.full(self.capacity, jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
            cur_rank = 0
        )

        def cond_fn(state: TopsortState):
            return jnp.any(state.cur_nodes)
        
        def body_fn(state: TopsortState, forward: bool):
            from_, to_ = (graph.from_nodes, graph.to_nodes) if forward else (graph.to_nodes, graph.from_nodes)
            # set rank of current nodes
            node_rank = jnp.where(state.cur_nodes, state.cur_rank, state.node_rank)
            # get mask of connections from current nodes
            connection_mask = state.cur_nodes[from_] & graph.enabled_mask
            # get nodes connected to current nodes
            to_nodes = jnp.where(connection_mask, to_, 0)
            new_nodes = jnp.zeros(self.capacity, dtype=jnp.bool_)
            new_nodes = new_nodes.at[to_nodes].max(connection_mask)        
            # set reachable nodes
            reachable = state.reachable | new_nodes
            return TopsortState(
                reachable=reachable,
                cur_nodes=new_nodes,
                node_rank=node_rank,
                cur_rank=state.cur_rank + 1
            )
        
        forward_pass = partial(body_fn, forward=True)
        backward_pass = partial(body_fn, forward=False)

        fw_state = jax.lax.while_loop(cond_fn, forward_pass, topsort_state)

        init_bw_reachable = jnp.zeros(self.capacity, dtype=jnp.bool_).at[self.output_idxs].set(True)
        bw_state = topsort_state.replace( # pylint: disable=no-member
            reachable = init_bw_reachable,
            cur_nodes = init_bw_reachable,
        )

        bw_state = jax.lax.while_loop(cond_fn, backward_pass, bw_state)
        common = fw_state.reachable & bw_state.reachable
        valid_edges = common[graph.from_nodes] & common[graph.to_nodes] & graph.enabled_mask
        edge_ranks = jnp.where(
            valid_edges,
            fw_state.node_rank[graph.from_nodes],
            jnp.iinfo(jnp.int32).max
        )
        sorted_rank_indices = jnp.argsort(edge_ranks, axis=-1)
        sorted_graph = jax.tree_util.tree_map(lambda x: x[sorted_rank_indices], graph)
        condensed_size = jnp.sum(edge_ranks != jnp.iinfo(jnp.int32).max, axis=-1)
        return self.replace( # pylint: disable=no-member
            graph=sorted_graph,
            condensed_size=condensed_size
        )
    

    def crossover(self, rng: chex.PRNGKey, other: Genome) -> Genome:
        """ Cross genome with another genome.
        * assumes both genomes are ordered by innovation
        """
        rng_conn, rng_node = jax.random.split(rng, 2)
        # cross connections
        common_innovations = (self.graph.innovation_ids == other.graph.innovation_ids)
        take_conn_from_other = jax.random.bernoulli(rng_conn, 0.5, shape=(self.capacity,))
        take_conn_from_other &= common_innovations

        # cross nodes
        take_node_from_other = jax.random.bernoulli(rng_node, 0.5, shape=(self.capacity,))
        highest_common_node = self.highest_common_node(other)
        take_node_from_other &= (jnp.arange(self.capacity) <= highest_common_node)

        return self.replace( # pylint: disable=no-member
            graph = jax.tree_util.tree_map(
                lambda g, o: jnp.where(take_conn_from_other, o, g),
                self.graph, other.graph),
            node_activation_ids = jnp.where(take_node_from_other, other.node_activation_ids, self.node_activation_ids),
            node_biases = jnp.where(take_node_from_other, other.node_biases, self.node_biases)
        )


    def get_hidden_node_counts(self, use_condensed=True) -> chex.Array:
        valid_node_mask = jnp.logical_not(self.output_mask | self.input_mask) \
            & (self.condensed_node_mask if use_condensed else True) & self.node_mask
        return valid_node_mask.sum(axis=-1)
    
    
    def get_valid_connections(self, node_id: jnp.int32) -> chex.Array:
        """Get valid nodes to connect a given node to."""
        state = ValidConnectionState(
            visit_mask = jnp.zeros(self.capacity, dtype=jnp.bool_).at[node_id].set(True), 
            added_new = jnp.ones(self.capacity, dtype=jnp.bool_)
        )

        def cond_fn(state: ValidConnectionState):
            return jnp.any(state.added_new)
        
        def body_fn(state: ValidConnectionState):
            connection_mask = state.visit_mask[self.graph.to_nodes]
            connected_nodes = jnp.where(connection_mask, self.graph.from_nodes, node_id)
            new_visit_mask = state.visit_mask.at[connected_nodes].set(True)
            return state.replace(
                added_new = jnp.not_equal(state.visit_mask, new_visit_mask),
                visit_mask = new_visit_mask
            )
        
        state = jax.lax.while_loop(cond_fn, body_fn, state)

        invalid_mask = state.visit_mask | self.input_mask | (~self.node_mask)
        # can't duplicate an existing enabled connection
        connect_from_node_id_mask = self.graph.from_nodes == node_id
        already_connected_to = jnp.where(connect_from_node_id_mask & self.graph.enabled_mask, self.graph.to_nodes, node_id)
        invalid_mask = invalid_mask.at[already_connected_to].set(True)
        return ~invalid_mask
    

    def disable_random_connection(self, rng: chex.PRNGKey) -> Genome:
        """Disable a random connection that is currently enabled."""
        enabled_mask = self.graph.enabled_mask
        # randomly select a connection to disable
        probs = enabled_mask / enabled_mask.sum(axis=-1)
        connection = jax.random.choice(rng, self.capacity, p=probs)
        # disable the connection
        enabled_mask = enabled_mask.at[connection].set(False)
        # if we set an already disabled connection to False, it will remain False
        # this happens when there are no enabled connections to disable
        return self.replace(graph = self.graph.replace(enabled_mask=enabled_mask)) # pylint: disable=no-member


    def add_random_connection(self, rng: chex.PRNGKey, innovation_id: jnp.int32, weight_mean: float, weight_std: float) -> Genome:
        """Add a new connection to the genome."""
        # randomly select a node to connect from
        valid_from_nodes = self.node_mask & (~self.output_mask)
        from_rng, to_rng, init_rng = jax.random.split(rng, 3)
        probs = valid_from_nodes / valid_from_nodes.sum(axis=-1)
        from_node = jax.random.choice(from_rng, self.capacity, p=probs)

        valid_connection_mask = self.get_valid_connections(from_node)
        # this will contain nans if there are no valid connections
        # but our conditional will handle this
        probs = valid_connection_mask / valid_connection_mask.sum(axis=-1)
        to_node = jax.random.choice(to_rng, self.capacity, p=probs)
        # check if a disabled connection already exists (there can only be one)
        from_mask = self.graph.from_nodes == from_node
        to_mask = self.graph.to_nodes == to_node
        existing_connection = from_mask & to_mask & (~self.graph.enabled_mask)
        is_existing_connection = existing_connection.any()
        # if one exists, replace the connection at that innovation idx instead (re-enable it with a new weight)
        innovation_id = jnp.where(
            is_existing_connection,
            self.graph.innovation_ids[existing_connection.argmax()],
            innovation_id
        )

        return jax.lax.cond(
            valid_connection_mask.any(),
            lambda _: jax.lax.cond(
                is_existing_connection,
                lambda _: self.update_connection(init_rng, innovation_id, weight_mean, weight_std),
                lambda _: self.init_connection(init_rng, from_node, to_node, innovation_id, weight_mean, weight_std),
                operand=None),
            lambda _: self,
            operand=None
        )
    

    def add_random_node(self, rng: chex.PRNGKey, innovation_id_1: chex.PRNGKey, innovation_id_2: chex.PRNGKey,
                        num_activation_fns: int) -> Genome:
        """Split a random enabled connection and add a new node in between. 
        If there are no enabled connections, its a no-op."""
        mask = self.initialized_conn_mask & self.graph.enabled_mask
        return jax.lax.cond(
            mask.any(),
            lambda _: self._add_node(rng, mask, innovation_id_1, innovation_id_2, num_activation_fns),
            lambda _: self,
            operand=None
        )


    def _add_node(self, rng: chex.PRNGKey, mask: chex.Array, innovation_id_1: chex.PRNGKey, innovation_id_2: chex.PRNGKey, 
                  num_activation_fns: int) -> Genome:
        """Split a random existing connection and add a new node in between."""
        rng_replace, rng_activation, rng_conn1, rng_conn2 = jax.random.split(rng, 4)
        # randomly select an enabled connection to replace
        probs = mask / mask.sum(axis=-1)
        replace_idx = jax.random.choice(rng_replace, self.capacity, p=probs)
        # get nodes from replaced connection
        from_node, to_node = self.graph.from_nodes[replace_idx], self.graph.to_nodes[replace_idx]
        new_node_idx = self.next_node_idx
        # randomly select an activation function for the new node
        activation_id = jax.random.choice(rng_activation, num_activation_fns)
        old_weight = self.graph.weights[replace_idx]
        # add new connections
        genome = self.init_connection(rng_conn1, from_node, new_node_idx, innovation_id_1, 1.0, 0.0)
        genome = genome.init_connection(rng_conn2, new_node_idx, to_node, innovation_id_2, old_weight, 0.0)
        # disabled replaced connection
        enabled_mask = genome.graph.enabled_mask
        enabled_mask = enabled_mask.at[replace_idx].set(False)
        return genome.replace(
            graph = genome.graph.replace(
                enabled_mask = enabled_mask,
            ),
            node_activation_ids = genome.node_activation_ids.at[new_node_idx].set(activation_id),
            next_node_idx = new_node_idx + 1,
            node_mask = genome.node_mask.at[new_node_idx].set(True),
            node_biases = genome.node_biases.at[new_node_idx].set(0.0),
        )
    

    def disable_random_node(self, rng: chex.PRNGKey) -> Genome:
        """Disable a random non-input, non-output node and all connections to and from it."""
        valid_nodes = self.node_mask & (~self.output_mask) & (~self.input_mask)
        # randomly select a node to disable
        probs = valid_nodes / valid_nodes.sum()
        node_id = jax.random.choice(rng, self.capacity, p=probs)
        # disable all connections to and from the node
        apply_mask = valid_nodes.any()
        enabled_mask = self.graph.enabled_mask
        enabled_mask &= ~((self.graph.from_nodes == node_id) & apply_mask)
        enabled_mask &= ~((self.graph.to_nodes == node_id) & apply_mask)
        return self.replace( # pylint: disable=no-member
            graph = self.graph.replace(
                enabled_mask = enabled_mask,
            ),
            node_mask = self.node_mask.at[node_id].set(~apply_mask),
        )


    def mutate_connections(self, rng: chex.PRNGKey, mutation_prob: float, noise_std: float) -> Genome:
        """Mutate the weights of connections in the genome"""
        rng_noise, rng_mutate = jax.random.split(rng, 2)
        # randomly select connections to mutate
        mutate_mask = jax.random.bernoulli(rng_mutate, mutation_prob, shape=(self.capacity,))
        mutate_mask &= self.graph.enabled_mask
        noise = jax.random.normal(rng_noise, shape=(self.capacity,)) * noise_std
        # mutate the weights of the selected connections
        weights = self.graph.weights
        weights = jnp.where(mutate_mask, weights + noise, weights)
        return self.replace(graph = self.graph.replace(weights = weights)) # pylint: disable=no-member


    def mutate_nodes(self, rng: chex.PRNGKey, mutate_act_prob: float, mutate_bias_prob: float, num_activation_fns: int,
                     bias_noise_std: float, allow_output_activation_mutation: bool = False,
                     allow_input_activation_mutation: bool = False, allow_output_bias_mutation: bool = True,
                     allow_input_bias_mutation: bool = False) -> Genome:
        """Mutate the activation functions and biases of nodes."""
        rng_mutate_act, rng_mutate_bias, rng_activation, rng_bias = jax.random.split(rng, 4)
        # randomly select nodes to mutate
        act_mutate_mask = jax.random.bernoulli(rng_mutate_act, mutate_act_prob, shape=(self.capacity,))
        # must be enabled and not an input or output node (unless allowed)
        act_mutate_mask &= self.node_mask & \
            ((~self.output_mask) | allow_output_activation_mutation) & \
            ((~self.input_mask) | allow_input_activation_mutation)
        
        bias_mutate_mask = jax.random.bernoulli(rng_mutate_bias, mutate_bias_prob, shape=(self.capacity,))
        bias_mutate_mask &= self.node_mask & \
            ((~self.output_mask) | allow_output_bias_mutation) & \
            ((~self.input_mask) | allow_input_bias_mutation)
 
        # randomly select new activation functions for the selected nodes
        activation_ids = jax.random.choice(rng_activation, num_activation_fns, shape=(self.capacity,))
        # randomly select new biases for the selected nodes
        bias_noise = jax.random.normal(rng_bias, shape=(self.capacity,)) * bias_noise_std
        # mutate the activation functions and biases of the selected nodes
        node_activation_ids = self.node_activation_ids
        node_activation_ids = jnp.where(act_mutate_mask, activation_ids, node_activation_ids)
        new_bias = jnp.where(bias_mutate_mask, self.node_biases + bias_noise, self.node_biases)
        return self.replace( # pylint: disable=no-member
            node_activation_ids = node_activation_ids,
            node_biases = new_bias
        )
    
    
    def compatibility_distance(self, other: Genome, c_conn: float, c_weight: float) -> chex.Array:
        """ get compatibility distances between two batched genomes"""
        # get intersecting connections by innovation number (assume sorted by innovation)
        intersect_mask = (other.graph.innovation_ids == self.graph.innovation_ids)
        # mask out uninitialized connections
        intersect_mask &= self.initialized_conn_mask
        
        intersect_conn_counts = jnp.sum(intersect_mask, axis=-1)
    
        non_homologous_conn_count_self = self.next_conn_idx - intersect_conn_counts
        non_homologous_conn_count_other = other.next_conn_idx - intersect_conn_counts
        non_homologous_counts = non_homologous_conn_count_self + non_homologous_conn_count_other

        weight_diff = jnp.abs(self.graph.weights - other.graph.weights)
        masked_weight_diff = jnp.sum(jnp.where(intersect_mask, weight_diff, 0), axis=-1)
        total_weight_diff = masked_weight_diff
        # get total genome sizes
        size_self = self.next_conn_idx 
        size_other = other.next_conn_idx
        # get mean weight diff
        mean_weight_diff = total_weight_diff / jnp.maximum(intersect_conn_counts, 1)
        # get larger genome size
        n = jnp.maximum(jnp.maximum(size_self, size_other), 1)
        return (c_conn * non_homologous_counts / n) + (c_weight * mean_weight_diff)


    def prepare_for_inference(self) -> Genome:
        if self.is_single:
            return self.topsort_and_condense()
        else:
            return jax.vmap(Genome.topsort_and_condense)(self)

    def prepare_for_crossover(self) -> Genome:
        if self.is_single:
            return self.sort_graph_by_innovation()
        else:
            return jax.vmap(Genome.sort_graph_by_innovation)(self)


def init_genome(
    rng: chex.PRNGKey,
    batch_size: int,
    capacity: int,
    input_size: int,
    output_size: int,
    input_activation_ids: List[int],
    output_activation_ids: List[int],
    weight_mean: float,
    weight_std: float,
    mode: str,
    initial_innovation_id: int = 0
) -> Genome:
    """ initializes a genome"""

    assert input_size > 0
    assert output_size > 0
    assert batch_size > 0
    assert capacity >= input_size + output_size

    input_activation_ids = jnp.array(input_activation_ids)
    output_activation_ids = jnp.array(output_activation_ids)

    num_nodes = input_size + output_size
    # inputs
    input_idxs = jnp.arange(input_size)
    # outputs
    output_idxs = jnp.arange(input_size, input_size + output_size)
    # node mask
    node_mask = jnp.zeros(capacity, dtype=jnp.bool_).at[:num_nodes].set(True)
    next_node_idx = input_size + output_size

    graph = init_empty_graph(capacity)
   
    activation_ids = jnp.zeros(capacity, dtype=jnp.int32)
    activation_ids = activation_ids.at[:input_size].set(input_activation_ids)
    activation_ids = activation_ids.at[input_size:num_nodes].set(output_activation_ids)
    biases = jnp.zeros(capacity, dtype=jnp.float32)

    genome = Genome(
        graph=graph,
        input_idxs=input_idxs,
        output_idxs=output_idxs,
        node_mask=node_mask,
        next_conn_idx=jnp.int32(0),
        next_node_idx=jnp.int32(next_node_idx),
        node_activation_ids=activation_ids,
        condensed_size=jnp.int32(0),
        node_biases=biases,
        fitness=jnp.finfo(jnp.float32).min,
        species_id=jnp.int32(0),
        num_initial_connections=jnp.int32(0),
    )
    batch_rng = jax.random.split(rng, batch_size)
    batched_genome = jax.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), genome)
    if mode == 'full':
        num_conns = input_size * output_size
        innovation_ids = jnp.arange(initial_innovation_id, initial_innovation_id + (batch_size * num_conns), step=num_conns)
        batched_genome = apply(batched_genome, Genome.fully_connect, batch_rng, innovation_ids, weight_mean=weight_mean, weight_std=weight_std)
    elif mode == 'partial':
        num_conns = input_size
        innovation_ids = jnp.arange(initial_innovation_id, initial_innovation_id + (batch_size * num_conns), step=num_conns)
        batched_genome = apply(batched_genome, Genome.partially_connect, batch_rng, innovation_ids, weight_mean=weight_mean, weight_std=weight_std)
    elif mode == 'empty':
        pass

    return batched_genome
