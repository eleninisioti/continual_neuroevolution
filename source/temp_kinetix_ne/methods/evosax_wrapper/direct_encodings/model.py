
from typing import NamedTuple, Optional, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from typing import Mapping, NamedTuple, Tuple
import equinox.nn as nn
from flax import linen
import pickle
from methods.Kinetix.kinetix.models import make_network_from_config
from methods.Kinetix.kinetix.models.actor_critic import ScannedRNN
from typing import NamedTuple
import numpy as np

class PolicyState(NamedTuple):
    weights: jax.Array
    adj: jax.Array
    rnn_state: Optional[jax.Array]
    n_dormant: Optional[jax.Array] = jnp.array([0.0])


class MLP(eqx.Module):
    action_dims: int
    obs_dims: int
    mlp: Optional[jax.Array]


    #norm_data: Optional[jax.Array]
    #pretrained_params: Optional[jax.Array]

    def __init__(self,  action_dims, obs_dims, num_layers=2, num_hidden=32, activation="tang", final_activation="tanh", *, key: jax.Array):

        
        if activation == "tanh":
            activation = linen.tanh
        elif activation == "relu":
            activation = linen.relu
        elif activation == "sigmoid":
            activation = linen.sigmoid
        if final_activation == "tanh":
            final_activation = linen.tanh
        elif final_activation == "relu":
            final_activation = linen.relu
        elif final_activation == "sigmoid":
            final_activation = linen.sigmoid
        elif final_activation == "linear":
            final_activation = lambda x: x
            
            
        if final_activation == "tanh":
            final_activation = linen.tanh
        elif final_activation == "relu":
            final_activation = linen.relu
        elif final_activation == "sigmoid":
            final_activation = linen.sigmoid
            
            
        self.action_dims = action_dims
        self.obs_dims = obs_dims
        
        # Create hidden layer sizes list
        hidden_sizes = [num_hidden] * num_layers
        """
        self.mlp = MLPWithDormantTracking(
            input_size=obs_dims,
            output_size=action_dims,
            hidden_sizes=hidden_sizes,
            activation=activation,
            final_activation=final_activation,
            key=key
        )
        """
        self.mlp = nn.MLP(obs_dims,
                    action_dims,
                #16, 2,
                num_hidden, num_layers,
                    #activation=linen.relu, final_activation=linen.tanh,
                    final_activation=final_activation, activation=activation,
                                            #activation=linen.relu, final_activation=lambda x: x,

                key=key, use_bias=True, use_final_bias=True)



    def initialize(self, init_key, current_gen=None):

        # extract weights and biases

        final_policy = PolicyState(weights=jnp.zeros((1,1)), adj=jnp.zeros((1,1)), rnn_state=jnp.zeros((jnp.zeros((1,1)).shape[0],)), n_dormant=jnp.zeros((1,)))
        interm_policies =jax.tree_map(lambda x: x[None,:], final_policy)
        return final_policy, interm_policies



    def get_phenotype(self, mlp):

        # extract weights and biases
        params = mlp.layers

        width = 0
        height = 0
        for el in params:
            kernel = el.weight
            height += kernel.shape[1]
            width += kernel.shape[0]

        weights = jnp.zeros((width, height))
        start_x = 0
        start_y = self.obs_dims
        for el in params:
            kernel = el.weight
            weights = weights.at[start_x:start_x + kernel.shape[1], start_y:start_y + kernel.shape[0]].set(jnp.transpose(el.weight))
            start_x += kernel.shape[1]
            start_y += kernel.shape[0]
        # process bias
        width = 0
        for el in params:
            kernel = el.bias
            width += kernel.shape[0]
        bias = jnp.zeros((width,))
        start_x = 0
        for el in params:
            bias = bias.at[start_x:start_x + el.bias.shape[0]].set(el.bias)
            start_x += el.bias.shape[0]
        adj = jnp.where(weights, 1.0, 0.0)
        final_policy = PolicyState(weights=weights, adj=adj, rnn_state=jnp.zeros((weights.shape[0],)), n_dormant=jnp.zeros((weights.shape[0],)))
        interm_policies =jax.tree_map(lambda x: x[None,:], final_policy)
        return final_policy, interm_policies

    def __call__(self, obs: jax.Array, state: PolicyState, key: jax.Array, obs_size=None, action_size=None) -> Tuple[jax.Array, PolicyState]:
        #jax.debug.print("obs: {}",obs)
        
        #a, dormant_ratio = self.mlp(obs)
        a = self.mlp(obs)
        dormant_ratio=jnp.array([0.0])
        #jax.debug.print("inside model: {}", a)
        
        # Update state with dormant neuron information
        updated_state = state._replace(n_dormant=dormant_ratio)

        return a, updated_state


class MLPWithDormantTracking(eqx.Module):
    """MLP that tracks dormant neurons using Brax logic"""
    layers: list
    activation: callable
    final_activation: callable
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list, 
                 activation=jax.nn.relu, final_activation=lambda x: x, *, key):
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        
        # Create layers
        self.layers = []
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(eqx.nn.Linear(prev_size, hidden_size, key=keys[i]))
            prev_size = hidden_size
        
        # Final layer
        self.layers.append(eqx.nn.Linear(prev_size, output_size, key=keys[-1]))
        
        self.activation = activation
        self.final_activation = final_activation
    
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns (output, dormant_ratio) - copied from Brax MLP logic"""
        hidden = x
        n_dormant = 0 
        n_neurons = 0
        
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if i != len(self.layers) - 1:  # Not the final layer
                hidden = self.activation(hidden)
                
            # Calculate dormant neurons for this layer (Brax logic)
            mean_act = jnp.mean(jnp.abs(hidden))
            per_neuron_act = jnp.abs(hidden) / mean_act
            
            #jax.debug.print("mean_act: {}", mean_act)
            #jax.debug.print("per_neuron_act: {}", per_neuron_act)
                
            n_dormant += jnp.sum(jnp.where(per_neuron_act <= 0.01, 1, 0))
            n_neurons += hidden.size
        
        # Apply final activation
        output = self.final_activation(hidden)
        
        # Calculate dormant ratio
        dormant_ratio = n_dormant / n_neurons
        
        return output, jnp.array([dormant_ratio])


class RNN(eqx.Module):
    action_dims: int
    obs_dims: int
    weights: Optional[jax.Array]
    policy_iters: int
    #norm_data: Optional[jax.Array]
    #pretrained_params: Optional[jax.Array]

    def __init__(self,  action_dims, obs_dims, total_nodes, *, key: jax.Array ):

        self.action_dims = action_dims
        self.obs_dims = obs_dims
        self.policy_iters = 5
        self.weights = jr.normal(key, (total_nodes,
                                       total_nodes))  # self.mlp = nn.MLP(8, action_dims, 32, 4, activation=linen.swish, final_activation=lambda x: x,

    def initialize(self, init_key):

        # extract weights and biases
        weights = self.weights
        adj = jnp.where(weights, 1, 0)
        final_policy = PolicyState(weights=weights,  adj=adj, rnn_state=jnp.zeros((weights.shape[0],)))
        interm_policies =jax.tree_map(lambda x: x[None,:], final_policy)
        return final_policy, interm_policies



    def get_phenotype(self, weights):

        # extract weights and biases
        adj = jnp.where(weights, 1.0, 0.0)
        final_policy = PolicyState(weights=weights,  adj=adj, rnn_state=jnp.zeros((weights.shape[0],)))
        interm_policies =jax.tree_map(lambda x: x[None,:], final_policy)
        return final_policy, interm_policies

    def __call__(self, obs: jax.Array, state: PolicyState, key: jax.Array, obs_size=None, action_size=None) -> Tuple[jax.Array, PolicyState]:
        w = (state.weights)

        def set_input(h):
            h = h.at[0].set(1)
            h = h.at[1:self.obs_dims + 1].set(obs)
            return h

        def rnn_step(h):
            h = set_input(h)
            h = linen.tanh(jnp.matmul(w, h))
            return h

        h = jax.lax.fori_loop(0, self.policy_iters, lambda _, h: rnn_step(h), state.rnn_state)

        a = h[-self.action_dims:]
        state = state._replace(rnn_state=h)
        return a, state


class AtariCNN(eqx.Module):
    """CNN module with three heads for MinAtar games."""
    
    action_dims: int
    obs_dims: int
    conv1: nn.Conv2d
   #conv3: nn.Conv2d
    fc_hidden: nn.Linear
    activation: callable
    #output: nn.Linear

    def __init__(self, action_dims, obs_dims,  *, key: jax.Array):
        self.action_dims = action_dims
        self.obs_dims = obs_dims
        
        # Split key for different layers
        key, conv1_key, conv2_key, conv3_key, fc_key, out_key = jr.split(key, 6)
        
        # Define the layers properly using Equinox
        self.conv1 = nn.Conv2d(obs_dims, 16, kernel_size=3, stride=1, key=conv1_key)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, key=conv3_key)
        
        # Calculate feature size after conv layers
        # For 10x10 input: 10x10 -> 1x1 -> 1x1 -> 1x1 = 1x1x64 = 64 features
        self.fc_hidden = nn.Linear(1024, action_dims, key=fc_key)
        #self.output = nn.Linear(256, action_dims, key=out_key)
        
        self.activation = jnn.relu

    def initialize(self, init_key):
        # For CNN, we need to create a compatible policy state
        # The actual CNN parameters are in the module itself
        # Use a fixed size for the evolutionary framework compatibility
        weights = jnp.zeros((1, 1))  # Fixed size for compatibility
        adj = jnp.zeros((1, 1))
        final_policy = PolicyState(weights=weights, adj=adj, rnn_state=jnp.zeros((1,)))
        interm_policies = jax.tree_map(lambda x: x[None, :], final_policy)
        return final_policy, interm_policies

    def get_phenotype(self, weights):
        # CNN has fixed architecture, so phenotype is the same
        # Use a fixed size for the evolutionary framework compatibility
        adj = jnp.zeros((1, 1))
        final_policy = PolicyState(weights=weights, adj=adj, rnn_state=jnp.zeros((100,)))
        interm_policies = jax.tree_map(lambda x: x[None, :], final_policy)
        return final_policy, interm_policies

    def __call__(self, obs: jax.Array, state: PolicyState, key: jax.Array, obs_size=None, action_size=None) -> Tuple[jax.Array, PolicyState]:
        # Handle MinAtar observations: reshape from (10, 10, 7) to (7, 10, 10) for Conv2d

        def pad_last_dim(x):
            # Current shape
            *rest, last = x.shape
            if last == self.obs_dims:
                return x
            elif last < self.obs_dims:
                #jax.debug.print("padding last dim obs: {}", jnp.sum(x))
                pad_width = [(0, 0)] * (len(x.shape) - 1) + [(0, self.obs_dims - last)]
                new_obs =   jnp.pad(x, pad_width, mode='constant')
                #jax.debug.print("after padding: {}", jnp.sum(new_obs))

                return new_obs
            
            
        
        obs = pad_last_dim(obs)

        """
        
        # If obs size is less than self.obs_dims, append zeros to match the expected dimension
        if obs_size is not None and obs.size < self.obs_dims:
            padding_size = self.obs_dims - obs.size
            obs = jnp.concatenate([obs.reshape(-1), jnp.zeros(padding_size)])
            # Reshape back to original dimensions for the transpose operation
            obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
        """
    
        
        x = jnp.transpose(obs, (2, 0, 1))
        # Apply conv layers
        x = self.activation(self.conv1(x))
        #x = self.activation(self.conv2(x))
        #x = self.activation(self.conv3(x))
        
        # Flatten and apply hidden FC layer
        x = x.reshape(-1)  # Flatten to 1D
        output = self.fc_hidden(x)
        #output = self.output(output)

        
        return output, state


class Kinetix(eqx.Module):
    
    network_params: int
    network: jax.Array
    def __init__(self, key, network, env, env_params):
        self.network = network
        
        # TODO
        _rng = key
        obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, 1), env_params)
        dones = jnp.zeros((1), dtype=jnp.bool_)
        
        init_hstate = ScannedRNN.initialize_carry(1)
        init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
        
        
        self.network_params = network.init(key, init_hstate, init_x)

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.network_params))

        print(param_count)
        
        print("check")
        

    def __call__(self, obs: jax.Array, state: PolicyState, done:jax.Array, key: jax.Array, obs_size=None, action_size=None) -> Tuple[jax.Array, PolicyState]:
        _rng = jax.random.PRNGKey(0)
        _rng = key
        obs = jax.tree.map(lambda x: x[np.newaxis, :], obs)
        ac_in = (jax.tree.map(lambda x: x[np.newaxis, :], obs), done[np.newaxis, :])

        hstate, pi = self.network.apply(self.network_params, state.rnn_state, ac_in)
        action = pi.sample(seed=_rng)
        action = action[0,0,:]
        state = state._replace(rnn_state=hstate)
        return action, state


    
    def initialize(self, init_key):
        init_hstate = ScannedRNN.initialize_carry(1)

        return PolicyState(weights=jnp.zeros((1,1)), adj=jnp.zeros((1,1)), rnn_state=init_hstate), jax.tree_map(lambda x: x[None,:], PolicyState(weights=jnp.zeros((1,1)), adj=jnp.zeros((1,1)), rnn_state=jnp.zeros((1,1))))


        
        
        
        


def make_model(config, key, env=None, env_params=None, kinetix_config=None):
    """ Creates a direct encoding
    """

    key, key_model = jr.split(key)

    action_size = config["env_config"]["action_size"]
    input_size = config["env_config"]["observation_size"]


    if config["model_config"]["network_type"] == "MLP":

        model = MLP( key=key_model,
                     obs_dims=input_size,
                     action_dims=action_size,
                     num_layers=config["model_config"]["model_params"]["num_layers"],
                     num_hidden=config["model_config"]["model_params"]["num_hidden"],
                     activation=config["model_config"]["model_params"]["activation"],
                     final_activation=config["model_config"]["model_params"]["final_activation"],
                    )
    elif config["model_config"]["network_type"] == "RNN":
        max_nodes = action_size + input_size + config["model_config"]["model_params"]["max_hidden_neurons"] + 1

        model = RNN(key=key_model,
                    obs_dims=input_size,
                    action_dims=action_size,
                    total_nodes=max_nodes,
                    )
    elif config["model_config"]["network_type"] == "AtariCNN":

        model = AtariCNN(key=key_model,
                    obs_dims=input_size,
                    action_dims=action_size,
                        )
        
    elif config["model_config"]["network_type"] == "kinetix":
        network = make_network_from_config(env, env_params, kinetix_config)
        model = Kinetix(network=network, key=key_model, env=env, env_params=env_params)

    return model
