# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""D4PG agent implementation."""

import copy
import dataclasses
from typing import Iterator, List, Optional, Tuple

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as reverb_adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.d4pg import learning
from acme.tf import networks as network_utils
from acme.tf import utils
from acme.tf import variable_utils
from acme.utils import counting
from acme.utils import loggers

import dm_env
import reverb
import sonnet as snt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
import agent.learning as learning

from absl import flags
FLAGS = flags.FLAGS

class KernelLayer(snt.Module):
    """
    Trainable 1D kernel that shares weights but can compute exposures
    from different anchor points on the tenor grid.

    Usage:
        layer = KernelLayer(
            tenor_grid=None,
            init_params=None,
            a_scale=..., b_scale=..., c_scale=..., d_scale=...,
            eta_scale=..., beta_scale=...,
            default_anchor_index=52)
        # Compute exposures with shared parameters at any anchor:
        exposure_1y = layer(risk_vector)                  # uses default anchor
        exposure_6m = layer(risk_vector, anchor_index=26)  # override anchor
    """
    def __init__(self,
                 tenor_grid: np.ndarray = None,
                 init_params: dict = None,
                 a_scale: float = 1.0,
                 b_scale: float = 1.0,
                 c_scale: float = 1.0,
                 d_scale: float = 1.0,
                 eta_scale: float = 0.95,
                 beta_scale: float = 0.5,
                 default_anchor_index: int = 52,
                 name: str = None):
        super().__init__(name=name)
        # Build tenor grid: default 0 to 2 years at 1/52 increments
        if tenor_grid is None:
            tau_vals = np.arange(0, 2 + 1e-3, 1/52).astype(np.float32)
        else:
            tau_vals = np.array(tenor_grid, dtype=np.float32)
        self.tau_grid = tf.constant(tau_vals, dtype=tf.float32)
        self.default_anchor = default_anchor_index

        # Default initial params
        defaults = init_params or {
            'a': a_scale*0.5, 'b': b_scale*0.5, 'c': c_scale*0.5, 'd':d_scale* 0.5,
            'eta': eta_scale*0.5, 'beta': beta_scale*0.5
        }
        def make_bij(scale, shift=0.0):
            chain = []
            if shift != 0.0:
                chain.append(tfb.Shift(shift))
            chain.append(tfb.Scale(scale))
            chain.append(tfb.Sigmoid())
            return tfb.Chain(chain)

        # Trainable parameters with smooth (0,1) constraint then scaled
        self.a = tfp.util.TransformedVariable(
            defaults['a'], bijector=make_bij(a_scale), name='a')
        self.b = tfp.util.TransformedVariable(
            defaults['b'], bijector=make_bij(b_scale), name='b')
        self.c = tfp.util.TransformedVariable(
            defaults['c'], bijector=make_bij(c_scale), name='c')
        self.d = tfp.util.TransformedVariable(
            defaults['d'], bijector=make_bij(d_scale, shift=0.001), name='d')
        self.eta = tfp.util.TransformedVariable(
            defaults['eta'], bijector=make_bij(eta_scale), name='eta')
        self.beta = tfp.util.TransformedVariable(
            defaults['beta'], bijector=make_bij(beta_scale, shift=0.001), name='beta')

    def _weights_for_anchor(self, anchor_index: int) -> tf.Tensor:
        """
        Compute the kernel weight vector [T] for a given anchor index.
        """
        tau = self.tau_grid
        vols = (self.a + self.b * tau) * tf.exp(-self.c * tau) + self.d
        vol_ref = tf.gather(vols, anchor_index)
        tau_ref = tf.gather(tau, anchor_index)
        time_dist = tf.abs(tau - tau_ref)
        corr = self.eta + (1. - self.eta) * tf.exp(-self.beta * time_dist)
        return corr * (vols / vol_ref)
    @tf.function(jit_compile=True)
    def __call__(self,
                 risk_vector: tf.Tensor,
                 anchor_index: int = None) -> tf.Tensor:
        """
        Args:
          risk_vector: Tensor [B, T] or [T]
          anchor_index: optional override of default anchor position
        Returns:
          exposure: same leading batch dims, sum(weights · risk_vector)
        """
        # Determine anchor to use
        idx = anchor_index if anchor_index is not None else self.default_anchor
        # Compute weight vector for this anchor
        weights = self._weights_for_anchor(idx)  # [T]
        # Inner product along last dim
        return tf.reduce_sum(tf.cast(risk_vector,tf.float32) * weights, axis=-1)
    
    def weight_single(self, source_idx: int, *, target_idx: Optional[int] = None) -> tf.Tensor:
        """Convenience: K(target_idx, source_idx)."""
        tgt = target_idx if target_idx is not None else self.default_anchor


        return self._weights_for_anchor(tgt)[source_idx]

class ObservationWithKernel(snt.Module):
    """
    Sonnet module that applies shared KernelLayer transforms to raw observations.
    Expects input tensor of shape [..., N], where:
      - indices 0:4       = base features [swaption_gamma, swaption_vega, swap_delta_hed, swap_delta_liab]
      - indices 4:7       = three additional base features
      - indices 7:7+105   = gamma_vector
      - indices 7+105:7+210 = vega_vector
      - indices 7+210:     = delta_vector

    Returns tensor of shape [..., 7]:
      [gamma_ratio, vega_ratio, delta_ratio_hed, delta_ratio_liab, base_features]
    """
    def __init__(self,
                 vol_kernel: KernelLayer,
                 volvol_kernel: KernelLayer,
                 default_anchor_index: int = np.int32(52),
                 name: str = None):
        super().__init__(name=name)
        self.vol_kernel = vol_kernel
        self.volvol_kernel = volvol_kernel
        self.default_anchor = default_anchor_index
        self.anchor_hed = int(default_anchor_index)
        self.anchor_liab = int(default_anchor_index * 2)
    @tf.function(jit_compile=True)
    def __call__(self, observation: tf.Tensor) -> tf.Tensor:
        # 1) base features
        swaption_gamma = observation[..., 0]
        swaption_vega  = observation[..., 1]
        swap_delta_hed = observation[..., 2]
        swap_delta_liab = observation[..., 3]
        base_features = observation[..., 4:7]  # [...,3]

        # 2) greek block
        greek_block = observation[..., 7:]
        gamma_vector = greek_block[..., :105]
        vega_vector  = greek_block[..., 105:210]
        delta_vector = greek_block[..., 210:]

        # 3) compute hedge ratios
        gamma_ratio = tf.math.divide_no_nan(
            self.vol_kernel(gamma_vector), swaption_gamma)
        vega_ratio = tf.math.divide_no_nan(
            self.volvol_kernel(vega_vector), swaption_vega)
        delta_ratio_hed = tf.math.divide_no_nan(
            self.vol_kernel(delta_vector, anchor_index=self.default_anchor),
            swap_delta_hed)
        delta_ratio_liab = tf.math.divide_no_nan(
            self.vol_kernel(delta_vector,
                             anchor_index=self.default_anchor * 2),
            swap_delta_liab)
        # cross‑kernel entries -------------------------------------------
        k12_raw = self.vol_kernel.weight_single(self.anchor_hed, target_idx=self.anchor_liab)  # K(1y→2y)
        k21_raw = self.vol_kernel.weight_single(self.anchor_liab, target_idx=self.anchor_hed)  # K(2y→1y)
        k12 = tf.math.divide_no_nan(k12_raw, swap_delta_hed)  # effect in 1y-delta units per 2y swap notional
        k21 = tf.math.divide_no_nan(k21_raw, swap_delta_liab)  # effect in 2y-delta units per 1y swap notional
        # 4) expand dims
        gamma_ratio   = tf.expand_dims(gamma_ratio, -1)
        vega_ratio    = tf.expand_dims(vega_ratio,  -1)
        delta_ratio_hed   = tf.expand_dims(delta_ratio_hed,  -1)
        delta_ratio_liab  = tf.expand_dims(delta_ratio_liab,  -1)
        # 5) concatenate output
        return tf.concat([
            gamma_ratio,
            vega_ratio,
            delta_ratio_hed,
            delta_ratio_liab,
            k12,
            k21,
            base_features, 
        ], axis=-1)  # [...,7]


class PolicyWithHedge(snt.Module):
    def __init__(self,  base_policy: snt.Module, name=None):
        super().__init__(name=name)
        self.base_pol = base_policy
        self.anchor_hed = 52
        self.anchor_liab = self.anchor_hed * 2

    def _static_delta_opt(self, delta_ratio_hed, delta_ratio_liab, k12, k21):
        S = tf.stack([[1,k12], [k21,1]])
        d= tf.stack([[delta_ratio_hed],[delta_ratio_liab]])
        obt_bounds = tf.linalg.solve(S,d)
        return obt_bounds



    def __call__(self, obs):

        # 2) Unpack first 4 features
        gamma_ratio, vega_ratio, delta_h, delta_l = tf.unstack(obs[..., :4], axis=-1)

        # 3) Actor outputs two scalars per example
        action_mag, action_dir = tf.unstack(self.base_pol(obs), axis=-1)

        # 4) Compute the final hedge
        applied_hedge = action_mag * ((1 - action_dir) * gamma_ratio
                                    +    action_dir  * vega_ratio)
        
        # 5) Stack everything on the last axis
        action = tf.stack([
        action_mag,
        action_dir,
        gamma_ratio,
        vega_ratio,
        -applied_hedge,
        -delta_h,
        -delta_l
        ], axis=-1)
        
        #tf.print(action[...,:4])
        #tf.print(action[...,4:],"\n")
        return action


@dataclasses.dataclass
class D4PGConfig:
    """Configuration options for the D4PG agent."""
    obj_func: str = 'var'
    critic_loss_type: str = 'c51'
    threshold: float = 0.95
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    policy_optimizer: Optional[snt.Optimizer] = None
    critic_optimizer: Optional[snt.Optimizer] = None
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: Optional[float] = 32.0
    n_step: int = 20
    sigma: float = 0.3
    clipping: bool = True
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


@dataclasses.dataclass
class D4PGNetworks:
    """Structure containing the networks for D4PG."""

    policy_network: snt.Module
    critic_network: snt.Module
    observation_network: snt.Module

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: snt.Module,
        observation_network: types.TensorTransformation,
    ):
        # This method is implemented (rather than added by the dataclass decorator)
        # in order to allow observation network to be passed as an arbitrary tensor
        # transformation rather than as a snt Module.
        # TODO(mwhoffman): use Protocol rather than Module/TensorTransformation.
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.observation_network = utils.to_sonnet_module(observation_network)

    def init(self, environment_spec: specs.EnvironmentSpec):
        """Initialize the networks given an environment spec."""
        # Get observation and action specs.
        act_spec = environment_spec.actions
        obs_spec = environment_spec.observations

        # Create variables for the observation net and, as a side-effect, get a
        # spec describing the embedding space.
        emb_spec = utils.create_variables(self.observation_network, [obs_spec])

        # Create variables for the policy and critic nets.
        _ = utils.create_variables(self.policy_network, [emb_spec])
        _ = utils.create_variables(self.critic_network, [emb_spec, act_spec])

    def make_policy(
        self,
        environment_spec: specs.EnvironmentSpec,
        sigma: float = 0.0,
    ) -> snt.Module:
        """Create a single network which evaluates the policy."""
        # Stack the observation and policy networks.
        stack = [
            self.observation_network,
            self.policy_network,
        ]
  

        # If a stochastic/non-greedy policy is requested, add Gaussian noise on
        # top to enable a simple form of exploration.
        # TODO(mwhoffman): Refactor this to remove it from the class.
        if sigma > 0.0:
           stack += [
               network_utils.ClippedGaussian(sigma),
               network_utils.ClipToSpec(environment_spec.actions),
           ]

        # Return a network which sequentially evaluates everything in the stack.
        return snt.Sequential(stack)


class D4PGBuilder:
    """Builder for D4PG which constructs individual components of the agent."""

    def __init__(self, config: D4PGConfig):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(
                self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer)

        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.NStepTransitionAdder.signature(
                environment_spec))

        return [replay_table]

    def make_dataset_iterator(
        self,
        reverb_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=reverb_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size)

        # TODO(b/155086959): Fix type stubs and remove.
        return iter(dataset)  # pytype: disable=wrong-arg-types

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> adders.Adder:
        """Create an adder which records data generated by the actor/environment."""
        return reverb_adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount)

    def make_actor(
        self,
        policy_network: snt.Module,
        adder: Optional[adders.Adder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ):
        """Create an actor instance."""
        if variable_source:
            # Create the variable client responsible for keeping the actor up-to-date.
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={'policy': policy_network.variables},
                update_period=1000,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        else:
            variable_client = None

        # Create the actor which defines how we take actions.
        return actors.FeedForwardActor(
            policy_network=policy_network,
            adder=adder,
            variable_client=variable_client,
        )

    def make_learner(
        self,
        networks: Tuple[D4PGNetworks, D4PGNetworks],
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = False,
        checkpoint_folder: str = '~/acme'
    ):
        """Creates an instance of the learner."""
        online_networks, target_networks = networks

        # The learner updates the parameters (and initializes them).
        return learning.D4PGLearner(
            obj_func=self._config.obj_func,
            critic_loss_type=self._config.critic_loss_type,
            threshold=self._config.threshold,
            policy_network=online_networks.policy_network,
            critic_network=online_networks.critic_network,
            observation_network=online_networks.observation_network,
            target_policy_network=target_networks.policy_network,
            target_critic_network=target_networks.critic_network,
            target_observation_network=target_networks.observation_network,
            policy_optimizer=self._config.policy_optimizer,
            critic_optimizer=self._config.critic_optimizer,
            clipping=self._config.clipping,
            discount=self._config.discount,
            target_update_period=self._config.target_update_period,
            dataset_iterator=dataset,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_folder=checkpoint_folder,
        )


class D4PG(agent.Agent):
    """D4PG Agent.
    This implements a single-process D4PG agent. This is an actor-critic algorithm
    that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policy (and as a result the
    behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy_network: snt.Module,
        critic_network: snt.Module,
        obj_func='var',
        critic_loss_type='c51',
        threshold=0.95,
        observation_network: types.TensorTransformation = tf.identity,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: float = 32.0,
        n_step: int = 5,
        sigma: float = 0.3,
        clipping: bool = True,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        checkpoint_folder: str = '~/acme'
    ):
        """Initialize the agent.
        Args:
          environment_spec: description of the actions, observations, etc.
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          observation_network: optional network to transform the observations before
            they are fed into any network.
          discount: discount to use for TD updates.
          batch_size: batch size for updates.
          prefetch_size: size to prefetch from replay.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          policy_optimizer: optimizer for the policy network updates.
          critic_optimizer: optimizer for the critic network updates.
          min_replay_size: minimum replay size before updating.
          max_replay_size: maximum replay size.
          samples_per_insert: number of samples to take from replay for every insert
            that is made.
          n_step: number of steps to squash into a single transition.
          sigma: standard deviation of zero-mean, Gaussian exploration noise.
          clipping: whether to clip gradients by global norm.
          replay_table_name: string indicating what name to give the replay table.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """
        # Create the Builder object which will internally create agent components.
        builder = D4PGBuilder(
            # TODO(mwhoffman): pass the config dataclass in directly.
            # TODO(mwhoffman): use the limiter rather than the workaround below.
            # Right now this modifies min_replay_size and samples_per_insert so that
            # they are not controlled by a limiter and are instead handled by the
            # Agent base class (the above TODO directly references this behavior).
            D4PGConfig(
                obj_func=obj_func,
                critic_loss_type=critic_loss_type,
                threshold=threshold,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                min_replay_size=1,  # Let the Agent class handle this.
                max_replay_size=max_replay_size,
                samples_per_insert=None,  # Let the Agent class handle this.
                n_step=n_step,
                sigma=sigma,
                clipping=clipping,
                replay_table_name=replay_table_name,
            ))
        #tf.print("policy internal network dimensions: ", policy_network)
        #tf.print("observation network internal dimensions:", observation_network)
        # TODO(mwhoffman): pass the network dataclass in directly.
        online_networks = D4PGNetworks(policy_network=policy_network,
                                       critic_network=critic_network,
                                       observation_network=observation_network)

        # Target networks are just a copy of the online networks.
        target_networks = copy.deepcopy(online_networks)

        # Initialize the networks.
        online_networks.init(environment_spec)
        target_networks.init(environment_spec)

        # TODO(mwhoffman): either make this Dataclass or pass only one struct.
        # The network struct passed to make_learner is just a tuple for the
        # time-being (for backwards compatibility).
        networks = (online_networks, target_networks)

        # Create the behavior policy.
        policy_network = online_networks.make_policy(environment_spec, sigma)

        # Create the replay server and grab its address.
        replay_tables = builder.make_replay_tables(environment_spec)
        replay_server = reverb.Server(replay_tables, port=None)
        replay_client = reverb.Client(f'localhost:{replay_server.port}')

        # Create actor, dataset, and learner for generating, storing, and consuming
        # data respectively.
        adder = builder.make_adder(replay_client)
        actor = builder.make_actor(policy_network, adder)
        dataset = builder.make_dataset_iterator(replay_client)
        learner = builder.make_learner(networks, dataset, counter, logger,
                                       checkpoint, checkpoint_folder)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert)

        # Save the replay so we don't garbage collect it.
        self._replay_server = replay_server


class VegaHedgeAgent(core.Actor):
    '''
    This is the Delta-Vega Agent implementation. 
    Output: Hedging Actions - Alpha, computed analytically following the alpha approach defined in the paper.
    '''
    def __init__(self, running_env) -> None:
        self.env = running_env
        super().__init__()
    
    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        episode = self.env.sim_episode
        t = self.env.t
        current_vega = observation[3]
        hedge_option = self.env.portfolio.hed_port.options[episode,t]
        hed_share = -current_vega/hedge_option.vega_path[t]/self.env.portfolio.utils.contract_size
        # action constraints
        gamma_action_bound = -self.env.portfolio.get_gamma(t)/self.env.portfolio.hed_port.options[episode, t].gamma_path[t]/self.env.portfolio.utils.contract_size
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]
        
        if FLAGS.vega_obs:
            # vega bounds
            vega_action_bound = -self.env.portfolio.get_vega(t)/self.env.portfolio.hed_port.options[episode, t].vega_path[t]/self.env.portfolio.utils.contract_size
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        low_val = np.min(action_low)
        high_val = np.max(action_high)

        alpha = (hed_share - low_val)/(high_val - low_val)
        
        return np.array([alpha])

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def update(self, wait: bool = False):
        pass

class GammaHedgeAgent(core.Actor):
    """
    This is the baseline-Delta Gamma Agent implementation.
    Output: Hedging Actions - Alpha, computed analytically following the alpha approach defined in the paper.
    """
    def __init__(self, running_env, hedge_ratio=1.0) -> None:
        self.env = running_env
        self.hedge_ratio = hedge_ratio
        super().__init__()
    
    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        episode = self.env.sim_episode
        t = self.env.t
        current_gamma = observation[1]
        
        hedge_gamma = self.hedge_ratio*current_gamma
        hedge_option = self.env.portfolio.hed_port.options[episode,t]
        hed_share = -hedge_gamma/hedge_option.gamma_path[t]/self.env.portfolio.utils.contract_size
        # action constraints
        gamma_action_bound = -self.env.portfolio.get_gamma(t)/self.env.portfolio.hed_port.options[episode, t].gamma_path[t]/self.env.portfolio.utils.contract_size
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]
        
        if FLAGS.vega_obs:
            # vega bounds
            vega_action_bound = -self.env.portfolio.get_vega(t)/self.env.portfolio.hed_port.options[episode, t].vega_path[t]/self.env.portfolio.utils.contract_size
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        low_val = np.min(action_low)
        high_val = np.max(action_high)

        alpha = (hed_share - low_val)/(high_val - low_val)
        
        return np.array([alpha])

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def update(self, wait: bool = False):
        pass

class DeltaHedgeAgent(core.Actor):
    """
    This is the baseline Delta Heging agent implementation
    Output: Hedging Actions - Alpha, computed analytically following the alpha approach defined in the paper.
    """
    def __init__(self,running_env, hedge_ratio=1.0) -> None:
        self.env = running_env
        self.hedge_ratio = hedge_ratio
        super().__init__()
    
    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        episode = self.env.sim_episode
        t = self.env.t
        
        hed_share = 0
        # action constraints
        gamma_action_bound = -self.env.portfolio.get_gamma(t)/self.env.portfolio.hed_port.options[episode, t].gamma_path[t]/self.env.portfolio.utils.contract_size
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]
        
        if FLAGS.vega_obs:
            # vega bounds
            vega_action_bound = -self.env.portfolio.get_vega(t)/self.env.portfolio.hed_port.options[episode, t].vega_path[t]/self.env.portfolio.utils.contract_size
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        low_val = np.min(action_low)
        high_val = np.max(action_high)

        alpha = (hed_share - low_val)/(high_val - low_val)
        
        return np.array([alpha])

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def update(self, wait: bool = False):
        pass
