import os
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import tensorflow as tf
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.tf.savers import make_snapshot
import acme.utils.loggers as log_utils
import dm_env
import numpy as np
import sonnet as snt
import pandas as pd
from environment.Environment import TradingEnv
from environment.utils import Utils
import agent.distributional as ad

from absl import app
from absl import flags
from collections import OrderedDict
import dataclasses
import datetime

def ordered_asdict(obj):
    return OrderedDict((f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj))

print("SETTING FLOAT PRECISION TO 32")

tf.keras.backend.set_floatx('float32')

FLAGS = flags.FLAGS
flags.DEFINE_integer('train_sim', 40_000, 'train episodes (Default 40_000)')
flags.DEFINE_integer('eval_sim', 5_000, 'evaluation episodes (Default 40_000)')
flags.DEFINE_integer('init_ttm', 60, 'number of days in one episode (Default 60)')
flags.DEFINE_float('mu', 0.0, 'spot drift (Default 0.2)')
flags.DEFINE_integer('n_step', 5, 'DRL TD Nstep (Default 5)')
flags.DEFINE_float('init_vol', 0.2, 'initial spot vol (Default 0.2)')
flags.DEFINE_float('poisson_rate', 1.0, 'possion rate of new optiosn in liability portfolio (Default 1.0)')
flags.DEFINE_float('moneyness_mean', 1.0, 'new optiosn moneyness mean (Default 1.0)')
flags.DEFINE_float('moneyness_std', 0.0, 'new optiosn moneyness std (Default 0.0)')
flags.DEFINE_string('critic', 'c51', 'critic distribution type - c51, qr-huber, qr-gl, qr-gl_tl, '
                                     'qr-lapl, qr-lapl_tl, iqn-huber (Default c51)')
flags.DEFINE_float('spread', 0.0, 'Hedging transaction cost (Default 0.0)')
flags.DEFINE_string('obj_func', 'var', 'Objective function select from meanstd, var or cvar (Default var)')
flags.DEFINE_float('std_coef', 1.645, 'Std coefficient when obj_func=meanstd. (Default 1.645)')
flags.DEFINE_float('threshold', 0.95, 'Objective function threshold. (Default 0.95)')
flags.DEFINE_float('vov', 0.0, 'Vol of vol, zero means BSM; non-zero means SABR (Default 0.0)')
flags.DEFINE_list('liab_ttms',['60',], 'List of maturities selected for new adding option (Default [60,])')
flags.DEFINE_integer('hed_ttm', 20, 'Hedging option maturity in days (Default 20)')
flags.DEFINE_list('action_space', ['0','3'], 'Hedging action space (Default [0,3])')
flags.DEFINE_string('logger_prefix', '', 'Prefix folder for logger (Default None)')
flags.DEFINE_string('agent_path', '', 'trained agent path, only used when eval_only=True')
flags.DEFINE_boolean('eval_only', False, 'Ignore training (Default False)')
flags.DEFINE_boolean('per', False, 'Use PER for Replay sampling (Default False)')
flags.DEFINE_float('lr', 1e-4, 'Learning rate for optimizer (Default 1e-4)')
flags.DEFINE_integer('batch_size', 256, 'Batch size to train the Network (Default 256)')
flags.DEFINE_float('priority_exponent', 0.6, 'priority exponent for the Prioritized replay table (Default 0.6)')
flags.DEFINE_float('importance_sampling_exponent', 0.2, 'importance sampling exponent for updating importance weight for PER (Default 0.2)')
flags.DEFINE_boolean('vega_obs', False, 'Include portfolio vega and hedging option vega in state variables (Default False)')
flags.DEFINE_integer('eval_seed', 1234, 'Evaluation Seed (Default 1234)')
flags.DEFINE_boolean('gbm', False, 'GBM (Default False)')
flags.DEFINE_boolean('sabr', False, 'SABR (Default False)')

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = f"/run_{TS}"
def make_logger(work_folder, label, terminal=False):
    
    loggers = [
        log_utils.CSVLogger(f'./logs/{RUN_FOLDER}/{work_folder}', label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label))
    
    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    #loggers = log_utils.TimeFilter(logger, 1.0)
    return logger

def make_loggers(work_folder):
    return dict(
        train_loop=make_logger(work_folder, 'train_loop', terminal=True),
        eval_loop=make_logger(work_folder, 'eval_loop', terminal=True),
        learner=make_logger(work_folder, 'learner')
    )

def make_environment(utils, logger = None) -> dm_env.Environment:
    # Make sure the environment obeys the dm_env.Environment interface.
    environment = wrappers.GymWrapper(TradingEnv(
    utils=utils, 
    logger=logger))
    # Clip the action returned by the agent to the environment spec.
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)

    return environment




def sinusoidal_time_embedding(
    t_years: tf.Tensor,
    dim: int,
    base: float = 100.0,
    scale: float = 2 * np.pi
) -> tf.Tensor:
    """
    t_years: [...,1] or [...] float32 tensor in [0,1] (fraction of year)
    dim:     even integer, total embedding size
    base:    controls decay of frequencies
    scale:   overall multiplier (2π to wrap 1 cycle over t=1)
    returns: [..., dim] float32 tensor
    """
    # make sure we have shape [...], not [...,1]
    t = tf.cast(tf.squeeze(t_years, axis=-1), tf.float32)  # now shape [...]
    half = dim // 2
    # build [half] frequency vector
    i = tf.cast(tf.range(half), tf.float32)
    freqs = tf.pow(base, -2.0 * i / tf.cast(dim, tf.float32))  # [half]
    # angles: broadcast to [..., half]
    angles = tf.expand_dims(t, -1) * scale * freqs
    sin_emb = tf.sin(angles)  # [..., half]
    cos_emb = tf.cos(angles)  # [..., half]
    emb = tf.concat([sin_emb, cos_emb], axis=-1)  # [..., dim]
    return tf.cast(emb, tf.float32)

def embed_time_observation(obs: tf.Tensor, time_dim: int = 16, dt: float = 1/52) -> tf.Tensor:
    # Split off the last element as the integer time index
    core  = obs[..., :-1]               # all features except t
    t_raw = obs[..., -1:]               # shape [...,1]

    # Convert step index to continuous years
    t_cont = tf.cast(t_raw, tf.float32) * dt  # now in [0, total_years]

    # Sinusoidal embed into `time_dim` dims
    t_embed = sinusoidal_time_embedding(t_cont, time_dim)

    # Final observation vector
    return tf.cast(tf.concat([core, t_embed], axis=-1), tf.float32)

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Alias for bijectors
tfb = tfp.bijectors

class OneYearKernelLayer(snt.Module):
    """
    Sonnet module acting like a layer: takes a risk vector [T] and returns
    a single exposure scalar. Internally computes trainable kernel weights
    anchored at a fixed tenor (1y at index anchor_index).

    Parameters a,b,c,d,eta,beta are constrained smoothly via Sigmoid bijectors,
    with optional scaling ranges provided via init args.

    Usage:
        layer = OneYearKernelLayer(
            anchor_index=52,
            a_scale=0.2, b_scale=0.2, c_scale=1.5, d_scale=0.5,
            eta_scale=0.95, beta_scale=0.5)
        exposure = layer(risk_vector)
    """
    def __init__(self,
                 anchor_index: int,
                 tenor_grid: np.ndarray = None,
                 init_params: dict = None,
                 a_scale: float = 1.0,
                 b_scale: float = 1.0,
                 c_scale: float = 1.0,
                 d_scale: float = 1.0,
                 eta_scale: float = 1.0,
                 beta_scale: float = 1.0,
                 name: str = None):
        super().__init__(name=name)
        self.anchor_index = anchor_index
        # Tenor grid default 0 to 3 years at 1/52 increments
        if tenor_grid is None:
            tau_vals = np.arange(0, 3, 1/52).astype(np.float32)
        else:
            tau_vals = np.array(tenor_grid, dtype=np.float32)
        self.tau_grid = tf.constant(tau_vals, dtype=tf.float32)

        # Default initial params
        defaults = init_params or {
            'a': 0.5, 'b': 0.5, 'c': 0.5, 'd': 0.001,
            'eta': 0.5, 'beta': 0.5
        }
        # Build bijectors: sigmoid then scale
        def make_bij(scale, shift=0.0):
            chain = []
            if shift != 0.0:
                chain.append(tfb.Shift(shift))
            # scale after sigmoid to map (0,1) -> (0,scale)
            chain.append(tfb.Scale(scale))
            chain.append(tfb.Sigmoid())
            return tfb.Chain(chain)

        # Create trainable TransformedVariables
        self.a = tfp.util.TransformedVariable(
            defaults['a'], bijector=make_bij(a_scale), name='a')
        self.b = tfp.util.TransformedVariable(
            defaults['b'], bijector=make_bij(b_scale), name='b')
        self.c = tfp.util.TransformedVariable(
            defaults['c'], bijector=make_bij(c_scale), name='c')
        # For d, ensure minimum base level via shift
        self.d = tfp.util.TransformedVariable(
            defaults['d'], bijector=make_bij(d_scale, shift=0.001), name='d')
        self.eta = tfp.util.TransformedVariable(
            defaults['eta'], bijector=make_bij(eta_scale), name='eta')
        self.beta = tfp.util.TransformedVariable(
            defaults['beta'], bijector=make_bij(beta_scale), name='beta')

    def _compute_weights(self) -> tf.Tensor:
        tau = self.tau_grid
        vols = (self.a + self.b * tau) * tf.exp(-self.c * tau) + self.d
        vol_ref = vols[self.anchor_index]
        tau_ref = tau[self.anchor_index]
        time_dist = tf.abs(tau - tau_ref)
        corr = self.eta + (1. - self.eta) * tf.exp(-self.beta * time_dist)
        return corr * (vols / vol_ref)

    def __call__(self, risk_vector: tf.Tensor) -> tf.Tensor:
        weights = self._compute_weights()
        return tf.reduce_sum(weights * risk_vector)



# The default settings in this network factory will work well for the
# TradingENV task but may need to be tuned for others. In
# particular, the vmin/vmax and num_atoms hyperparameters should be set to
# give the distributional critic a good dynamic range over possible discounted
# returns. Note that this is very different than the scale of immediate rewards.






def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
    max_time_steps: int = 100,  # Maximum number of time steps in an episode
    time_embedding_dim: int = 16,  # Dimension of the time embedding
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)



    # Create a time embedding layer.
    time_embedding_layer = tf.keras.layers.Embedding(input_dim=max_time_steps, output_dim=time_embedding_dim)
    
    # Create the shared observation network.
    def observation_with_time(obs):
        # Assume the last dimension of the observation is the time step (integer).
        features, time_step = obs[..., :-1], tf.cast(obs[..., -1], tf.int32)
        time_embedding = time_embedding_layer(time_step)
        return tf.concat([features, time_embedding], axis=-1)
    
    observation_network = observation_with_time
    # Create the shared observation network; here simply a state-less operation.
    #observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])

    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.RiskDiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }

def make_quantile_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256), # (64, 64)
    #policy_layer_sizes: Sequence[int] = (16, 32),
    critic_layer_sizes: Sequence[int] =  (512, 512, 256),
    #critic_layer_sizes: Sequence[int] =  (32, 32),
    quantile_interval: float = 0.01, 
    ) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    vol_kernel = OneYearKernelLayer(a_scale=0.1, b_sclae=0.3, c_scale=1.5, d_scale = 0.1)
    volvol_kernel = OneYearKernelLayer(a_scale=1.5, b_scale=0.3, c_scale=3, d_scale = 0.5)

    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.QuantileDiscreteValuedHead(quantiles=quantiles, prob_type=ad.QuantileDistProbType.MID),
    ])
    
    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }

def make_iqn_networks(
    action_spec: specs.BoundedArray,
    cvar_th: float,
    n_cos=64, n_tau=8, n_k=32,
    policy_layer_sizes: Sequence[int] = (64, 64), # (256, 256, 256)
    critic_layer_sizes: Sequence[int] = (128, 128), # (512, 512, 256)
    quantile_interval: float = 0.01
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
    # Create the critic network.
    critic_network = ad.IQNCritic(cvar_th, n_cos, n_tau, n_k, critic_layer_sizes, quantiles, ad.QuantileDistProbType.MID)
    
    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }

def save_policy(policy_network, checkpoint_folder):
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = make_snapshot(policy_network)
    save_path = f"{checkpoint_folder}/policy_{timestamp}"
    tf.saved_model.save(snapshot, save_path)
    print(f"Policy saved to {save_path}")

def load_policy(policy_network, checkpoint_folder):
    trainable_variables_snapshot = {}
    load_net = tf.saved_model.load(checkpoint_folder+'/policy')
    for var in load_net.trainable_variables:
        trainable_variables_snapshot['/'.join(
            var.name.split('/')[1:])] = var.numpy()
    for var in policy_network.trainable_variables:
        var_name_wo_name_scope = '/'.join(var.name.split('/')[1:])
        var.assign(
            trainable_variables_snapshot[var_name_wo_name_scope])

def main(argv):
    
    if FLAGS.per == True:
      from agent_per.agent_per import D4PG
    else:
      from agent.agent import D4PG

    # work_folder = f'spread={FLAGS.spread}_obj={FLAGS.obj_func}_threshold={FLAGS.threshold}_critic={FLAGS.critic}_v={FLAGS.vov}_hedttm={FLAGS.hed_ttm}_elastic_reward_k={FLAGS.elastic_reward_k}'
    work_folder = f'spread={FLAGS.spread}_obj={FLAGS.obj_func}_threshold={FLAGS.threshold}_critic={FLAGS.critic}_v={FLAGS.vov}_hedttm={FLAGS.hed_ttm}'
    if FLAGS.logger_prefix:
        work_folder = FLAGS.logger_prefix + "/" + work_folder
    # Create an environment, grab the spec, and use it to create networks.
    # utils = Utils(init_ttm=FLAGS.init_ttm, np_seed=1234, num_sim=FLAGS.train_sim, spread=FLAGS.spread, volvol=FLAGS.vov, sabr=FLAGS.sabr, gbm=FLAGS.gbm, hed_ttm=FLAGS.hed_ttm,
    #               init_vol=FLAGS.init_vol, poisson_rate=FLAGS.poisson_rate, 
    #               moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std, 
    #               mu=FLAGS.mu, ttms=[int(ttm) for ttm in FLAGS.liab_ttms],
    #              action_low=float(FLAGS.action_space[0]), action_high=float(FLAGS.action_space[1]))
    utils = Utils(n_episodes=FLAGS.train_sim, tenor=4)
    loggers = make_loggers(work_folder=work_folder)
    environment = make_environment(utils=utils)#, logger=loggers['train_loop'])
    environment_spec = specs.make_environment_spec(environment)
    if FLAGS.critic == 'c51':
        agent_networks = make_networks(action_spec=environment_spec.actions, max_time_steps=utils.num_period)
    elif 'qr' in FLAGS.critic:
        agent_networks = make_quantile_networks(action_spec=environment_spec.actions, time_embedding_dim=16, dt=utils.dt)
    elif FLAGS.critic == 'iqn':
        assert FLAGS.obj_func == 'cvar', 'IQN only support CVaR objective.'
        agent_networks = make_iqn_networks(action_spec=environment_spec.actions,cvar_th=FLAGS.threshold, max_time_steps=FLAGS.init_ttm)

    # Construct the agent.
    agent = D4PG(
        obj_func=FLAGS.obj_func,
        threshold=FLAGS.threshold,
        critic_loss_type=FLAGS.critic,
        environment_spec=environment_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        n_step=FLAGS.n_step,
        discount=1.0,
        sigma=0.3,  # pytype: disable=wrong-arg-types
        checkpoint=False,
        logger=loggers['learner'],
        batch_size=FLAGS.batch_size,
        policy_optimizer=snt.optimizers.Adam(FLAGS.lr),
        critic_optimizer=snt.optimizers.Adam(FLAGS.lr),
    )

    # Create the environment loop used for training.
    if not FLAGS.eval_only:
        train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop', logger=loggers['train_loop'])
        train_loop.run(num_episodes=FLAGS.train_sim)    
        save_policy(agent._learner._policy_network, f'./logs/{RUN_FOLDER}/{work_folder}')

    # Create the evaluation policy.
    if FLAGS.eval_only:
        policy_net = agent._learner._policy_network
        if FLAGS.agent_path == '':
            load_policy(policy_net, f'./logs/{RUN_FOLDER}/{work_folder}')
        else:
            load_policy(policy_net, FLAGS.agent_path)
        eval_policy = snt.Sequential([
            agent_networks['observation'],
            policy_net,
        ])
    else:
        eval_policy = snt.Sequential([
            agent_networks['observation'],
            agent_networks['policy'],
        ])

    print("Starting evaluation")
    # Create the evaluation actor and loop.
    eval_actor = actors.FeedForwardActor(policy_network=eval_policy)
    # eval_utils = Utils(init_ttm=FLAGS.init_ttm, np_seed=FLAGS.eval_seed, num_sim=FLAGS.eval_sim, spread=FLAGS.spread, volvol=FLAGS.vov, sabr=FLAGS.sabr, gbm=FLAGS.gbm, hed_ttm=FLAGS.hed_ttm,
    #                    init_vol=FLAGS.init_vol, poisson_rate=FLAGS.poisson_rate, 
    #                    moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std, 
    #                    mu=0.0, ttms=[int(ttm) for ttm in FLAGS.liab_ttms],
    #                    action_low=float(FLAGS.action_space[0]), action_high=float(FLAGS.action_space[1]))
    # TODO: FIND UD AF HVORFOR DEN TERMINERER UDEN AT EVALUATE
    eval_utils = Utils(n_episodes=FLAGS.eval_sim, tenor=4, spread=FLAGS.spread)
    eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder,'eval_env'))
    eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=loggers['eval_loop'])
    eval_loop.run(num_episodes=FLAGS.eval_sim)   
    print("Successfully finished.")
    Path(f'./logs/{RUN_FOLDER}/{work_folder}/ok').touch()

if __name__ == '__main__':
    app.run(main)
