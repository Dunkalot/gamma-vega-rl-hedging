import os
from pathlib import Path
import pickle
from typing import Mapping, Sequence
from gym import spaces
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
from environment.Environment import TradingEnv, TrainLog, EvalLog
from environment.utils import Utils
import agent.distributional as ad
from agent.agent import D4PG
from absl import app
from absl import flags
from collections import OrderedDict
import dataclasses
import datetime

def ordered_asdict(obj):
    return OrderedDict((f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj))

print("SETTING FLOAT PRECISION TO 32")

tf.keras.backend.set_floatx('float32')
tf.keras.mixed_precision.set_global_policy("mixed_float16")

FLAGS = flags.FLAGS
flags.DEFINE_float('spread_train', 0.0, 'Hedging transaction cost (Default 0.0)')
flags.DEFINE_float('spread_eval', 0.0, 'Hedging transaction cost (Default 0.0)')
flags.DEFINE_string('obj_func', 'var', 'Objective function select from meanstd, var or cvar (Default var)')
flags.DEFINE_integer('train_sim', 40_000, 'train episodes (Default 40_000)')
flags.DEFINE_integer('eval_sim', 5_000, 'evaluation episodes (Default 40_000)')
flags.DEFINE_string('critic', 'c51', 'critic distribution type - c51, qr-huber, qr-gl, qr-gl_tl, '
                                     'qr-lapl, qr-lapl_tl, iqn-huber (Default c51)')

flags.DEFINE_float('threshold', 0.95, 'Objective function threshold. (Default 0.95)')
flags.DEFINE_float('std_coef', 1.645, 'Std coefficient when obj_func=meanstd. (Default 1.645)')

flags.DEFINE_string('dataset', '', 'Prefix folder for logger (Default None)')
flags.DEFINE_string('agent_path', '', 'trained agent path, only used when eval_only=True')
flags.DEFINE_boolean('eval_only', False, 'Ignore training (Default False)')
flags.DEFINE_boolean('per', False, 'Use PER for Replay sampling (Default False)')
#flags.DEFINE_float('lr', 1e-4, 'Learning rate for optimizer (Default 1e-4)')
flags.DEFINE_integer('n_step', 5, 'DRL TD Nstep (Default 5)')
flags.DEFINE_integer('batch_size', 256, 'Batch size to train the Network (Default 256)')
flags.DEFINE_float('priority_exponent', 0.6, 'priority exponent for the Prioritized replay table (Default 0.6)')
flags.DEFINE_float('importance_sampling_exponent', 0.2, 'importance sampling exponent for updating importance weight for PER (Default 0.2)')
flags.DEFINE_boolean('specific_folder', False, 'override the manual timestamped folder generation')
flags.DEFINE_string('dataset_train', '','name of the training dataset from the data folder')
flags.DEFINE_string('dataset_eval', '','name of the evaluation dataset from the data folder')
flags.DEFINE_integer('eval_offset', 0, 'index offset in the given dataset from where the evaluation index is meant to start. ')
flags.DEFINE_boolean('train_only', False, 'if true, doesnt evaluate')
flags.DEFINE_string('run_tag', '', 'unique tag on run if desired')
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def make_logger(work_folder, label, terminal=False):
    
    loggers = [
        log_utils.CSVLogger(f'./logs/{work_folder}/run_{FLAGS.run_tag}', label=label, add_uid=False)
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

def make_environment(utils,log_bef=None, log_af=None, logger = None) -> dm_env.Environment:
    # Make sure the environment obeys the dm_env.Environment interface.
    environment = wrappers.GymWrapper(TradingEnv(
    utils=utils,
    log_bef=log_bef,
    log_af=log_af,
    logger=logger))
    # Clip the action returned by the agent to the environment spec.
    #environment = wrappers.CanonicalSpecWrapper(environment, clip=False) # no need for scaling output
    environment = wrappers.SinglePrecisionWrapper(environment)

    return environment



def make_quantile_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (64,64),
    critic_layer_sizes: Sequence[int] =  (128,128),
    quantile_interval: float = 0.01, 
    ) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""
    print("USING QUANTILE NETWORK")
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

def save_agent(policy_network, observation_network, checkpoint_folder):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_snapshot = make_snapshot(policy_network)
    observation_snapshot = make_snapshot(observation_network)
    policy_save_path = f"{checkpoint_folder}/policy"
    observation_save_path = f"{checkpoint_folder}/observation"
    tf.saved_model.save(policy_snapshot, policy_save_path)
    tf.saved_model.save(observation_snapshot, observation_save_path)
    print(f"Policy and observation networks saved to {policy_save_path} and {observation_save_path}")

def load_agent(policy_network, observation_network, checkpoint_folder):
    # First initialize the networks with dummy inputs to create variables
    # Create dummy batch with appropriate dimensions for the environment
    dummy_obs = tf.zeros([1, 8], dtype=tf.float32)  # Adjust dimensions as needed for your environment
    
    # Initialize observation network
    processed_obs = observation_network(dummy_obs)
    
    # Initialize policy network with processed observation
    _ = policy_network(processed_obs)
    
    # Now that variables are initialized, load the saved weights
    trainable_variables_snapshot = {}
    load_policy_net = tf.saved_model.load(checkpoint_folder+'/policy')
    load_observation_net = tf.saved_model.load(checkpoint_folder+'/observation')
    
    print(f"Loading policy network with {len(load_policy_net.trainable_variables)} variables")
    print(f"Target policy network has {len(policy_network.trainable_variables)} variables")
    
    for var in load_policy_net.trainable_variables:
        var_name = '/'.join(var.name.split('/')[1:])
        trainable_variables_snapshot[var_name] = var.numpy()
        
    for var in policy_network.trainable_variables:
        var_name_wo_name_scope = '/'.join(var.name.split('/')[1:])
        if var_name_wo_name_scope in trainable_variables_snapshot:
            var.assign(trainable_variables_snapshot[var_name_wo_name_scope])
        else:
            print(f"WARNING: Variable {var_name_wo_name_scope} not found in saved model")
    
    trainable_variables_snapshot = {}
    print(f"Loading observation network with {len(load_observation_net.trainable_variables)} variables")
    print(f"Target observation network has {len(observation_network.trainable_variables)} variables")
    
    for var in load_observation_net.trainable_variables:
        var_name = '/'.join(var.name.split('/')[1:])
        trainable_variables_snapshot[var_name] = var.numpy()
        
    for var in observation_network.trainable_variables:
        var_name_wo_name_scope = '/'.join(var.name.split('/')[1:])
        if var_name_wo_name_scope in trainable_variables_snapshot:
            var.assign(trainable_variables_snapshot[var_name_wo_name_scope])
        else:
            print(f"WARNING: Variable {var_name_wo_name_scope} not found in saved model")

def main(argv):

    


    work_folder = f'{FLAGS.obj_func}/{FLAGS.dataset_train}/{FLAGS.spread_train}/{FLAGS.train_sim}'
    # Create an environment, grab the spec, and use it to create networks.
    utils = Utils(n_episodes=FLAGS.train_sim, tenor=4, spread=FLAGS.spread_train, data_path=FLAGS.dataset_train)
    loggers = make_loggers(work_folder=work_folder)

    train_log_bef = lambda self, result,t: None#train_logfunc._log_before
    train_log_af = lambda self, result,t: None#train_logfunc._log_after
    environment = make_environment(utils=utils, log_bef=train_log_bef, log_af=train_log_af)#, logger=loggers['train_loop'])
    environment_spec = specs.make_environment_spec(environment)


    agent_networks = make_quantile_networks(action_spec=environment_spec.actions)

    # Construct the agent.
    agent = D4PG(
        obj_func=FLAGS.obj_func,
        threshold=0.95,
        critic_loss_type=FLAGS.critic,
        environment_spec=environment_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        n_step=FLAGS.n_step,
        discount=0.985,
        sigma=0.3,  # pytype: disable=wrong-arg-types
        checkpoint=False,
        logger=loggers['learner'],
        batch_size=FLAGS.batch_size,
        policy_optimizer=snt.optimizers.Adam(1e-5),
        critic_optimizer=snt.optimizers.Adam(2e-4),
        annealer_steps = 200000*6
    )

    
    # Create the environment loop used for training.
    if not FLAGS.eval_only:
        train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop', logger=loggers['train_loop'])
        train_loop.run(num_episodes=FLAGS.train_sim)
        save_agent(agent._learner._policy_network, agent._learner._observation_network, f'./logs/{work_folder}/run_{FLAGS.run_tag}')
    if not FLAGS.train_only:
        # Create the evaluation policy.
        if FLAGS.eval_only:
            policy_net = agent._learner._policy_network
            observation_net = agent._learner._observation_network
            if FLAGS.agent_path == '':
                load_agent(policy_net, observation_net, f'./logs/{work_folder}/run_{FLAGS.run_tag}')
            else:
                load_agent(policy_net, observation_net, FLAGS.agent_path)
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
    
        eval_utils = Utils(n_episodes=FLAGS.eval_sim, tenor=4, spread=FLAGS.spread_eval, test=True, test_episode_offset=FLAGS.eval_offset, data_path=FLAGS.dataset_eval)
        #eval_utils.vol_kernel = agent._learner._target_observation_network.vol_kernel
        #eval_utils.volvol_kernel = agent._learner._target_observation_network.volvol_kernel
        eva_logfunc = EvalLog()
        eval_log_bef = eva_logfunc._log_before
        eval_log_af = eva_logfunc._log_after
        eval_folder = f'/{FLAGS.dataset_eval}/{FLAGS.spread_eval}'
        eval_env = make_environment(utils=eval_utils,log_bef = eval_log_bef, log_af = eval_log_af, logger=make_logger(work_folder+eval_folder,'eval_env'))
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=loggers['eval_loop'])
        eval_loop.run(num_episodes=FLAGS.eval_sim)   
    else:
        print("FLAG: 'train_only is TRUE. Skipping evaluation.")
    print("Successfully finished.")
    Path(f'./logs/{work_folder}/run_{FLAGS.run_tag}/ok').touch()

if __name__ == '__main__':
    app.run(main)
