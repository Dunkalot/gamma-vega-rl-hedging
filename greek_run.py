import os
from pathlib import Path

import acme
from acme import wrappers
import acme.utils.loggers as log_utils
import dm_env

from environment.Environment import TradingEnv
from environment.utils import Utils
from agent.agent import DeltaHedgeAgent, GammaHedgeAgent, VegaHedgeAgent
from environment.Environment import EvalLog

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('eval_sim', 5_000, 'evaluation episodes (Default 40_000)')
flags.DEFINE_float('spread', 0.005, 'Hedging transaction cost (Default 0.0)')
flags.DEFINE_string('strategy', 'delta', 'Hedging strategy opt: delta / gamma/ vega (Default delta')
flags.DEFINE_string('run_tag','',"unique tag for run")
flags.DEFINE_integer('eval_offset',0, 'idx offset in the eval dataset')
flags.DEFINE_string('dataset','',"name of dataset in the data folder")

    
    

def make_logger(work_folder, label, terminal=False):
    loggers = [
        log_utils.CSVLogger(f'./logs/{work_folder}', label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label))
    
    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    # loggers = log_utils.TimeFilter(logger, 1.0)
    return logger

# def make_environment(utils, logger = None) -> dm_env.Environment:
#     # Make sure the environment obeys the dm_env.Environment interface.
#     environment = wrappers.GymWrapper(TradingEnv(
#     utils=utils,
#     logger=logger))
#     # Clip the action returned by the agent to the environment spec.
#     # environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
#     environment = wrappers.SinglePrecisionWrapper(environment)

#     return environment
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


def main(argv):
    gamma_hedge_ratio = 1.0
    
    work_folder = f'greek_baseline/{FLAGS.strategy}/{FLAGS.dataset}/{FLAGS.spread}/{FLAGS.eval_sim}/run_{FLAGS.run_tag}'
    
    # Create an environment, grab the spec, and use it to create networks.
    # eval_utils = Utils(init_ttm=FLAGS.init_ttm, np_seed=4321, num_sim=FLAGS.eval_sim, spread=FLAGS.spread, volvol=FLAGS.vov, sabr=FLAGS.sabr, gbm=FLAGS.gbm, hed_ttm=FLAGS.hed_ttm,
    #                    init_vol=FLAGS.init_vol, poisson_rate=FLAGS.poisson_rate, 
    #                    moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std, 
    #                    mu=FLAGS.mu, ttms=[int(ttm) for ttm in FLAGS.liab_ttms])
    eva_logfunc = EvalLog()
    eval_log_bef = eva_logfunc._log_before
    eval_log_af = eva_logfunc._log_after
    eval_utils = Utils(n_episodes=FLAGS.eval_sim, tenor=4, spread=FLAGS.spread, test_episode_offset=FLAGS.eval_offset, test=True, data_path=FLAGS.dataset)
    eval_env = make_environment(utils=eval_utils,log_bef = eval_log_bef, log_af = eval_log_af, logger=make_logger(work_folder,'eval_env'))
    # Create the evaluation actor and loop.
    if FLAGS.strategy == 'gamma_risk_limit':
        # gamma hedging
        #eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, f'eval_gamma_env'))
        eval_actor = GammaHedgeAgent(eval_env, gamma_hedge_ratio, risk_limit=True)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, f'eval_gamma_loop',True))
        eval_loop.run(num_episodes=FLAGS.eval_sim)
    elif FLAGS.strategy == 'gamma':
        # gamma hedging
        #eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, f'eval_gamma_env'))
        eval_actor = GammaHedgeAgent(eval_env, gamma_hedge_ratio)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, f'eval_gamma_loop',True))
        eval_loop.run(num_episodes=FLAGS.eval_sim)
    elif FLAGS.strategy == 'gamma_partial80':
        # gamma hedging
        #eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, f'eval_gamma_env'))
        eval_actor = GammaHedgeAgent(eval_env, 0.8)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, f'eval_gamma_loop',True))
        eval_loop.run(num_episodes=FLAGS.eval_sim)
    elif FLAGS.strategy == 'gamma_partial90':
        # gamma hedging
        #eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, f'eval_gamma_env'))
        eval_actor = GammaHedgeAgent(eval_env, 0.9)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, f'eval_gamma_loop',True))
        eval_loop.run(num_episodes=FLAGS.eval_sim)    
    elif FLAGS.strategy == 'gamma_partial70':
        # gamma hedging
        #eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, f'eval_gamma_env'))
        eval_actor = GammaHedgeAgent(eval_env, 0.7)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, f'eval_gamma_loop',True))
        eval_loop.run(num_episodes=FLAGS.eval_sim)    
    elif FLAGS.strategy == 'gamma_partial50':
        # gamma hedging
        #eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, f'eval_gamma_env'))
        eval_actor = GammaHedgeAgent(eval_env, 0.5)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, f'eval_gamma_loop',True))
        eval_loop.run(num_episodes=FLAGS.eval_sim)
    elif FLAGS.strategy == 'delta':
        # delta hedging
        #eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, 'eval_delta_env'))
        eval_actor = DeltaHedgeAgent(eval_env, gamma_hedge_ratio)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, 'eval_delta_loop', True))
        eval_loop.run(num_episodes=FLAGS.eval_sim)
    elif FLAGS.strategy == 'vega':
        # vega hedging
        #eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, 'eval_vega_env'))
        eval_actor = VegaHedgeAgent(eval_env)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, 'eval_vega_loop', True))
        eval_loop.run(num_episodes=FLAGS.eval_sim)

    Path(f'./logs/{work_folder}/ok').touch()

if __name__ == '__main__':
    app.run(main)
