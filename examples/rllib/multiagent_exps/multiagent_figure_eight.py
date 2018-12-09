"""Example of a multi-agent environment containing a figure eight with
one autonomous vehicle and an adversary that is allowed to perturb
the accelerations of figure eight."""

# WARNING: Expected total reward is zero as adversary reward is
# the negative of the AV reward

from copy import deepcopy
import json

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.vehicles import Vehicles
from flow.scenarios.figure_eight import ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# time horizon of a single rollout
HORIZON = 500
# number of rollouts per training iteration
N_ROLLOUTS = 40
# number of parallel workers
N_CPUS = 10

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = Vehicles()
vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    speed_mode='no_collide',
    num_vehicles=13)
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    speed_mode='no_collide',
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag='ma_f8_9',

    # name of the flow environment the experiment is running on
    env_name='MultiAgentAccelEnv',

    # name of the scenario class the experiment is running on
    scenario='Figure8Scenario',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 20,
            'max_accel': 3,
            'max_decel': 3,
            'perturb_weight': 0.03
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


def setup_exps():
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]}) #DIFF
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['kl_target'] = 0.02
    config['num_sgd_iter'] = 10
    config['horizon'] = HORIZON

    ### <-- old params
    # alg_run = 'PPO'
    # agent_cls = get_agent_class(alg_run)
    # config = agent_cls._default_config.copy()
    # config['num_workers'] = N_CPUS
    # config['train_batch_size'] = HORIZON * N_ROLLOUTS
    # config['simple_optimizer'] = True
    # config['gamma'] = 0.999  # discount rate
    # config['model'].update({'fcnet_hiddens': [100, 50, 25]})
    # config['use_gae'] = True
    # config['lambda'] = 0.97
    # config['sgd_minibatch_size'] = 128
    # config['kl_target'] = 0.02
    # config['num_sgd_iter'] = 10
    # config['horizon'] = HORIZON
    # config['observation_filter'] = 'NoFilter'
    ### -->

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})
    # # <-- old
    # # Setup PG with an ensemble of `num_policies` different policy graphs
    # policy_graphs = {'av': gen_policy(), 'adversary': gen_policy()}

    # def policy_mapping_fn(agent_id):
    #     return agent_id

    # config.update({
    #     'multiagent': {
    #         'policy_graphs': policy_graphs,
    #         'policy_mapping_fn': tune.function(policy_mapping_fn)
    #     }
    # })
    # # old -->

    # <-- new
    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'av': gen_policy()}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policy_graphs': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })
    # new  -->

    return alg_run, env_name, config


if __name__ == '__main__':

    alg_run, env_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS+1)

    run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 1,
            'stop': {
                'training_iteration': 5
            },
            'config': config,
            'upload_dir': 's3://kathy.experiments/rllib/experiments',
            # 'upload_dir': 's3://<BUCKET NAME>'
        },
    })
