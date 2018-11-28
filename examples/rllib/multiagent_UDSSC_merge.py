"""Example of a multi-agent environment containing the UDSSC merge
scenario and an adversary that is allowed to perturb
the accelerations of figure eight."""

# WARNING: Expected total reward is zero as adversary reward is
# the negative of the AV reward

import json

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.controllers import SumoLaneChangeController, ContinuousRouter
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import InFlows
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
# from flow.scenarios.figure8.figure8_scenario import ADDITIONAL_NET_PARAMS

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# Training settings
HORIZON = 500
SIM_STEP = 1
ITR = 100
N_ROLLOUTS = 40
exp_tag = "ma_19"  # experiment prefix

# # Local settings
# N_CPUS = 1
# RENDER = True
# MODE = "local"
# RESTART_INSTANCE = True
# # SEEDS = [1]

# Autoscaler settings
N_CPUS = 10
RENDER = False
MODE = "local"
RESTART_INSTANCE = True
SEEDS = [1, 2, 5, 91, 104, 32]  


inflow = InFlows()
    
inflow.add(veh_type="rl", edge="inflow_0", name="rl", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_0", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_0", name="idm", vehs_per_hour=50)

inflow.add(veh_type="rl", edge="inflow_1", name="rl", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = Vehicles()
# Inner ring vehicles
vehicles.add(veh_id="idm",
             acceleration_controller=(IDMController, {"noise": 0.1}),
             lane_change_controller=(SumoLaneChangeController, {}),
             routing_controller=(ContinuousRouter, {}),
             speed_mode="all_checks",
             num_vehicles=1,
             sumo_car_following_params=SumoCarFollowingParams(
                 accel=1,
                 decel=1, 
                 tau=1.1,
                 impatience=0.05
             ),
         #  lane_change_mode=1621,
             lane_change_mode=0,
             sumo_lc_params=SumoLaneChangeParams())

# A single learning agent in the inner ring
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
         # acceleration_controller=(IDMController, {}),
             lane_change_controller=(SumoLaneChangeController, {}),
             routing_controller=(ContinuousRouter, {}),
             speed_mode="no_collide",
             num_vehicles=1,
             sumo_car_following_params=SumoCarFollowingParams(
                 tau=1.1,
                 impatience=0.05
             ),
         #  lane_change_mode="no_lat_collide",
             lane_change_mode="aggressive",
             sumo_lc_params=SumoLaneChangeParams())

flow_params = dict(
    # name of the experiment
    exp_tag=exp_tag,

    # name of the flow environment the experiment is running on
    env_name='MultiAgentUDSSCMergeEnv',

    # name of the scenario class the experiment is running on
    scenario='UDSSCMergingScenario',

    # name of the generator used to create/modify network configuration files
    generator='UDSSCMergingGenerator',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=SIM_STEP,
        render=RENDER,
        restart_instance=RESTART_INSTANCE
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            # maximum acceleration for autonomous vehicles, in m/s^2
            "max_accel": 1,
            # maximum deceleration for autonomous vehicles, in m/s^2
            "max_decel": 1,
            # desired velocity for all vehicles in the network, in m/s
            "target_velocity": 8,
            # number of observable vehicles preceding the rl vehicle
            "n_preceding": 1, # HAS TO BE 1
            # number of observable vehicles following the rl vehicle
            "n_following": 1, # HAS TO BE 1
            # number of observable merging-in vehicle from the larger loop
            "n_merging_in": 6,
            # rl action noise
            "rl_action_noise": 0.5,
            # noise to add to the state space
            # "state_noise": 0.1,
            # what portion of the ramp the RL vehicle isn't controlled for 
            # "control_length": 0.1,
            'perturb_weight': 0.03
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params= {
            # radius of the loops
            "ring_radius": 15,#15.25,
            # length of the straight edges connected the outer loop to the inner loop
            "lane_length": 55,
            # length of the merge next to the roundabout
            "merge_length": 15,
            # number of lanes in the inner loop
            "inner_lanes": 1,
            # number of lanes in the outer loop
            "outer_lanes": 1,
            # max speed limit in the roundabout
            "roundabout_speed_limit": 8,
            # max speed limit in the rest of the roundabout
            "outside_speed_limit": 8,
            # resolution of the curved portions
            "resolution": 100,
            # num lanes
            "lane_num": 1,
        }
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        x0=50,
        spacing="custom",
        additional_params={"merge_bunching": 0}
    ),
)

if __name__ == '__main__':
    ray.init()

    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [100, 50, 25]})
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['sgd_minibatch_size'] = 128
    config['kl_target'] = 0.02
    config['horizon'] = HORIZON
    config['observation_filter'] = 'NoFilter'
    # <-- Tune
    config['lr'] = 0.01
    config['vf_loss_coeff'] = 1.0
    config['num_sgd_iter'] = 30
    config['vf_clip_param'] = 10.0
    # -->

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'av': gen_policy(), 'adversary': gen_policy()}

    def policy_mapping_fn(agent_id):
        return agent_id

    policy_ids = list(policy_graphs.keys())
    config.update({
        'multiagent': {
            'policy_graphs': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn)
        }
    })

    run_experiments({
        flow_params['exp_tag']: {
            'run': 'PPO',
            'env': env_name,
            'checkpoint_freq': 20,
            'stop': {
                'training_iteration': ITR
            },
            'config': config,
            'upload_dir': 's3://kathy.experiments/rllib/experiments',
        },
    })
