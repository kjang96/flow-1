"""Figure eight example."""

import json

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
from flow.controllers import RLController, IDMController, \
    SumoLaneChangeController, ContinuousRouter
from flow.scenarios.figure8.figure8_scenario import ADDITIONAL_NET_PARAMS
from flow.core.params import InFlows
# Training settings
HORIZON = 500
SIM_STEP = 1
BATCH_SIZE = 20000
ITR = 100
N_ROLLOUTS = 20
N_CPUS = 2
exp_tag = "ecc_26"  # experiment prefix

# Sumo settings
FLOW_RATE = 350
FLOW_PROB = FLOW_RATE/3600
# 50 is pretty dec for striking the balance between having an RL
# there but not too often
RL_FLOW_RATE = 50
RL_FLOW_PROB = RL_FLOW_RATE/3600

# # Local settings
# N_PARALLEL = 1
# SUMO_BINARY = "sumo-gui"
# MODE = "local"
# RESTART_INSTANCE = False
# SEEDS = [1]

# # EC2 settings
# N_PARALLEL = 8
# SUMO_BINARY = "sumo"
# MODE = "ec2"
# RESTART_INSTANCE = True
# SEEDS = [1, 2, 5, 91]

# Autoscaler settings
N_PARALLEL = 10
SUMO_BINARY = "sumo"
MODE = "local"
RESTART_INSTANCE = True
SEEDS = [1, 2, 5, 91, 104, 32] 

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

# <-- deterministic setting
inflow = InFlows()

inflow.add(veh_type="rl", edge="inflow_0", name="rl", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_0", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_0", name="idm", vehs_per_hour=50)

inflow.add(veh_type="rl", edge="inflow_1", name="rl", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
# -->

additional_env_params = {
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
    # "rl_action_noise": 0.7,
    # noise to add to the state space
    # "state_noise": 0.1,
    # what portion of the ramp the RL vehicle isn't controlled for 
    "control_length": 0.2,
}

additional_net_params = {
    # radius of the loops
    "ring_radius": 15,#15.25,
    # length of the straight edges connected the outer loop to the inner loop
    "lane_length": 70,
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


flow_params = dict(
    # name of the experiment
    exp_tag="UDSSC_rllib",

    # name of the flow environment the experiment is running on
    env_name="UDSSCMergeEnv",

    # name of the scenario class the experiment is running on
    scenario="UDSSCMergingScenario",

    # name of the generator used to create/modify network configuration files
    generator="UDSSCMergingGenerator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(sim_step=SIM_STEP, sumo_binary=SUMO_BINARY, restart_instance=RESTART_INSTANCE),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial= InitialConfig(
        x0=50,
        spacing="custom", # TODO make this custom? 
        additional_params={"merge_bunching": 0}
    )
)

if __name__ == "__main__":
    ray.init(num_cpus=N_CPUS+1, redirect_output=False)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [100, 50, 25]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    # config["sgd_batchsize"] = min(16 * 1024, config["train_batch_size"])
    # config["sgd_stepsize"] = 5e-5
    config["clip_param"] = 0.2
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = HORIZON
    config["observation_filter"] = "NoFilter"

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": "PPO",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 1,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200
            },
            "num_samples": 3,
        },
    })
