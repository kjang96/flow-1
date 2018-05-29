"""
Trains a fraction of vehicles in a ring road structure to regulate the flow of
vehicles through an intersection. In this example, every vehicle in the
network is an autonomous vehicle.

Action Dimension: (16, )

Observation Dimension: (32, )

Horizon: 1500 steps
"""

from flow.utils.rllib import make_create_env
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers import ContinuousRouter, RLController
from flow.scenarios.figure8.figure8_scenario import ADDITIONAL_NET_PARAMS

# time horizon of a single rollout
HORIZON = 1500

# We place 16 autonomous vehicle and 0 human-driven vehicles in the network
vehicles = Vehicles()
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             speed_mode="no_collide",
             num_vehicles=14)

print(14)

flow_params = dict(
    # name of the experiment
    exp_tag="figure_eight_intersection_control",

    # name of the flow environment the experiment is running on
    env_name="AccelEnv",

    # name of the scenario class the experiment is running on
    scenario="Figure8Scenario",

    # name of the generator used to create/modify network configuration files
    generator="Figure8Generator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.1,
        sumo_binary="sumo",
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 30,
            "max_accel": 3,
            "max_decel": 3,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=ADDITIONAL_NET_PARAMS,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)

# get the env name and a creator for the environment (used by rllib)
create_env, env_name = make_create_env(params=flow_params, version=0)
