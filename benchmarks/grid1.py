"""
Grid/green wave example

Action Dimension: (9, )

Observation Dimension: (447, )

Horizon: 400 steps
"""

from flow.utils.rllib import make_create_env
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.controllers import SumoCarFollowingController, GridRouter

# time horizon of a single rollout
HORIZON = 400


def gen_edges(row_num, col_num):
    edges = []
    for i in range(col_num):
        edges += ["left" + str(row_num) + '_' + str(i)]
        edges += ["right" + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ["bot" + str(i) + '_' + '0']
        edges += ["top" + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(v_enter, vehs_per_hour, col_num, row_num, additional_net_params):
    initial_config = InitialConfig(spacing="uniform",
                                   lanes_distribution=float("inf"),
                                   shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(veh_type="idm", edge=outer_edges[i], vehs_per_hour=vehs_per_hour,
                   departLane="free", departSpeed=v_enter)

    net_params = NetParams(in_flows=inflow,
                           no_internal_links=False,
                           additional_params=additional_net_params)

    return initial_config, net_params


def get_non_flow_params(enter_speed, additional_net_params):
    additional_init_params = {"enter_speed": enter_speed}
    initial_config = InitialConfig(additional_params=additional_init_params)
    net_params = NetParams(no_internal_links=False,
                           additional_params=additional_net_params)

    return initial_config, net_params


v_enter = 30

inner_length = 300
long_length = 100
short_length = 300
n = 3
m = 3
num_cars_left = 1
num_cars_right = 1
num_cars_top = 1
num_cars_bot = 1
edge_inflow = 600
rl_veh = 0
tot_cars = (num_cars_left + num_cars_right) * m \
           + (num_cars_bot + num_cars_top) * n

grid_array = {"short_length": short_length, "inner_length": inner_length,
              "long_length": long_length, "row_num": n, "col_num": m,
              "cars_left": num_cars_left, "cars_right": num_cars_right,
              "cars_top": num_cars_top, "cars_bot": num_cars_bot,
              "rl_veh": rl_veh}

additional_env_params = {"target_velocity": 50, "num_steps": HORIZON,
                         "control-length": 150, "switch_time": 2.0,
                         "total_inflow": edge_inflow*n*m}

additional_net_params = {"speed_limit": 35, "grid_array": grid_array,
                         "horizontal_lanes": 1, "vertical_lanes": 1}

vehicles = Vehicles()
vehicles.add(veh_id="idm",
             acceleration_controller=(SumoCarFollowingController, {}),
             sumo_car_following_params=SumoCarFollowingParams(
                 minGap=2.5,
                 max_speed=v_enter,
             ),
             routing_controller=(GridRouter, {}),
             num_vehicles=tot_cars,
             speed_mode="right_of_way")

initial_config, net_params = \
    get_flow_params(v_enter, edge_inflow, n, m, additional_net_params)


flow_params = dict(
    # name of the experiment
    exp_tag="grid_1",

    # name of the flow environment the experiment is running on
    env_name="PO_TrafficLightGridEnv",

    # name of the scenario class the experiment is running on
    scenario="SimpleGridScenario",

    # name of the generator used to create/modify network configuration files
    generator="SimpleGridGenerator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        restart_instance=True,
        sim_step=1,
        sumo_binary="sumo",
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=initial_config,
)

# get the env name and a creator for the environment (used by rllib)
create_env, env_name = make_create_env(params=flow_params, version=0)
