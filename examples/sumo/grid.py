"""Grid example."""
from flow.controllers.routing_controllers import GridRouter
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
     InFlows
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
# from flow.envs.green_wave_env import PO_TrafficLightGridEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.grid.gen import SimpleGridGenerator
from flow.scenarios.grid.grid_scenario import SimpleGridScenario
from flow.core.params import SumoCarFollowingParams

from flow.controllers import SumoCarFollowingController, GridRouter, ContinuousRouter

# Settings
SIM_STEP = 1
EXP_PREFIX = "greenwave_0"
HORIZON = 500

# # Local Settings
RESTART_INSTANCE = False
N_PARALLEL = 1
ITR = 2
SUMO_BINARY = "sumo"
BATCH_SIZE = 15000
MODE = "local"
SEEDS = [1]

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


def get_flow_params(v_enter, vehs_per_hour, col_num, row_num,
                    additional_net_params, inflow_prob=0):
    initial_config = InitialConfig(spacing="uniform",
                                   lanes_distribution=float("inf"),
                                   shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(veh_type="idm", edge=outer_edges[i],
                #    vehs_per_hour=vehs_per_hour,
                   probability=inflow_prob,
                   departLane="free", departSpeed=v_enter)

    net_params = NetParams(in_flows=inflow,
                           no_internal_links=False,
                           additional_params=additional_net_params)

    return initial_config, net_params

def grid_example(sumo_binary=None):
    """
    Perform a simulation of vehicles on a grid.

    Parameters
    ----------
    sumo_binary: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a grid.
    """
    inner_length = 300
    long_length = 500
    short_length = 300
    n = 2
    m = 2
    num_cars_left = 1
    num_cars_right = 1
    num_cars_top = 1
    num_cars_bot = 1
    tot_cars = (num_cars_left + num_cars_right) * m \
        + (num_cars_bot + num_cars_top) * n
    num_observed = 2
    inflow_rate = 350
    inflow_prob = 1/11
    # inner_length = 300
    # long_length = 500
    # short_length = 300
    # n = 2
    # m = 3
    # num_cars_left = 20
    # num_cars_right = 20
    # num_cars_top = 20
    # num_cars_bot = 20
    # tot_cars = (num_cars_left + num_cars_right) * m \
    #     + (num_cars_top + num_cars_bot) * n
    grid_array = {"short_length": short_length, "inner_length": inner_length,
                  "long_length": long_length, "row_num": n, "col_num": m,
                  "cars_left": num_cars_left, "cars_right": num_cars_right,
                  "cars_top": num_cars_top, "cars_bot": num_cars_bot}

    sumo_params = SumoParams(sim_step=SIM_STEP,
                             sumo_binary=SUMO_BINARY,
                             restart_instance=RESTART_INSTANCE)


    vehicles = Vehicles()
    vehicles.add(veh_id="idm",
                 acceleration_controller=(SumoCarFollowingController, {}),
                 sumo_car_following_params=SumoCarFollowingParams(
                   accel=max_accel,
                   decel=max_decel,
                   tau=1.1,
                   max_speed=speed_limit),
                 routing_controller=(GridRouter, {}),
                #  routing_controller=(ContinuousRouter, {}),
                 num_vehicles=tot_cars,
                 speed_mode="all_checks")

    # additional_env_params = {"target_velocity": target_velocity,
    #                          "min_yellow_time": 4.0, "min_green_time": 8.0,
    #                          "num_observed": num_observed}

    env_params = EnvParams(horizon=HORIZON, additional_params=ADDITIONAL_ENV_PARAMS)

    tl_logic = TrafficLights(baseline=False)
    phases = [{
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "GGGrrrGGGrrr"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "state": "yyyrrryyyrrr"
    }, {
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "rrrGGGrrrGGG"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "state": "rrryyyrrryyy"
    }]
    tl_logic.add("center0", phases=phases, programID=1)
    tl_logic.add("center1", phases=phases, programID=1)
    tl_logic.add("center2", phases=phases, programID=1)
    tl_logic.add("center3", phases=phases, programID=1)
    # tl_logic.add("center2", tls_type="actuated", phases=phases, programID=1)

    additional_net_params = {"speed_limit": speed_limit, "grid_array": grid_array,
                             "horizontal_lanes": 1, "vertical_lanes": 1}


    initial_config, net_params = get_flow_params(v_enter, inflow_rate, n, m, additional_net_params,
                                                 inflow_prob=inflow_prob)

    scenario = SimpleGridScenario(name="grid-intersection",
                                  generator_class=SimpleGridGenerator,
                                  vehicles=vehicles,
                                  net_params=net_params,
                                  initial_config=initial_config,
                                  traffic_lights=tl_logic)

    env = AccelEnv(env_params, sumo_params, scenario)
    # env = PO_TrafficLightGridEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = grid_example()

    # run for a set number of rollouts / time steps
    exp.run(ITR, HORIZON)
