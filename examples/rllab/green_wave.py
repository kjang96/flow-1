from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
     InFlows
    
from flow.core.params import SumoCarFollowingParams

from flow.controllers import SumoCarFollowingController, GridRouter, ContinuousRouter

from flow.scenarios.grid.gen import SimpleGridGenerator
from flow.scenarios.grid.grid_scenario import SimpleGridScenario

# Settings
SIM_STEP = 1
EXP_PREFIX = "greenwave_2"
HORIZON = 500

# Local Settings
# RESTART_INSTANCE = False
# N_PARALLEL = 1
# ITR = 500
# SUMO_BINARY = "sumo-gui"
# BATCH_SIZE = 15000
# MODE = "local"
# SEEDS = [1]

# # # EC2 Settings
RESTART_INSTANCE = True
N_PARALLEL = 8
ITR = 500
SUMO_BINARY = "sumo"
BATCH_SIZE = 25000
MODE = "ec2"
SEEDS = [1, 7, 91]

def main():
    for seed in SEEDS:
        run_experiment_lite(
            run_task,
            n_parallel=N_PARALLEL,
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a
            # random seed will be used
            seed=seed,
            mode=MODE,
            exp_prefix=EXP_PREFIX,
            # plot=True,
        )

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


def get_non_flow_params(enter_speed, additional_net_params):
    additional_init_params = {"enter_speed": enter_speed}
    initial_config = InitialConfig(additional_params=additional_init_params)
    net_params = NetParams(no_internal_links=False,
                           additional_params=additional_net_params)

    return initial_config, net_params


def run_task(*_):
    v_enter = 20
    target_velocity = 20
    speed_limit = 25
    switch_time = 3.0
    
    max_accel = 2.6
    max_decel = 4.5
    inner_length = 200
    long_length = 200
    short_length = 200
    n = 1
    m = 1
    num_cars_left = 1
    num_cars_right = 1
    num_cars_top = 1
    num_cars_bot = 1
    tot_cars = (num_cars_left + num_cars_right) * m \
        + (num_cars_bot + num_cars_top) * n
    num_observed = 4
    inflow_rate = 350
    inflow_prob = 1/11


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

    tl_logic = TrafficLights(baseline=False)

    additional_env_params = {"target_velocity": target_velocity,
                             "min_yellow_time": 4.0, "min_green_time": 2.0,
                             "num_observed": num_observed}

    env_params = EnvParams(horizon=HORIZON, additional_params=additional_env_params)

    additional_net_params = {"speed_limit": speed_limit, "grid_array": grid_array,
                             "horizontal_lanes": 1, "vertical_lanes": 1}

    # initial_config, net_params = get_flow_params(v_enter, inflow_rate, inflow_prob, n, m,
    #                                              additional_net_params)
    initial_config, net_params = get_flow_params(v_enter, inflow_rate, n, m, additional_net_params,
                                                 inflow_prob=inflow_prob)

    # initial_config, net_params = get_non_flow_params(v_enter, additional_net_params)

    scenario = SimpleGridScenario(name="grid-intersection",
                                  generator_class=SimpleGridGenerator,
                                  vehicles=vehicles,
                                  net_params=net_params,
                                  initial_config=initial_config,
                                  traffic_lights=tl_logic)

    env_name = "PO_TrafficLightGridEnv"
    # env_name = "TrafficLightGridEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=BATCH_SIZE,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=ITR,
        discount=0.999,
        # step_size=0.01,
    )
    algo.train()

if __name__ == "__main__":
    main()
