from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoLaneChangeParams
from flow.core.params import SumoCarFollowingParams

from flow.controllers import SumoCarFollowingController, GridRouter

from flow.scenarios.grid.gen import UDSSCGridGenerator
from flow.scenarios.grid.grid_scenario import SimpleGridScenario
from flow.controllers import RLController, IDMController, \
    SumoLaneChangeController, ContinuousRouter

# Training settings
HORIZON = 500
SIM_STEP = 1
BATCH_SIZE = 20000
ITR = 100
exp_tag = "grid_0"  # experiment prefix

# Sumo settings
FLOW_RATE = 350
FLOW_PROB = FLOW_RATE/3600
# 50 is pretty dec for striking the balance between having an RL
# there but not too often
RL_FLOW_RATE = 50
RL_FLOW_PROB = RL_FLOW_RATE/3600

# Local settings
N_PARALLEL = 1
RENDER = True
MODE = "local"
RESTART_INSTANCE = False
SEEDS = [1]

# # EC2 settings
# N_PARALLEL = 8
# RENDER = False
# MODE = "ec2"
# RESTART_INSTANCE = True
# SEEDS = [1, 2, 5, 91]

# # # Autoscaler settings
# N_PARALLEL = 10
# RENDER = False
# MODE = "local"
# RESTART_INSTANCE = True
# SEEDS = [1, 2, 5, 91, 104, 32] 

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


def run_task(*_):
    """Implement the run_task method needed to run experiments with rllab."""
    v_enter = 10
    inner_length = 300
    long_length = 48
    short_length = 48
    n = 1
    m = 1
    num_cars_left = 1
    num_cars_right = 1
    num_cars_top = 1
    num_cars_bot = 1
    tot_cars = (num_cars_left + num_cars_right) * m \
        + (num_cars_bot + num_cars_top) * n

    grid_array = {
        "short_length": short_length,
        "inner_length": inner_length,
        "long_length": long_length,
        "row_num": n,
        "col_num": m,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    sumo_params = SumoParams(sim_step=SIM_STEP, render=RENDER, restart_instance=RESTART_INSTANCE)

    vehicles = Vehicles()
    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {"noise": 0.1}),
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 speed_mode="all_checks",
                 num_vehicles=3,
                 sumo_car_following_params=SumoCarFollowingParams(
                     accel=1,
                     decel=1, 
                     tau=1.1,
                     impatience=0.5
                 ),
                #  lane_change_mode=1621,
                 lane_change_mode=0,
                 sumo_lc_params=SumoLaneChangeParams())

    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                # acceleration_controller=(IDMController, {}),
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 speed_mode="no_collide",
                 num_vehicles=1,
                 sumo_car_following_params=SumoCarFollowingParams(
                     tau=1.1,
                     impatience=0.5
                 ),
                #  lane_change_mode="no_lat_collide",
                 lane_change_mode="aggressive",
                 sumo_lc_params=SumoLaneChangeParams())

    tl_logic = TrafficLights(baseline=False)

    additional_env_params = {
        "target_velocity": 8,
        "num_steps": 500,
        "switch_time": 3.0,
        "num_observed": 5,
        # maximum acceleration for autonomous vehicles, in m/s^2
        "max_accel": 1,
        # maximum deceleration for autonomous vehicles, in m/s^2
        "max_decel": 1,
    }
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {
        "speed_limit": 8,
        "grid_array": grid_array,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }

    initial_config = InitialConfig(
        spacing="uniform", lanes_distribution=float("inf"), shuffle=True)

    inflow = InFlows()
    inflow.add(veh_type="idm", edge="bot0_0", name="idm", probability=50/3600)
    inflow.add(veh_type="idm", edge="top0_1", name="idm", probability=50/3600)
    inflow.add(veh_type="idm", edge="left1_0", name="idm", probability=100/3600)
    inflow.add(veh_type="idm", edge="right0_0", name="idm", probability=100/3600)
    
    inflow.add(veh_type="rl", edge="bot0_0", name="rl", probability=50/3600)
    inflow.add(veh_type="rl", edge="top0_1", name="rl", probability=50/3600)

    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    scenario = SimpleGridScenario(
        name=exp_tag,
        generator_class=UDSSCGridGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    env_name = "UDSSCGridEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=BATCH_SIZE,#64 * 3 * horizon,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=ITR,
        discount=0.999,
        # step_size=0.01,
    )
    algo.train()


for seed in [6]:  # , 7, 8]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=N_PARALLEL,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode=MODE,  # "local_docker", "ec2"
        exp_prefix=exp_tag,
        # plot=True,
    )
