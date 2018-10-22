"""
Cooperative merging example, consisting of 1 learning agent and 6 additional
vehicles in an inner ring, and 10 vehicles in an outer ring attempting to
merge into the inner ring. rllab version.

File name: UDSSC_merge.py
"""
import sys
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

from flow.controllers import RLController, IDMController, \
    SumoLaneChangeController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
from flow.scenarios.UDSSC_merge.gen import UDSSCMergingGenerator
from flow.scenarios.UDSSC_merge.scenario import UDSSCMergingScenario
from flow.core.params import InFlows

# Training settings
HORIZON = 500
SIM_STEP = 1
BATCH_SIZE = 20000
ITR = 50
exp_tag = "yung_test"  # experiment prefix

# Sumo settings
FLOW_RATE = 350
FLOW_PROB = FLOW_RATE/3600
# 50 is pretty dec for striking the balance between having an RL
# there but not too often
RL_FLOW_RATE = 50
RL_FLOW_PROB = RL_FLOW_RATE/3600

# # Local settings
# N_PARALLEL = 1
# SUMO_BINARY = "sumo"
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
SEEDS = [1, 2, 5, 91]

def main():
    for seed in SEEDS:
        run_experiment_lite(
            run_task,
            # Number of parallel workers for sampling
            n_parallel=N_PARALLEL,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a
            # random seed will be used
            seed=seed,
            mode=MODE,
            exp_prefix=exp_tag,
            # plot=True,
        )

def run_task(*_):
    # checks()

    sumo_params = SumoParams(sim_step=SIM_STEP, sumo_binary=SUMO_BINARY, restart_instance=RESTART_INSTANCE)
    # # <--
    # inflow = InFlows()
    # # inflow.add(veh_type="idm", edge="inflow_1", vehs_per_hour=FLOW_RATE)
    # # inflow.add(veh_type="idm", edge="inflow_0", vehs_per_hour=FLOW_RATE)
    # inflow.add(veh_type="idm", edge="inflow_1", name="idm", probability=FLOW_PROB)
    # inflow.add(veh_type="idm", edge="inflow_0", name="idm", probability=FLOW_PROB)
    # # Add RL vehicles on equation
    # # inflow.add(veh_type="rl", edge="inflow_0", name="rl", probability=RL_FLOW_PROB)
    # inflow.add(veh_type="rl", edge="inflow_1", name="rl", vehs_per_hour=RL_FLOW_RATE)
    # # -->

    inflow = InFlows()
    
    inflow.add(veh_type="rl", edge="inflow_0", name="rl", vehs_per_hour=50)
    inflow.add(veh_type="idm", edge="inflow_0", name="idm", vehs_per_hour=50)
    inflow.add(veh_type="idm", edge="inflow_0", name="idm", vehs_per_hour=50)
    # inflow.add(veh_type="rl", edge="inflow_0", name="rl", probability=50/3600)


    # inflow.add(veh_type="idm", edge="inflow_1", name="idm", probability=300/3600)
    inflow.add(veh_type="rl", edge="inflow_1", name="rl", vehs_per_hour=50)
    inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
    inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
    inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)

    # note that the vehicles are added sequentially by the generator,
    # so place the merging vehicles after the vehicles in the ring
    vehicles = Vehicles()

    # Inner ring vehicles
    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {"noise": 0.1}),
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 speed_mode="all_checks",
                 num_vehicles=1,
                 sumo_car_following_params=SumoCarFollowingParams(
                     tau=1.1,
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
                 ),
                #  lane_change_mode="no_lat_collide",
                 lane_change_mode="aggressive",
                 sumo_lc_params=SumoLaneChangeParams())

    additional_env_params = {
        # maximum acceleration for autonomous vehicles, in m/s^2
        "max_accel": 1,
        # maximum deceleration for autonomous vehicles, in m/s^2
        "max_decel": 1,
        # desired velocity for all vehicles in the network, in m/s
        "target_velocity": 15,
        # number of observable vehicles preceding the rl vehicle
        "n_preceding": 1, # HAS TO BE 1
        # number of observable vehicles following the rl vehicle
        "n_following": 1, # HAS TO BE 1
        # number of observable merging-in vehicle from the larger loop
        "n_merging_in": 6,
        # rl action noise
        # "rl_action_noise": 0.1,
        # noise to add to the state space
        # "state_noise": 0.1
    }

    env_params = EnvParams(horizon=HORIZON,
                           additional_params=additional_env_params)

    additional_net_params = {
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
        "outside_speed_limit": 15,
        # resolution of the curved portions
        "resolution": 100,
        # num lanes
        "lane_num": 1,
    }

    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params
    )

    initial_config = InitialConfig(
        x0=50,
        spacing="custom", # TODO make this custom? 
        additional_params={"merge_bunching": 0}
    )

    scenario = UDSSCMergingScenario(
        name=exp_tag,
        generator_class=UDSSCMergingGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    env_name = "UDSSCMergeEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params,
                   net_params, initial_config, scenario)
    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)
    )

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

def checks():
    """
    KJ Personal utils to remind myself not to do stupid things
    """
    if SIM_STEP == 1:
        cont = input("The SIM_STEP you entered is not 0.1. Continue? [y/n]  ")
        if cont == "y" or cont == "Y": 
            pass
        else:
            sys.exit()


# exp_tag = "UDSSCMerge_14"  # experiment prefix


if __name__ == "__main__":
    main()
