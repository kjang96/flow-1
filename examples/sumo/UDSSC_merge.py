"""
Cooperative merging example, consisting of 1 learning agent and 6 additional
vehicles in an inner ring, and 10 vehicles in an outer ring attempting to
merge into the inner ring. rllab version.

File name: UDSSC_merge.py
"""
from flow.core.experiment import SumoExperiment
from flow.controllers import IDMController, \
    SumoLaneChangeController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
from flow.scenarios.UDSSC_merge.gen import UDSSCMergingGenerator
from flow.scenarios.UDSSC_merge.scenario import UDSSCMergingScenario
from flow.envs.UDSSC_merge_env import UDSSCMergeEnv
from flow.core.params import InFlows

HORIZON = 500
FLOW_RATE = 350
FLOW_PROB = FLOW_RATE/3600 # in veh/s
# FLOW_PROB = 0.115
SIM_STEP = 1
ITR = 50

def merge_example(sumo_binary=None):
    sumo_params = SumoParams(sim_step=SIM_STEP, sumo_binary="sumo", restart_instance=False)

    inflow = InFlows()
    # inflow.add(veh_type="idm", edge="inflow_1", vehs_per_hour=FLOW_RATE)
    # inflow.add(veh_type="idm", edge="inflow_0", vehs_per_hour=FLOW_RATE)
    inflow.add(veh_type="idm", edge="inflow_1", probability=FLOW_PROB)
    inflow.add(veh_type="idm", edge="inflow_0", probability=FLOW_PROB)


    # note that the vehicles are added sequentially by the generator,
    # so place the merging vehicles after the vehicles in the ring
    vehicles = Vehicles()

    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {"noise": 0.1}),
                #  acceleration_controller=(IDMController, {"noise": 0.2}),
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 speed_mode="all_checks",
                 num_vehicles=1,
                 sumo_car_following_params=SumoCarFollowingParams(
                     tau=1.1,
                 ),
                 lane_change_mode=1621,
                #  lane_change_mode=0,
                 sumo_lc_params=SumoLaneChangeParams())

    additional_env_params = {
        # maximum acceleration for autonomous vehicles, in m/s^2
        "max_accel": 3,
        # maximum deceleration for autonomous vehicles, in m/s^2
        "max_decel": 3,
        # desired velocity for all vehicles in the network, in m/s
        "target_velocity": 15,
        # number of observable vehicles preceding the rl vehicle
        "n_preceding": 3,
        # number of observable vehicles following the rl vehicle
        "n_following": 3,
        # number of observable merging-in vehicle from the larger loop
        "n_merging_in": 4,
    }

    env_params = EnvParams(horizon=HORIZON,
                           additional_params=additional_env_params)

    additional_net_params = {
        # radius of the loops
        "ring_radius": 15,
        # length of the straight edges connected the outer loop to the inner loop
        "lane_length": 30,
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
        in_flows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params
    )

    initial_config = InitialConfig(
        x0=50,
        spacing="custom", # TODO make this custom? 
        additional_params={"merge_bunching": 0}
    )

    exp_tag = "UDSSC_Merge_3"
    scenario = UDSSCMergingScenario(
        name=exp_tag,
        generator_class=UDSSCMergingGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    env = UDSSCMergeEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = merge_example()
    
    # run for a set number of rollouts / time steps
    exp.run(ITR, HORIZON)
    
    # added by kj
    # import numpy as np
    # std = np.std(exp.env.accels)
    # print("Standard deviation of accelerations is: ", std)
    # import ipdb; ipdb.set_trace() 