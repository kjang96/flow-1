"""
Cooperative merging example, consisting of 1 learning agent and 6 additional
vehicles in an inner ring, and 10 vehicles in an outer ring attempting to
merge into the inner ring. rllab version.
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

HORIZON = 200
FLOW_RATE = 300

def merge_example(sumo_binary=None):
    sumo_params = SumoParams(sim_step=1, sumo_binary="sumo-gui", restart_instance=False)

    inflow = InFlows()
    inflow.add(veh_type="idm", edge="inflow_1", vehs_per_hour=FLOW_RATE)
    inflow.add(veh_type="idm", edge="inflow_0", vehs_per_hour=FLOW_RATE)

    # note that the vehicles are added sequentially by the generator,
    # so place the merging vehicles after the vehicles in the ring
    vehicles = Vehicles()

    # sumo_car_following_params=SumoCarFollowingParams(
    #                accel=max_accel,
    #                decel=max_decel,
    #                tau=1.1,
    #                max_speed=speed_limit),
    # Inner ring vehicles

    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {"noise": 0.2}),
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 speed_mode="all_checks",
                 num_vehicles=1,
                 sumo_car_following_params=SumoCarFollowingParams(
                     tau=1.1,
                 ),
                 lane_change_mode=1621,
                 sumo_lc_params=SumoLaneChangeParams())

    additional_env_params = {
        # maximum acceleration for autonomous vehicles, in m/s^2
        "max_accel": 3,
        # maximum deceleration for autonomous vehicles, in m/s^2
        "max_decel": 3,
        # desired velocity for all vehicles in the network, in m/s
        "target_velocity": 15,
        # number of observable vehicles preceding the rl vehicle
        "n_preceding": 2,
        # number of observable vehicles following the rl vehicle
        "n_following": 2,
        # number of observable merging-in vehicle from the larger loop
        "n_merging_in": 2,
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
        # max speed limit in the network
        "speed_limit": 15,
        # resolution of the curved portions
        "resolution": 100,
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
    exp.run(1, 1500)