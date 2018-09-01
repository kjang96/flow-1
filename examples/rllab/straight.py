"""Script used to train a single autonomous vehicle on a straight road.
"""

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from flow.scenarios.straight.gen import StraightGenerator
from flow.scenarios.straight.scenario import StraightScenario
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.vehicles import Vehicles
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, SumoCarFollowingParams
from rllab.envs.gym_env import GymEnv

HORIZON = 500  # this doesn't appear to be changing?


def run_task(*_):
    ###### Variables ######
    num_rl = 1
    num_idm = 1
    target_velocity = 10
    target_headway = 10
    max_accel = 5
    max_decel = 5
    speed_limit = 15 # as requested by UD 

    length = 1500 # whittle this down as much as possible
    #######################

    sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo", seed=0)
    # sumo_cfp = SumoCarFollowingParams()

    vehicles = Vehicles()
    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=num_rl,
                 speed_mode="aggressive"
                 )

    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, 
                                          {"noise": 0.1,
                                           "v0": target_velocity,
                                           "b": max_decel}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=num_idm)

    additional_env_params = {"target_velocity": target_velocity,
                             "max_accel": max_accel,
                             "max_decel": max_decel,
                             "target_headway": target_headway,
                            }

    env_params = EnvParams(horizon=HORIZON,
                           additional_params=additional_env_params)

    additional_net_params = {"speed_limit": speed_limit, "length": length}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="custom")

    print("UD Straight Road", exp_tag)
    scenario = StraightScenario(exp_tag, StraightGenerator, vehicles, net_params,
                            initial_config=initial_config)

    env_name = "StraightEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16, 16)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5000,#3600 * 72 * 2, # KJ why is this the way it is? does it optimize in some way?
        # batch_size=1000,
        max_path_length=horizon,
        n_itr=150,
        # n_itr=5,
        # whole_paths=True,
        discount=0.999,
        # step_size=v["step_size"],
    )
    algo.train(),


exp_tag = "delaware_6"

# for seed in [5]:
for seed in [5, 20, 68]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=8,
        # Keeps the snapshot parameters for all iterations
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="ec2",
        exp_prefix=exp_tag,
        # plot=True,
    )
