"""Runs the environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. This script than handles running the rllab
specific RL algorithm. Specifically, this script is designed for hyperparameter
tuning of the TRPO algorithm.
"""
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights

# use this to specify the environment to run
from flow.benchmarks.grid1 import flow_params

# number of rollouts per training iteration
N_ROLLOUTS = 50
# number of parallel workers
PARALLEL_ROLLOUTS = 8


def run_task(*_):
    """Implement the ``run_task`` method needed to run experiments with rllab.

    Note that the flow-specific parameters are imported at the start of this
    script and unzipped and processed here.
    """
    env_name = flow_params["env_name"]
    exp_tag = flow_params["exp_tag"]
    sumo_params = flow_params["sumo"]
    vehicles = flow_params["veh"]
    env_params = flow_params["env"]
    net_params = flow_params["net"]
    initial_config = flow_params.get("initial", InitialConfig())
    traffic_lights = flow_params.get("tls", TrafficLights())

    # import the scenario and generator classes
    module = __import__("flow.scenarios", fromlist=[flow_params["scenario"]])
    scenario_class = getattr(module, flow_params["scenario"])
    module = __import__("flow.scenarios", fromlist=[flow_params["generator"]])
    generator_class = getattr(module, flow_params["generator"])

    # create the scenario object
    scenario = scenario_class(
        name=exp_tag,
        generator_class=generator_class,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    env = normalize(env)

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(100, 50, 25))

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    horizon = flow_params["env"].horizon

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=horizon * (N_ROLLOUTS - PARALLEL_ROLLOUTS + 1),
        max_path_length=horizon,
        n_itr=500,
        discount=0.999,
        step_size=0.01,
    )
    algo.train(),


for seed in [5, 20, 68, 100, 128]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=PARALLEL_ROLLOUTS,
        # Keeps the snapshot parameters for all iterations
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",
        exp_prefix=flow_params["exp_tag"],
        sync_s3_pkl=True,
    )
