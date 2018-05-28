"""
Runner script for environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. Furthermore, the rllib specific algorithm/
parameters can be specified here once and used on multiple environments.
"""
import json

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.rllib import FlowParamsEncoder

# use this to specify the environment to run
from benchmarks.merge0 import flow_params, env_name, create_env

# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
PARALLEL_ROLLOUTS = 20


if __name__ == "__main__":
    ray.init(num_cpus=PARALLEL_ROLLOUTS, redirect_output=True)
    horizon = flow_params["env"].horizon

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = PARALLEL_ROLLOUTS
    config["timesteps_per_batch"] = horizon * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = horizon

    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)
    config['env_config']['flow_params'] = flow_json

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        "highway_stabilize": {
            "run": "PPO",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 5,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },
            "repeat": 3,
            "trial_resources": {
                "cpu": 1,
                "gpu": 0,
                "extra_cpu": PARALLEL_ROLLOUTS - 1,
            },
        },
    })
