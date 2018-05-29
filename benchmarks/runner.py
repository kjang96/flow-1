"""
Runner script for environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. Furthermore, the rllib specific algorithm/
parameters can be specified here once and used on multiple environments.
"""
import json
import time
import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune import grid_search
from ray.tune.registry import register_env

from flow.utils.rllib import FlowParamsEncoder

# use this to specify the environment to run
from benchmarks.figureeight2 import flow_params, env_name, create_env

# number of rollouts per training iteration
N_ROLLOUTS = 15
# number of parallel workers
PARALLEL_ROLLOUTS = 15


if __name__ == "__main__":
    start = time.time()
    print("STARTTTTTT")
    ray.init(redis_address="localhost:6379", redirect_output=True)
    horizon = flow_params["env"].horizon
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = PARALLEL_ROLLOUTS
    config["timesteps_per_batch"] = horizon * N_ROLLOUTS
    config["vf_loss_coeff"] = 1.0
    config["kl_target"] = 0.02
    config["use_gae"] = True
    config["horizon"] = horizon
    config["gamma"] = grid_search([0.995, 0.999, 1.0])  # discount rate
    config["model"].update({"fcnet_hiddens": grid_search([[100, 50, 25], [256, 256], [32, 32]])})
    config["lambda"] = grid_search([0.9, 0.99])
    config["sgd_batchsize"] = grid_search([64, 1024, min(16 * 1024, config["timesteps_per_batch"])])
    config["num_sgd_iter"] = grid_search([10, 30])
    config["entropy_coeff"] = grid_search([0, -1e-4, 1e-4])
    config["kl_coeff"] = grid_search([0.0, 0.2])
    #config["clip_param"] = grid_search([0.2, 0.3])
    #config["ADB"] = grid_search([True, False])
    config["clip_param"] = 0.2
    config["ADB"] = False

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
                "training_iteration": 1,
            },
            "repeat": 3,
            "trial_resources": {
                "cpu": 1,
                "gpu": 0,
                "extra_cpu": PARALLEL_ROLLOUTS - 1,
            },
        },
    })

    end = time.time()

    print("IT TOOK " + str(end-start))
