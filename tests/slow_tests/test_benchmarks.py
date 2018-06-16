import unittest
import os
import json

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.rllib import make_create_env, FlowParamsEncoder

from flow.benchmarks.figureeight0 import flow_params as figgureeight0
from flow.benchmarks.figureeight1 import flow_params as figgureeight1
from flow.benchmarks.figureeight2 import flow_params as figgureeight2
from flow.benchmarks.merge0 import flow_params as merge0
from flow.benchmarks.merge1 import flow_params as merge1
from flow.benchmarks.merge2 import flow_params as merge2
from flow.benchmarks.bottleneck0 import flow_params as bottleneck0
from flow.benchmarks.bottleneck1 import flow_params as bottleneck1
from flow.benchmarks.bottleneck2 import flow_params as bottleneck2
from flow.benchmarks.grid0 import flow_params as grid0
from flow.benchmarks.grid1 import flow_params as grid1

os.environ["TEST_FLAG"] = "True"

ALL_FLOW_PARAMS = [
    figgureeight0, figgureeight1, figgureeight2, merge0, merge1, merge2,
    bottleneck0, bottleneck1, bottleneck2, grid0, grid1,
]


class TestBenchmarksRLlib(unittest.TestCase):

    def test_ppo(self):
        """Tests each of the benchmarks in flow/benchmarks on the PPO algorithm
        in RLLib."""
        # initialize a ray instance
        ray.init(redirect_output=False)

        for flow_params in ALL_FLOW_PARAMS:
            horizon = flow_params["env"].horizon

            # get the env name and a creator for the environment
            create_env, env_name = make_create_env(params=flow_params)

            config = ppo.DEFAULT_CONFIG.copy()
            config["num_workers"] = 2
            config["timesteps_per_batch"] = horizon
            config["horizon"] = horizon

            # save the flow params for replay
            flow_json = json.dumps(flow_params, cls=FlowParamsEncoder,
                                   sort_keys=True, indent=4)
            config['env_config']['flow_params'] = flow_json

            # Register as rllib env
            register_env(env_name, create_env)

            run_experiments({
                flow_params["exp_tag"]: {
                    "run": "PPO",
                    "env": env_name,
                    "config": {
                        **config
                    },
                    "checkpoint_freq": 5,
                    "max_failures": 999,
                    "stop": {"training_iteration": 1},
                    "repeat": 1,
                    "trial_resources": {
                        "cpu": 1,
                        "gpu": 0,
                        "extra_cpu": 1,
                    },
                },
            })


if __name__ == "__main__":
    unittest.main()
