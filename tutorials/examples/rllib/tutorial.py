# ring road scenario class
scenario_name = "LoopScenario"

# ring road generator class
generator_name = "CircleGenerator"

# input parameter classes to the scenario class
from flow.core.params import NetParams, InitialConfig

# name of the scenario
name = "training_example"

# network-specific parameters
from flow.scenarios.loop.loop_scenario import ADDITIONAL_NET_PARAMS
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

# initial configuration to vehicles
initial_config = InitialConfig(spacing="uniform", perturbation=1)

# vehicles class
from flow.core.vehicles import Vehicles

# vehicles dynamics models
from flow.controllers import IDMController, ContinuousRouter

vehicles = Vehicles()
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=21)

from flow.controllers import RLController

vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=1)

from flow.core.params import SumoParams

sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo")

from flow.core.params import EnvParams

# Define horizon as a variable to ensure consistent use across notebook
HORIZON=100

env_params = EnvParams(
    # length of one rollout
    horizon=HORIZON,

    additional_params={
        # maximum acceleration of autonomous vehicles
        "max_accel": 1,
        # maximum deceleration of autonomous vehicles
        "max_decel": 1,
        # bounds on the ranges of ring road lengths the autonomous vehicle 
        # is trained on
        "ring_length": [220, 270],
    },
)

env_name = "WaveAttenuationPOEnv"

# Creating flow_params. Make sure the dictionary keys are as specified. 
flow_params = dict(
    exp_tag=name,  # experiment name
    env_name=env_name,  # environment name as specified earlier
    scenario=scenario_name,  # scenario name as specified earlier
    generator=generator_name,  # generator name as specified earlier
    sumo=sumo_params,  # params objects as created earlier
    env=env_params,
    net=net_params,
    veh=vehicles,  # vehicles object
    initial=initial_config)

import json

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.rllib import make_create_env, FlowParamsEncoder

# number of parallel workers
PARALLEL_ROLLOUTS = 1
# number of rollouts per training iteration
N_ROLLOUTS = 20

ray.init(redirect_output=False)

config=ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = PARALLEL_ROLLOUTS  # number of parallel workers
config["timesteps_per_batch"] = HORIZON * N_ROLLOUTS  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [16, 16]})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97  
config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])  # stochastic gradient descent
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 10  # number of SGD iterations
config["horizon"] = HORIZON  # rollout horizon

# save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  # generating a string version of flow_params
config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env with Gym
register_env(env_name, create_env)

trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": "PPO",  # RL algorithm to run
        "env": env_name,  # environment name generated earlier
        "config": {  # configuration params (must match "run" value)
            **config
        },
        "checkpoint_freq": 1,  # number of iterations between checkpoints
        "max_failures": 999,
        "stop": {  # stopping conditions
            "training_iteration": 1,  # number of iterations to stop after
        },
        "repeat": 1,  # number of times to repeat training
        "trial_resources": {
            "cpu": 1,
            "gpu": 0,
            "extra_cpu": PARALLEL_ROLLOUTS - 1,
        },
    },
})