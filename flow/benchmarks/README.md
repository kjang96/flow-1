# Flow Benchmarks

This folder contains several (blank) for mixed-autonomy traffic.

## Description of Benchmarks

For a detailed description of each benchmark, we refer the user to the Flow 
Benchmarks paper, see *Citing Flow Benchmarks*. At a slightly higher level,
the traffic benchmarks presented in this folder are as follows:

**Stabilizing open network merges:** description. how discrete problems differ.
- `flow.benchmarks.merge0` description
- `flow.benchmarks.merge1` description
- `flow.benchmarks.merge2` description

**Increasing throughput in a typical bottleneck geometry:** description. how 
discrete problems differ.
- `flow.benchmarks.bottleneck0` description
- `flow.benchmarks.bottleneck1` description
- `flow.benchmarks.bottleneck2` description

**Intersection timing in a figure eight:** description. how discrete problems 
differ.
- `flow.benchmarks.lanedrop0` description
- `flow.benchmarks.lanedrop1` description
- `flow.benchmarks.lanedrop2` description

**Traffic light optimization on a grid:** description. how discrete problems 
differ.
- `flow.benchmarks.grid0` description
- `flow.benchmarks.grid1` description
- `flow.benchmarks.grid2` description

## Training and Hyperparameter Tuning with RLLib and rllab

blank

## Training on Custom Algorithms

blank

The below code snippet presents a sample us case of our benchmarks in tandem 
with OpenAI Baselines, a high-quality implementation of several reinforcement 
learning algorithms. In order to test this section of code, we recommend 
installing [OpenAI Baselines](https://www.continuum.io/downloads). 
Alternatively, you can replace the algorithm importation and training sections 
with your own RL algorithm implementation.

```python
# import your custom RL algorithm
from foo import myAlgorithm

# import the experiment-specific parameters from flow.benchmarks
from flow.benchmarks.figureeight0 import flow_params

# import the make_create_env to register the environment with OpenAI gym
from flow.utils.registry import make_create_env

if __name__ == "__main__":
    # the make_create_env function produces a method that can be used to 
    # generate parameterizable gym environments that are compatible with Flow. 
    # This method will run both "register" and "make" (see gym  documentation).
    # If these are supposed to be run within your algorithm/library, we 
    # recommend referring to the make_create_env source code in 
    # flow/utils/registry.py.
    env_name, create_env = make_create_env(flow_params, version=0)

    # create and register the environment with OpenAI Gym
    env = create_env()

    # setup the algorithm with the traffic benchmark and begin training
    alg = myAlgorithm(env_name=env_name)
    alg.train()
```

## Reporting Optimal Scores

In order to encourage (blank).

## Citing Flow Benchmarks

If you use the following benchmarks for academic research, you are highly 
encouraged to cite our paper:

*paper to be specified at a later date
