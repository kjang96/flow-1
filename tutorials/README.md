# Flow Tutorial

## Setup

1. Make sure you have Python 3 installed (we recommend using the [Anaconda 
   Python distribution](https://www.continuum.io/downloads)).
2. **Install Jupyter** with `pip install jupyter`. Verify that you can start
   a Jupyter notebook with the command `jupyter-notebook`.
3. **Install Flow** by executing the following [installation instructions](
   https://berkeleyflow.readthedocs.io/en/latest/flow_setup.html).

## Exercises

Each file ``exercises/exercise*.ipynb`` is a separate exercise. They can be
opened in a Jupyter notebook by running the following commands.

```shell
cd <flow-path>/tutorials/exercises
jupyter-notebook
```

Instructions are written in each file. To do each exercise, first run all of
the cells in the Jupyter notebook. Then modify the ones that need to be 
modified in order to prevent any exceptions from being raised. Throughout these
exercises, you may find the [Flow documentation](
https://berkeleyflow.readthedocs.io/en/latest/) helpful. The content of each 
exercise is as follows:

**Exercise 1:** Running sumo simulations in Flow.

**Exercise 2:** Running RLlib experiments for mixed-autonomy traffic.

**Exercise 3:** Running rllab experiments for mixed-autonomy traffic.

**Exercise 4:** Saving and visualizing resuls from non-RL simulations and 
testing simulations in the presence of an rllib/rllab agent.

**Exercise 5:** Creating custom scenarios.

**Exercise 6:** Creating custom environments.

**Exercise 7:** Creating custom controllers.

**Exercise 8:** Traffic lights.

**Exercise 9:** Running simulations with inflows of vehicles.


## Examples

The `tutorials/examples` folder provides several examples demonstrating how 
both simulation and RL-oriented experiments can be setup and executed within 
the Flow framework on a variety of traffic problems. These examples are not 
written on jupyter notebook, but instead are .py files that may be executed 
either from terminal or via an editor. For example, in order to execute the 
sugiyama example in <flow-path>/tutorials/exercises/sumo, we run:

```shell
python3 <flow-path>/tutorials/examples/sumo/sugiyama.py
```

The examples are distributed into the following sections:

**examples/sumo/** contains examples of transportation network with vehicles
following human-dynamical models of driving behavior.

**examples/rllib/** provides similar networks as those presented in the 
previous point, but in the present of autonomous vehicle (AV) or traffic light 
agents being trained through RL algorithms provided by `RLlib`.

**examples/rllab/** provides similar examples as the one above, but where the 
RL agents are controlled and training the RL library `rllab`.
