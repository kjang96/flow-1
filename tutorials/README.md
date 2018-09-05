# Flow Tutorials

## Setup

1. Make sure you have Python 3 installed (we recommend using the [Anaconda 
   Python distribution](https://www.continuum.io/downloads)).
2. **Install Jupyter** with `pip install jupyter`. Verify that you can start
   a Jupyter notebook with the command `jupyter-notebook`.
3. **Install Flow** by executing the following [installation instructions](
   https://berkeleyflow.readthedocs.io/en/latest/flow_setup.html).

## Tutorials

Each file ``tutorials/tutorial*.ipynb`` is a separate tutorial. They can be
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

**Tutorial 1:** Running SUMO simulations in Flow.

**Tutorial 2:** Running RLlib experiments for mixed-autonomy traffic.

**Tutorial 3:** Running rllab experiments for mixed-autonomy traffic.

**Tutorial 4:** Saving and visualizing resuls from non-RL simulations and 
testing simulations in the presence of an RLlib/rllab agent.

**Tutorial 5:** Creating custom scenarios.

**Tutorial 6:** Creating custom environments.

**Tutorial 7:** Creating custom controllers.

**Tutorial 8:** Traffic lights.

**Tutorial 9:** Running simulations with inflows of vehicles.
