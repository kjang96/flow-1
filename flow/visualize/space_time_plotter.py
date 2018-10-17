"""
Visualizer for rllib experimenst

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::
    
        python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO

optional_named : ArgumentGroup
    Optional named command-line arguments
parser : ArgumentParser
    Command-line argument parser
required_named : ArgumentGroup
    Required named command-line arguments
"""

import argparse
import json
import importlib
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

import gym
import ray
import ray.rllib.ppo as ppo
from ray.rllib.agent import get_agent_class
from ray.tune.registry import get_registry, register_env as register_rllib_env

from flow.core.util import unstring_flow_params, get_rllib_config, get_flow_params

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO
OR
    python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO \
        --module cooperative_merge --flowenv TwoLoopsMergePOEnv \
        --exp_tag cooperative_merge_example    

Here the arguments are:
1 - the number of the checkpoint
PPO - the name of the algorithm the code was run with
cooperative_merge - the run script
TwoLoopsMergePOEnv - the gym environment that was used
cooperative_merge_example - Not actually used. Anything can be passed here.
"""
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a reinforcement learning agent "
                "given a checkpoint.", epilog=EXAMPLE_USAGE)

parser.add_argument(
    "result_dir", type=str, help="Directory containing results")
parser.add_argument(
    "checkpoint_num", type=str, help="Checkpoint number.")

required_named = parser.add_argument_group("required named arguments")
required_named.add_argument(
    "--run", type=str, required=True,
    help="The algorithm or model to train. This may refer to the name "
         "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
         "user-defined trainable function or class registered in the "
         "tune registry.")

optional_named = parser.add_argument_group("optional named arguments")
optional_named.add_argument(
    '--num_rollouts', type=int, default=1,
    help="The number of rollouts for plotting.")
optional_named.add_argument(
    '--module', type=str, default='',
    help='Location of the make_create_env function to use')
optional_named.add_argument(
    '--flowenv', type=str, default='',
    help='Flowenv being used')
optional_named.add_argument(
    '--exp_tag', type=str, default='',
    help='Experiment tag')


def main():
    args = parser.parse_args()

    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)
    # rllib_params = get_rllib_params(result_dir)

    # gamma = rllib_params['gamma']
    # horizon = rllib_params['horizon']
    # hidden_layers = rllib_params['hidden_layers']

    if args.module:
        module_name = 'examples.rllib.' + args.module
        env_module = importlib.import_module(module_name)

        make_create_env = env_module.make_create_env
        flow_params = env_module.flow_params

        flow_env_name = args.flowenv
        exp_tag = args.exp_tag
    else:
        flow_params, make_create_env = get_flow_params(result_dir)

        flow_env_name = flow_params['flowenv']
        exp_tag = flow_params['exp_tag']

    ray.init(num_cpus=1)

    # config = ppo.DEFAULT_CONFIG.copy()
    # config['horizon'] = horizon
    # config["model"].update({"fcnet_hiddens": hidden_layers})
    # config["gamma"] = gamma

    # Overwrite config for rendering purposes
    config["num_workers"] = 1

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(flow_env_name, flow_params,
                                           version=0, sumo="sumo")
    register_rllib_env(env_name, create_env)

    agent_cls = get_agent_class(args.run)
    agent = agent_cls(env=env_name, registry=get_registry(), config=config)
    checkpoint = result_dir + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    # Create and register a new gym environment for rendering rollout
    create_render_env, env_render_name = make_create_env(flow_env_name,
                                                         flow_params,
                                                         version=1,
                                                         sumo="sumo")
    env = create_render_env(None)
    env_num_steps = env.env.env_params.additional_params['num_steps']
    if env_num_steps != config['horizon']:
        print("WARNING: mismatch of experiment horizon and rendering horizon "
              "{} != {}".format(horizon, env_num_steps))
    rets = []

    for i in range(args.num_rollouts):
        state = env.reset()
        done = False
        ret = 0
        while not done:
            if isinstance(state, list):
                state = np.concatenate(state)
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            ret += reward
        rets.append(ret)
        print("Return:", ret)
    print("Average Return", np.mean(rets))

    # print("Average Return", np.mean(rets))
    extra_state = env.env.obs_var_labels
    edge_mapping = env.env.edge_mapping
    velocities = extra_state['velocities']#[:400] #[:,10:]
    positions = extra_state['positions']#[:400] / 5#[:,10:]
    # import ipdb; ipdb.set_trace()
    t = np.arange(env_num_steps) * 0.1
    # bot_veh = edge_mapping["bot"]
    filename = "spacetime_baseline1"
    plot_edge(velocities, positions, edge_mapping["bot"], "bot", filename + "_bot")
    plot_edge(velocities, positions, edge_mapping["top"], "top", filename + "_top")
    plot_edge(velocities, positions, edge_mapping["right"], "right", filename + "_right")
    plot_edge(velocities, positions, edge_mapping["left"], "left", filename + "_left")
    # space_time_diagram(positions, velocities, t, "test", 35)

def test():

    pos = np.array([[1., 2., 1., 5.],
                    [2., 3., 4., 5.],
                    [3., 4., 5., 5.]])
    vel = np.array([[3., 4., 5., 0.],
                    [3., 4., 5., 0.],
                    [3., 4., 5., 0.]])
    t = np.arange(3) * 0.1
    space_time_diagram(pos, vel, t, "test")

def plot_edge(vel, pos, indices, edge, filename):
    """
    Edge can be of type top, bot, right, left
    """
    # import ipdb; ipdb.set_trace()
    p = []
    v = []
    steps = pos.shape[0]
    for i in indices:
        if len(p) == 0 and len(v) == 0: 
            p = pos[:,i].reshape((steps, 1))
            v = vel[:,i].reshape((steps, 1))
        else:
            p = np.hstack((p, pos[:,i].reshape((steps, 1))))
            v = np.hstack((v, vel[:,i].reshape((steps, 1))))
    t = np.arange(steps) * 0.1
    max_speed = np.max(v)
    np.savetxt("pos_" + edge + ".csv", p, delimiter=",")
    np.savetxt("vel_" + edge + ".csv", v, delimiter=",")
    # import ipdb; ipdb.set_trace()
    # space_time_diagram(p, v, t, "Spacetime plot for " + edge + " edge", max_speed, filename)

# def save_data(vel, pos):
    
        


def space_time_diagram(pos, speed, time, title, max_speed=8, filename="test"):
    cdict = {'red'  :  ((0., 0., 0.), (0.2, 1., 1.), (0.6, 1., 1.), (1., 0., 0.)),
         'green':  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 1., 1.), (1., 1., 1.)),
         'blue' :  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 0., 0.), (1., 0., 0.))}

    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes()
    norm = plt.Normalize(0, max_speed) # TODO: Make this more modular
    cols = []
    for indx_car in range(pos.shape[1]):
        unique_car_pos = pos[:,indx_car]    

        # discontinuity from wraparound
        disc = np.where(np.abs(np.diff(unique_car_pos)) >= 5)[0]+1
        unique_car_time = np.insert(time, disc, np.nan)
        unique_car_pos = np.insert(unique_car_pos, disc, np.nan)
        unique_car_speed = np.insert(speed[:,indx_car], disc, np.nan)

        points = np.array([unique_car_time, unique_car_pos]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=my_cmap, norm=norm)

        # Set the values used for colormapping
        lc.set_array(unique_car_speed)
        lc.set_linewidth(1.75)
        cols = np.append(cols, lc)

    xmin, xmax = min(time), max(time)
    xbuffer = (xmax - xmin) * 0.025 # 2.5% of range
    ymin, ymax = np.amin(pos), np.amax(pos)
    ybuffer = (ymax - ymin) * 0.025 # 2.5% of range

    ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
    ax.set_ylim(ymin - ybuffer, ymax + ybuffer)
    
    plt.title(title, fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)

    for col in cols: line = ax.add_collection(col)
    cbar = plt.colorbar(line, ax = ax)
    cbar.set_label('Velocity (m/s)', fontsize = 20)

    # plt.show()
    # filename = "spacetime_baseline1"
    savepath = os.path.join(".", filename)
    plt.savefig(savepath)

cdict = {'red'  :  ((0., 0., 0.), (0.2, 1., 1.), (0.6, 1., 1.), (1., 0., 0.)),
         'green':  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 1., 1.), (1., 1., 1.)),
         'blue' :  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 0., 0.), (1., 0., 0.))}

my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

def horz_or_vert(env, rows, cols):
    """
    For a grid that is rows x cols, it returns the edges that are horizontal and the edges that are vertical
    :param rows: number of rows
    :param cols: number of cols
    :return: list of edges that are horizontal, list of edges that are vertical
    """
    # first generate the correct edges
    horz_edges = []
    vert_edges = []
    for row_index in range(rows):
        for col_index in range(cols+1):
            horz_edges += ["bot" + str(row_index) + '_' + str(col_index)]
            horz_edges += ["top" + str(row_index) + '_' + str(col_index)]
    for row_index in range(rows+1):
        for col_index in range(cols):
            vert_edges += ["left" + str(row_index) + '_' + str(col_index)]
            vert_edges += ["right" + str(row_index) + '_' + str(col_index)]

    # then convert them 
    horz = env.convert_edge(horz_edges)
    vert = env.convert_edge(vert_edges)
    return horz, vert


def horz_or_vert_cars(edge_row, horz_edges, vert_edges):
    """
    Takes in a single row from the obs_var matrix, containing the edge num
    and indexed by veh_id. Returns a tuple of cars in horz and cars in vert

    Currently does not support the case where vehicles are on a null edge
    """
    horz_cars = []
    vert_cars = []
    for i, edge in enumerate(edge_row):
        if edge in horz_edges:
            horz_cars.append(i)
        else:
            vert_cars.append(i)
    return horz_cars, vert_cars

    

if __name__ == "__main__":
    main()
    # test()
    