import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import LineCollection



# import ipdb; ipdb.set_trace()

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
    # plt.ylim(ymin=500)

    for col in cols: line = ax.add_collection(col)
    cbar = plt.colorbar(line, ax = ax)
    cbar.set_label('Velocity (m/s)', fontsize = 20)

    savepath = os.path.join(".", filename)
    plt.savefig(savepath)
    # plt.show()

def test():
    # 'velocities': np.zeros((self.steps, self.vehicles.num_vehicles)),
    # 'positions': np.zeros((self.steps, self.vehicles.num_vehicles))} 

    pos = np.array([[1., 2., 1., 5.],
                    [2., 3., 4., 5.],
                    [3., 4., 5., 5.]])
    vel = np.array([[3., 4., 5., 0.],
                    [3., 4., 5., 0.],
                    [3., 4., 5., 0.]])
    t = np.arange(3) * 0.1
    space_time_diagram(pos, vel, t, "test")


def main():
    filename = "spacetime_baseline2" + "_bot"
    
    cdict = {'red'  :  ((0., 0., 0.), (0.2, 1., 1.), (0.6, 1., 1.), (1., 0., 0.)),
            'green':  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 1., 1.), (1., 1., 1.)),
            'blue' :  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 0., 0.), (1., 0., 0.))}

    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
    
    
    pos = np.arange(5)
    vel =np.arange(5)
    # pos = np.genfromtxt ('pos_bot.csv', delimiter=",") * 0.1
    # vel = np.genfromtxt ('vel_bot.csv', delimiter=",") 
    # pos = np.genfromtxt ('pos_left.csv', delimiter=",") * 0.1
    # vel = np.genfromtxt ('vel_left.csv', delimiter=",")

    steps = pos.shape[0]
    t = np.arange(steps) * 1.0
    max_speed = np.max(vel)
    # import ipdb; ipdb.set_trace()
    title = "Space-time Diagram for vehicles on horizontal"
    space_time_diagram(pos, vel, t, title, max_speed, filename)

if __name__ == "__main__":
    # main()
    test()