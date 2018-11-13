import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from flow.core import rewards
from flow.envs.base_env import Env

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
}


class UDSSCGridEnv(Env):
    """Environment used to train traffic lights on the intersection part of
    the UDSSC network. 

    Required from env_params:

    * switch_time: minimum switch time for each traffic light (in seconds).
      Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL
      options are "controlled" and "actuated"

    States
        An observation is the distance of each vehicle to its intersection, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.
    """

    def __init__(self, env_params, sumo_params, scenario):
        self.grid_array = scenario.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sumo_params, scenario)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.vehicles.num_vehicles)),
            'velocities': np.zeros((self.steps, self.vehicles.num_vehicles)),
            'positions': np.zeros((self.steps, self.vehicles.num_vehicles))
        }
        self.node_mapping = scenario.get_node_mapping()

        # keeps track of the last time the light was allowed to change.
        self.last_change = np.zeros((self.rows * self.cols, 3))

        # when this hits min_switch_time we change from yellow to red
        # the second column indicates the direction that is currently being
        # allowed to flow. 0 is flowing top to bottom, 1 is left to right
        # For third column, 0 signifies yellow and 1 green or red
        self.min_switch_time = env_params.additional_params["switch_time"]
        # import ipdb; ipdb.set_trace()
        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.traci_connection.trafficlight.setRedYellowGreenState(
                    'center' + str(i), "GGGGGGGGGGGG")
                self.last_change[i, 2] = 1

        # Additional Information for Plotting
        self.edge_mapping = {"top": [], "bot": [], "right": [], "left": []}
        for i, veh_id in enumerate(self.vehicles.get_ids()):
            edge = self.vehicles.get_edge(veh_id)
            for key in self.edge_mapping:
                if key in edge:
                    self.edge_mapping[key].append(i)
                    break

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 2)

        # used while computing the reward
        self.env_params.additional_params["target_velocity"] = \
            max(self.scenario.speed_limit(edge)
                for edge in self.scenario.get_edge_list())

        # used during visualization
        self.observed_ids = []

        # Maintained as a stack, only apply_rl_actions to the top 1
        self.rl_stack = []
        self.rl_stack_2 = []

    @property
    def action_space(self):
        """See class definition."""
        return Box(low=-np.abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"],
                   shape=(2,),
                   dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.total_obs = self.num_observed * 2 * 4
                         
        box = Box(low=0.,
                  high=1,
                  shape=(self.total_obs,),
                  dtype=np.float32)          
        return box

    def get_state(self):
        """
        Returns self.num_observed number of vehicles closest to each traffic
        light and for each vehicle its velocity, distance to intersection,
        edge_number traffic light state. This is partially observed. 

        """
        speeds = []
        dist_to_intersec = []
        # edge_number = []
        max_speed = max(
            self.scenario.speed_limit(edge)
            for edge in self.scenario.get_edge_list())
        max_dist = max(self.scenario.short_length, self.scenario.long_length,
                       self.scenario.inner_length)
        all_observed_ids = []

        for node, edges in self.scenario.get_node_mapping():
            for edge in edges:
                observed_ids = \
                    self.k_closest_to_intersection(edge, self.num_observed)
                all_observed_ids += observed_ids

                # check which edges we have so we can always pad in the right
                # positions
                speeds += [
                    self.vehicles.get_speed(veh_id) / max_speed
                    for veh_id in observed_ids
                ]
                dist_to_intersec += [
                    (self.scenario.edge_length(self.vehicles.get_edge(veh_id))
                     - self.vehicles.get_position(veh_id)) / max_dist
                    for veh_id in observed_ids
                ]
                # edge_number += \
                #     [self._convert_edge(self.vehicles.get_edge(veh_id))
                #      / (self.scenario.num_edges - 1)
                #      for veh_id in observed_ids]

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    speeds += [0] * diff
                    dist_to_intersec += [0] * diff
                    # edge_number += [0] * diff

        state = np.array(
            np.concatenate([
                speeds, dist_to_intersec,# edge_number,
            ]))
        return state

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # Apply noise
        if "rl_action_noise" in self.env_params.additional_params:
            rl_action_noise = self.env_params.additional_params["rl_action_noise"]
            for i, rl_action in enumerate(rl_actions):
                perturbation = np.random.normal(0, rl_action_noise) # 0.7 is arbitrary. but since accels are capped at +- 1 i don't want thi sto be too big
                rl_actions[i] = rl_action + perturbation

            # Reclip
            if isinstance(self.action_space, Box):
                rl_actions = np.clip(
                    rl_actions,
                    a_min=self.action_space.low,
                    a_max=self.action_space.high)
        
        # Curation
        removal = [] 
        removal_2 = []
        for rl_id in self.rl_stack:
            if rl_id not in self.vehicles.get_rl_ids():
                removal.append(rl_id)
        for rl_id in self.rl_stack_2:
            if rl_id not in self.vehicles.get_rl_ids():
                removal_2.append(rl_id)
        for rl_id in removal:
            self.rl_stack.remove(rl_id)
        for rl_id in removal_2:
            self.rl_stack_2.remove(rl_id)

        # Apply RL Actions
        if self.rl_stack:
            rl_id = self.rl_stack[0]
            # if self.in_control(rl_id):
            self.apply_acceleration([rl_id], rl_actions[:1])
            print(rl_id, rl_actions[:1])

        if self.rl_stack_2:
            rl_id_2 = self.rl_stack_2[0]
            # if self.in_control(rl_id_2):
            self.apply_acceleration([rl_id_2], rl_actions[1:])
            print(rl_id_2, rl_actions[1:])

    def compute_reward(self, state, rl_actions, **kwargs):
        """See class definition."""
        min_delay = rewards.min_delay(self)
        return min_delay

    # ===============================
    # ============ UTILS ============
    # ===============================

    def record_obs_var(self):
        """
        Records velocities and edges in self.obs_var_labels at each time
        step. This is used for plotting.
        """
        for i, veh_id in enumerate(self.vehicles.get_ids()):
            self.obs_var_labels['velocities'][
                self.time_counter - 1, i] = self.vehicles.get_speed(veh_id)
            self.obs_var_labels['edges'][self.time_counter - 1, i] = \
                self._convert_edge(self.vehicles.get_edge(veh_id))
            x = self.get_x_by_id(veh_id)
            if x > 2000:  # hardcode
                x = 0
            self.obs_var_labels['positions'][self.time_counter - 1, i] = x

    def get_distance_to_intersection(self, veh_ids):
        """Determines the smallest distance from the current vehicle's position
        to any of the intersections.

        Parameters
        ----------
        veh_ids: str
            vehicle identifier

        Returns
        -------
        tup
            1st element: distance to closest intersection
            2nd element: intersection ID (which also specifies which side of
            the intersection the vehicle will be arriving at)
        """
        if isinstance(veh_ids, list):
            return [self.find_intersection_dist(veh_id) for veh_id in veh_ids]
        else:
            return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from the vehicle's current position to the position
        of the node it is heading toward."""
        edge_id = self.vehicles.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.scenario.edge_length(edge_id)
        relative_pos = self.vehicles.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def sort_by_intersection_dist(self):
        """Sorts the vehicle ids of vehicles in the network by their distance
        to the intersection.

        Returns
        -------
        sorted_ids: list
            a list of all vehicle IDs sorted by position
        sorted_extra_data: list or tuple
            an extra component (list, tuple, etc...) containing extra sorted
            data, such as positions. If no extra component is needed, a value
            of None should be returned
        """
        ids = self.vehicles.get_ids()
        sorted_indx = np.argsort(self.get_distance_to_intersection(ids))
        sorted_ids = np.array(ids)[sorted_indx]
        return sorted_ids

    def _convert_edge(self, edges):
        """Converts the string edge to a number.

        Start at the bottom left vertical edge and going right and then up, so
        the bottom left vertical edge is zero, the right edge beside it  is 1.

        The numbers are assigned along the lowest column, then the lowest row,
        then the second lowest column, etc. Left goes before right, top goes
        before bot.

        The values are are zero indexed.

        Parameters
        ----------
        edges: list <str> or str
            name of the edge(s)

        Returns
        -------
        list <int> or int
            a number uniquely identifying each edge
        """
        if isinstance(edges, list):
            return [self._split_edge(edge) for edge in edges]
        else:
            return self._split_edge(edges)

    def _split_edge(self, edge):
        """Utility function for convert_edge"""
        if edge:
            if edge[0] == ":":  # center
                center_index = int(edge.split("center")[1][0])
                base = ((self.cols + 1) * self.rows * 2) \
                    + ((self.rows + 1) * self.cols * 2)
                return base + center_index + 1
            else:
                pattern = re.compile(r"[a-zA-Z]+")
                edge_type = pattern.match(edge).group()
                edge = edge.split(edge_type)[1].split('_')
                row_index, col_index = [int(x) for x in edge]
                if edge_type in ['bot', 'top']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * (row_index + 1))
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'bot' else edge_num + 1
                if edge_type in ['left', 'right']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * row_index)
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'left' else edge_num + 1
        else:
            return 0

    def additional_command(self):
        """Used to insert vehicles that are on the exit edge and place them
        back on their entrance edge."""
        # for veh_id in self.vehicles.get_ids():
        #     self._reroute_if_final_edge(veh_id)

        [self.vehicles.set_observed(veh_id) for veh_id in self.observed_ids]
        try: 
            self.velocities.append(np.mean(self.vehicles.get_speed(self.vehicles.get_ids())))
        except AttributeError:
            self.velocities = []
            self.velocities.append(np.mean(self.vehicles.get_speed(self.vehicles.get_ids())))
        
        # Curate rl_stack
        for veh_id in self.vehicles.get_rl_ids():
            if veh_id not in self.rl_stack and self.vehicles.get_edge(veh_id) == "bot0_0":
                self.rl_stack.append(veh_id)
            elif veh_id not in self.rl_stack_2 and self.vehicles.get_edge(veh_id) == "top0_1":
                self.rl_stack_2.append(veh_id)
        # Curate second rl_stack
        removal = [] 
        removal_2 = []
        for rl_id in self.rl_stack:
            if rl_id not in self.vehicles.get_rl_ids():
                removal.append(rl_id)
        for rl_id in self.rl_stack_2:
            if rl_id not in self.vehicles.get_rl_ids():
                removal_2.append(rl_id)
        for rl_id in removal:
            self.rl_stack.remove(rl_id)
        for rl_id in removal_2:
            self.rl_stack_2.remove(rl_id)
        # Color RL vehicles
        rl_control = self.rl_stack[:min(1, len(self.rl_stack))]
        rl_control_2 = self.rl_stack_2[:min(1, len(self.rl_stack_2))]

        try:
            for veh_id in rl_control + rl_control_2:
                self.traci_connection.vehicle.setColor(
                            vehID=veh_id, color=(0, 255, 255, 255))
        except:
            pass

    def _reroute_if_final_edge(self, veh_id):
        """Checks if an edge is the final edge. If it is return the route it
        should start off at."""
        edge = self.vehicles.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')
        row_index, col_index = [int(x) for x in edge]

        # find the route that we're going to place the vehicle on if we are
        # going to remove it
        route_id = None
        if edge_type == 'bot' and col_index == self.cols:
            route_id = "bot{}_0".format(row_index)
        elif edge_type == 'top' and col_index == 0:
            route_id = "top{}_{}".format(row_index, self.cols)
        elif edge_type == 'left' and row_index == 0:
            route_id = "left{}_{}".format(self.rows, col_index)
        elif edge_type == 'right' and row_index == self.rows:
            route_id = "right0_{}".format(col_index)

        if route_id is not None:
            route_id = "route" + route_id
            # remove the vehicle
            self.traci_connection.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            type_id = self.vehicles.get_state(veh_id, "type")
            lane_index = self.vehicles.get_lane(veh_id)
            self.traci_connection.vehicle.addFull(
                veh_id,
                route_id,
                typeID=str(type_id),
                departLane=str(lane_index),
                departPos="0",
                departSpeed="max")
            speed_mode = self.vehicles.type_parameters[type_id]["speed_mode"]
            self.traci_connection.vehicle.setSpeedMode(veh_id, speed_mode)

    def k_closest_to_intersection(self, edges, k):
        """
        Return the veh_id of the k closest vehicles to an intersection for
        each edge. Performs no check on whether or not edge is going toward an
        intersection or not. Does no padding
        """
        if k < 0:
            raise IndexError("k must be greater than 0")
        dists = []

        def sort_lambda(veh_id):
            return self.get_distance_to_intersection(veh_id)

        if isinstance(edges, list):
            for edge in edges:
                vehicles = self.vehicles.get_ids_by_edge(edge)
                dist = sorted(
                    vehicles,
                    key=sort_lambda
                )
                dists += dist[:k]
        else:
            vehicles = self.vehicles.get_ids_by_edge(edges)
            dist = sorted(
                vehicles,
                key=lambda veh_id: self.get_distance_to_intersection(veh_id))
            dists += dist[:k]
        return dists
