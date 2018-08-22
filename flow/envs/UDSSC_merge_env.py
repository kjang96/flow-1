from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
    # number of observable vehicles preceding the rl vehicle
    "n_preceding": 2,
    # number of observable vehicles following the rl vehicle
    "n_following": 2,
    # number of observable merging-in vehicle from the larger loop
    "n_merging_in": 2,
}


class UDSSCMergeEnv(Env):
    """Environment for training cooperative merging behavior in a closed loop
    merge scenario.

    WARNING: only supports 1 RL vehicle

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * n_preceding: number of observable vehicles preceding the rl vehicle
    * n_following: number of observable vehicles following the rl vehicle
    * n_merging_in: number of observable merging-in vehicle from the larger
      loop

    States
        Observation space is the single RL vehicle, the 2 vehicles preceding
        it, the 2 vehicles following it, the next 2 vehicles to merge in, the
        queue length, and the average velocity of the inner and outer rings.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams. The
        actions are assigned in order of a sorting mechanism (see Sorting).

    Rewards
        Rewards system-level proximity to a desired velocity while penalizing
        variances in the headways between consecutive vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.

    Sorting
        Vehicles in this environment are sorted by their get_x_by_id values.
        The vehicle ids are then sorted by rl vehicles, then human-driven
        vehicles.
    """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.
                               format(p))

        self.n_preceding = env_params.additional_params["n_preceding"]
        self.n_following = env_params.additional_params["n_following"]
        self.n_merging_in = env_params.additional_params["n_merging_in"]
        self.n_obs_vehicles = \
            1 + self.n_preceding + self.n_following + 2*self.n_merging_in
        self.ring_radius = scenario.net_params.additional_params["ring_radius"]
        self.obs_var_labels = \
            ["speed", "pos", "queue_length", "velocity_stats"]
        self.accels = []

        super().__init__(env_params, sumo_params, scenario)

    @property
    def observation_space(self):
        box = Box(low=0.,
                  high=1,
                  shape=(self.n_obs_vehicles * 2,),
                  dtype=np.float32)
        return box

    @property
    def action_space(self):
        return Box(low=-np.abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"],
                   shape=(self.vehicles.num_rl_vehicles,),
                   dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        vel_reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        # Use a similar weighting of of the headway reward as the velocity
        # reward
        # max_cost = np.array([self.env_params.additional_params[
        #                          "target_velocity"]] *
        #                     self.vehicles.num_vehicles)
        # max_cost = np.linalg.norm(max_cost)
        # normalization = self.scenario.length / self.vehicles.num_vehicles
        # headway_reward = 0.2 * max_cost * rewards.penalize_headway_variance(
        #     self.vehicles, self.sorted_extra_data, normalization)
        # return vel_reward + headway_reward
        return vel_reward

    def get_state(self, **kwargs):
        """
        Want to include: 
        * dist, vel of vehicles closest to roundabout
        * dist, 2 vehicles ahead, 2 vehicles behind


        """
        try:
            # Get normalization factors 
            circ = float(2 * np.pi * self.ring_radius)
            max_speed = self.scenario.max_speed 
            merge_0_norm = self.scenario.edge_length('merge_in_0') + \
                        self.scenario.edge_length('inflow_0')
            merge_1_norm = self.scenario.edge_length('merge_in_1') + \
                        self.scenario.edge_length('inflow_1') 
            # RL POS AND VEL
            rl_pos = [self.get_x_by_id('rl_0') / circ]
            rl_vel = [self.vehicles.get_speed('rl_0') / max_speed]

            # DISTANCES
            merge_id_0, merge_id_1 = self.k_closest_to_merge(self.n_merging_in) # TODO check this is sorted
            merge_dists_0 = self.process(self._dist_to_merge_0(merge_id_0),
                                        length=self.n_merging_in,
                                        normalizer=merge_0_norm)
            merge_dists_1 = self.process(self._dist_to_merge_1(merge_id_1),
                                        length=self.n_merging_in,
                                        normalizer=merge_1_norm)

            # Get (ID, dist_from_RL) for the k vehicles closest to 
            # the RL vehicle. 0 if there is no k_closest.
            tailway, headway = self.k_closest_to_rl('rl_0', self.n_preceding) #todo
            tailway_ids = [x[0] for x in tailway]
            tailway_dists = [x[1] for x in tailway]
            tailway_dists = self.process(tailway_dists,
                                        length=self.n_preceding,
                                        normalizer=circ)

            headway_ids = [x[0] for x in headway]
            headway_dists = [x[1] for x in headway]
            headway_dists = self.process(headway_dists,
                                        length=self.n_preceding,
                                        normalizer=circ)


            # VELOCITIES
            merge_0_vel = self.process(self.vehicles.get_speed(merge_id_0),
                                    length=self.n_merging_in,
                                    normalizer=max_speed)
            merge_1_vel = self.process(self.vehicles.get_speed(merge_id_1),
                                    length=self.n_merging_in,
                                    normalizer=max_speed)
            tailway_vel = self.process(self.vehicles.get_speed(tailway_ids),
                                    length=self.n_preceding,
                                    normalizer=max_speed)
            headway_vel = self.process(self.vehicles.get_speed(headway_ids),
                                    length=self.n_following,
                                    normalizer=max_speed)

            state = np.array(np.concatenate([rl_pos, rl_vel,
                                            merge_dists_0, merge_0_vel,
                                            merge_dists_1, merge_1_vel,
                                            tailway_dists, tailway_vel,
                                            headway_dists, headway_vel]))
                                            
        except:
            return np.zeros(self.n_obs_vehicles*2)

        return state

    def sort_by_position(self):
        """
        See parent class

        Instead of being sorted by a global reference, vehicles in this
        environment are sorted with regards to which ring this currently
        reside on.
        """
        pos = [self.get_x_by_id(veh_id) for veh_id in self.vehicles.get_ids()]
        sorted_indx = np.argsort(pos)
        sorted_ids = np.array(self.vehicles.get_ids())[sorted_indx]

        sorted_human_ids = [veh_id for veh_id in sorted_ids
                            if veh_id not in self.vehicles.get_rl_ids()]

        sorted_rl_ids = [veh_id for veh_id in sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]

        sorted_separated_ids = sorted_human_ids + sorted_rl_ids

        return sorted_separated_ids, sorted_ids


    def k_closest_to_merge(self, k):
        """
        Return a list of IDs, NO DISTANCES. NO TRUNCATION.

        [veh_id, veh_id, veh_id] 
        """
        edges_0 = ["merge_in_0", "inflow_0", ":e_0",":e_5", ":c_0"]
        edges_1 = ["merge_in_1", "inflow_1", ":g_3", ":g_5", ":a_0"]

        close_0 = sorted(self.vehicles.get_ids_by_edge(["merge_in_0", "inflow_0"]), 
                         key=lambda veh_id:
                         -self.get_x_by_id(veh_id))

        close_1 = sorted(self.vehicles.get_ids_by_edge(["merge_in_1", "inflow_1"]), 
                         key=lambda veh_id:
                         -self.get_x_by_id(veh_id))

        if len(close_0) > k:
            close_0 = close_0[:k]

        if len(close_1) > k:
            close_1 = close_1[:k]
        
        return close_0, close_1

    def k_closest_to_rl(self, rl_id, k):
        """ 
        Return a list of ids and  distances to said rl vehicles

        In the form:
        [(veh_id, dist), (veh_id, dist)]

        TODO: Is padding necessary in this step? I'm leaning toward not.

        ASSUME NO TRUNCATION. PAD WITH ZEROS IN GET_STATE

        """

        # Arrays holding veh_ids of vehicles before/after the rl_id.
        # Will always be sorted by distance to rl_id 
        k_tailway = [] 
        k_headway = []

        # Prep.
        route = ["top", "left", "bottom", "right"]
        route = ["top", ":c_2", "left", ":d_2", "bottom", ":a_2", "right", ":b_2"]
        rl_edge = self.vehicles.get_edge(rl_id)
        if rl_edge == "":
            return [], []
        rl_index = route.index(rl_edge) 
        rl_x = self.get_x_by_id(rl_id)
        rl_pos = self.vehicles.get_position(rl_id)

        # Get preceding first.
        for i in range(rl_index, rl_index-2, -1): # Curr  edge and preceding edge
            if i == rl_index: # Same edge as rl_id
                veh_ids = [(v, rl_pos - self.vehicles.get_position(v)) 
                           for v in self.vehicles.get_ids_by_edge(route[i]) 
                           if self.vehicles.get_position(v) < rl_pos]
            else: # Preceding edge 
                edge_len = self.scenario.edge_length(route[i])
                veh_ids = [(v, rl_pos + (edge_len - self.vehicles.get_position(v))) 
                           for v in self.vehicles.get_ids_by_edge(route[i])]
            sorted_vehs = sorted(veh_ids, key=lambda v: self.get_x_by_id(v[0]))
            # k_tailway holds veh_ids in decreasing order of closeness 
            k_tailway = sorted_vehs + k_tailway

        # Get headways second.
        for i in range(rl_index, rl_index+2):
            i = i % len(route)
            # If statement is to cover the case of overflow in get_x 
            if i == rl_index: # Same edge as rl_id
                veh_ids = [(v, self.vehicles.get_position(v) - rl_pos) 
                           for v in self.vehicles.get_ids_by_edge(route[i]) 
                           if self.vehicles.get_position(v) > rl_pos]
            else:
                rl_dist = self.scenario.edge_length(rl_edge) - \
                          self.vehicles.get_position(rl_id)
                veh_ids = [(v, self.vehicles.get_position(v) + rl_dist)
                           for v in self.vehicles.get_ids_by_edge(route[i])]
            # The following statement is safe because it's being done
            # by one edge at a time
            sorted_vehs = sorted(veh_ids, key=lambda v: self.get_x_by_id(v[0]))
            k_headway += sorted_vehs

        return k_tailway[::-1], k_headway

    def _dist_to_merge_1(self, veh_id):
        reference = self.scenario.total_edgestarts_dict["merge_in_1"] + \
                    self.scenario.edge_length("merge_in_1")
        distances = [reference - self.get_x_by_id(v)
                     for v in veh_id]
        return distances

    def _dist_to_merge_0(self, veh_id):
        reference = self.scenario.total_edgestarts_dict["merge_in_0"] + \
                    self.scenario.edge_length("merge_in_0")
        distances = [reference - self.get_x_by_id(v)
                     for v in veh_id]
        return distances

    def process(self, state, length=None, normalizer=1):
        """
        Takes in a list, returns a normalized version of the list
        with padded zeros at the end 
        """
        if length: # Truncation or padding
            if len(state) < length:
                state += [0.] * (length - len(state))
            else:
                state = state[:length]
        state = [x / normalizer for x in state]
        return state
        
    def additional_command(self):
        try: 
            # self.velocities.append(np.mean(self.vehicles.get_speed(self.vehicles.get_controlled_ids())))
            self.velocities.append(np.mean(self.vehicles.get_speed(self.vehicles.get_ids())))
        except AttributeError:
            self.velocities = []
            # self.velocities.append(np.mean(self.vehicles.get_speed(self.vehicles.get_controlled_ids())))
            self.velocities.append(np.mean(self.vehicles.get_speed(self.vehicles.get_ids())))
        