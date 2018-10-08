from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from math import ceil

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

MERGE_EDGES = [":a_1", "right", ":b_1", "top", ":c_1",
              "left", ":d_1", "bottom", "inflow_1",
              ":g_2", "merge_in_1", ":a_0", ":b_0",
              "merge_out_0", ":e_1", "outflow_0", "inflow_0",
              ":e_0", "merge_in_0", ":c_0", ":d_0",
              "merge_out_1", ":g_0", "outflow_1" ]

ROUNDABOUT_EDGES = [":a_1", "right", ":b_1", "top", ":c_1",
                    "left", ":d_1", "bottom"]

ALL_EDGES = ROUNDABOUT_EDGES + \
            ['inflow_1', ':g_3', 'merge_in_1', ':a_0'
             , ':b_0', 'merge_out_0', ':e_2', 'outflow_0'
             , 'inflow_0', ':e_0', 'merge_in_0', ':c_0'
             , ':d_0', 'merge_out_1', ':g_0', 'outflow_1']


class RoundaboutEnv(Env):
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

        # Maintained as a stack, only apply_rl_actions to the top 1
        self.rl_stack = [] 

        super().__init__(env_params, sumo_params, scenario)

        self.all_edges = self.scenario.specify_absolute_order()
        self.roundabout_edges = self.all_edges[:self.all_edges.index('bottom') + 1]

    @property
    def observation_space(self):
        # Vehicle position and velocity, normalized
        # Queue length x 2
        # Roundabout state = len(MERGE_EDGES) * 3
        # Roundabout full = (ROUNDABOUT_LENGTH // 5)*2 # 2 cols
        # rl_pos_2, the pos in the roundabout: 1
        self.total_obs = self.n_obs_vehicles * 2 + 2 + \
                         int(self.roundabout_length // 5) * 2 + 1
        # import ipdb; ipdb.set_trace()
                         
        box = Box(low=0.,
                  high=1,
                  shape=(self.total_obs,),
                  dtype=np.float32)          
        return box

    @property
    def action_space(self):
        return Box(low=-np.abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"],
                #    shape=(self.vehicles.num_rl_vehicles,),
                   shape=(2,),
                   dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        # Curating rl_stack
        # Remove rl vehicles that are no longer in the system
        # more efficient to keep removal list than to resize continually
        # if 1:
        #     return
        removal = [] 
        for rl_id in self.rl_stack:
            if rl_id not in self.vehicles.get_rl_ids():
                removal.append(rl_id)
        for rl_id in removal:
            self.rl_stack.remove(rl_id)
        if self.rl_stack:
            self.apply_acceleration(self.rl_stack[:1], rl_actions[:1])

            direction = np.round(rl_actions[1:])
            for i, x in enumerate(direction):
                if x not in [-1, 0, 1]:
                    direction[i] = 0
            self.apply_lane_change(self.rl_stack[:1], direction=direction)


    def compute_reward(self, state, rl_actions, **kwargs):
        vel_reward = rewards.desired_velocity(self, fail=kwargs["fail"])
        avg_vel_reward = rewards.average_velocity(self, fail=kwargs["fail"])
        penalty = rewards.penalize_standstill(self, gain=1)
        total_vel = rewards.total_velocity(self, fail=kwargs["fail"])

        # Use a similar weighting of of the headway reward as the velocity
        # reward
        # max_cost = np.array([self.env_params.additional_params[
        #                          "target_velocity"]] *
        #                     self.vehicles.num_vehicles)
        # max_cost = np.linalg.norm(max_cost)
        # normalization = self.scenario.length / self.vehicles.num_vehicles
        # headway_reward = 0.2 * max_cost * rewards.penalize_headway_variance(
            # self.vehicles, self.sorted_extra_data, normalization)
        # return vel_reward + headway_reward
        # return vel_reward
        # return total_vel
        return avg_vel_reward + penalty

    def get_state(self, **kwargs):
        """
        Want to include:

        # state = np.array(np.concatenate([rl_pos, rl_vel,
        #                                 merge_dists_0, merge_0_vel,
        #                                 merge_dists_1, merge_1_vel,
        #                                 tailway_dists, tailway_vel,
        #                                 headway_dists, headway_vel,
        #                                 queue_0, queue_1,
        #                                 roundabout_state]))

        * dist, vel of all vehicles in the roundabout.
        * vel, dist of vehicle closest to merge_0 [merge_dists_0,merge_0_vel]
        * vel, dist of vehicle closest to merge_1 [merge_dists_1, merge_1_vel]
        * dist, vel 1 vehicle ahead, 1 vehicle behind [tailway_dists, tailway_vel,
                                                      [headway_dists, headway_vel]
        * dist, vel of first RL vehicle [rl_pos, rl_vel]
        * length of queues [queue_0, queue_1]

        The following variables are dependent on the (existence of)
        the RL vehicle and should be passed 0s if it does not exist:
            - rl_pos, rl_vel
            - tailway_dists, tailway_vel
            - headway_dists, headway_vel
        """
        # try:
        rl_id = None
        # Get normalization factors 
        circ = self.circumference()
        max_speed = self.scenario.max_speed 
        merge_0_norm = self.scenario.edge_length('inflow_0') + \
                        self.scenario.edge_length(':e_0') + \
                        self.scenario.edge_length('merge_in_0') + \
                        self.scenario.edge_length(':c_0')
        merge_1_norm = self.scenario.edge_length('inflow_1') + \
                        self.scenario.edge_length(':g_2') + \
                        self.scenario.edge_length('merge_in_1') + \
                        self.scenario.edge_length(':a_0')
        queue_0_norm = ceil(merge_0_norm/5 + 1) # 5 is the car length
        queue_1_norm = ceil(merge_1_norm/5 + 1)

        # Get the RL-dependent info
        # TODO potential error here if normalizing with self.scenario.length
        # because I'm not sure if this includes internal edges or not
        if self.rl_stack:
            # Get the rl_id
            rl_id = self.rl_stack[0]

            # rl_pos, rl_vel
            rl_pos = [self.get_x_by_id(rl_id) / self.scenario_length]
            rl_vel = [self.vehicles.get_speed(rl_id) / max_speed]
            if self.vehicles.get_edge(rl_id) in self.roundabout_edges:
                rl_pos_2 = [self.get_x_by_id(rl_id) / self.roundabout_length]
            else: 
                rl_pos_2 = [0]
            # print(rl_id, rl_pos)

            # tailway_dists, tailway_vel
            # headway_dists, headway_vel
            tail_id = self.vehicles.get_follower(rl_id)
            head_id = self.vehicles.get_leader(rl_id)
            # print(rl_id, head_id)
            # TODO BUG HERE

            # This is kinda shitty coding, but I'm not that confident
            # in get_lane_tailways atm, Idrk how it works 
            if tail_id: 
                tailway_vel = [self.vehicles.get_speed(tail_id) / max_speed]
                tailway_dists = self.vehicles.get_lane_tailways(rl_id)
                if not tailway_vel:
                    tailway_vel = [0]
                if not tailway_dists or tailway_dists[0] == 1e+3:
                    tailway_dists = [0]
                else:
                    tailway_dists = tailway_dists[:1]
                    tailway_dists[0] = tailway_dists[0] / self.scenario_length
            else: # No 
                tailway_vel = [0]
                tailway_dists = [0]
            if head_id:
                headway_vel = [self.vehicles.get_speed(head_id) / max_speed]
                headway_dists = self.vehicles.get_lane_headways(rl_id)
                # print(headway_dists)
                if not headway_vel:
                    headway_vel = [0]
                if not headway_dists or headway_dists[0] == 1e+3:
                    headway_dists = [0]
                else:
                    headway_dists = headway_dists[:1]
                    headway_dists[0] = headway_dists[0] / self.scenario_length
                # print(headway_dists)
            else: # No leader
                headway_vel = [0]
                headway_dists = [0]
            # print(rl_id, headway_dists[0] * self.scenario_length, tailway_dists[0]* self.scenario_length)


        else: # RL vehicle's not in the system. Pass in zeros here 
            rl_pos = [0]
            rl_pos_2 = [0]
            rl_vel = [0]
            tailway_vel = [0]
            tailway_dists = [0]
            headway_vel = [0]
            headway_dists = [0]

        # DISTANCES
        # sorted by closest to farthest
        merge_id_0, merge_id_1 = self.k_closest_to_merge(self.n_merging_in) # TODO check this is sorted
        merge_dists_0 = self.process(self._dist_to_merge_0(merge_id_0),
                                    length=self.n_merging_in,
                                    normalizer=merge_0_norm)
        merge_dists_1 = self.process(self._dist_to_merge_1(merge_id_1),
                                    length=self.n_merging_in,
                                    normalizer=merge_1_norm)

        # OBSOLETE NOW 
        # # Get (ID, dist_from_RL) for the k vehicles closest to 
        # # the RL vehicle. 0 if there is no k_closest.
        # tailway, headway = self.k_closest_to_rl('rl_0', self.n_preceding) #todo
        # tailway_ids = [x[0] for x in tailway]
        # tailway_dists = [x[1] for x in tailway]
        # tailway_dists = self.process(tailway_dists,
        #                             length=self.n_preceding,
        #                             normalizer=circ)

        # headway_ids = [x[0] for x in headway]
        # headway_dists = [x[1] for x in headway]
        # headway_dists = self.process(headway_dists,
        #                             length=self.n_preceding,
        #                             normalizer=circ)
        # tailway_dists = [self.vehicles.get_follower('rl_0')]


        # VELOCITIES
        merge_0_vel = self.process(self.vehicles.get_speed(merge_id_0),
                                length=self.n_merging_in,
                                normalizer=max_speed)
        merge_1_vel = self.process(self.vehicles.get_speed(merge_id_1),
                                length=self.n_merging_in,
                                normalizer=max_speed)
        # tailway_vel = self.process(self.vehicles.get_speed(tailway_ids),
        #                         length=self.n_preceding,
        #                         normalizer=max_speed)
        # headway_vel = self.process(self.vehicles.get_speed(headway_ids),
        #                         length=self.n_following,
        #                         normalizer=max_speed)

        queue_0, queue_1 = self.queue_length()
        queue_0 = [queue_0 / queue_0_norm]
        queue_1 = [queue_1 / queue_1_norm]
        
        roundabout_full = self.roundabout_full()
        # import ipdb; ipdb.set_trace()
        roundabout_full[:,0] = roundabout_full[:,0]/self.roundabout_length
        roundabout_full[:,1] = roundabout_full[:,1]/max_speed
        roundabout_full = roundabout_full.flatten().tolist()
        # roundabout_state = self.roundabout_state()
        state = np.array(np.concatenate([rl_pos, rl_pos_2, rl_vel,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        tailway_dists, tailway_vel,
                                        headway_dists, headway_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
        # if len(state) != 93:
        #     import ipdb; ipdb.set_trace()
    
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

        one lane: 
        [":a_1", "right", ":b_1", "top", ":c_1",
        "left", ":d_1", "bottom", "inflow_1",
        ":g_2", "merge_in_1", ":a_0", ":b_0",
        "merge_out_0", ":e_1", "outflow_0", "inflow_0",
        ":e_0", "merge_in_0", ":c_0", ":d_0",
        "merge_out_1", ":g_0", "outflow_1" ]
        """
        edges_0 = ["merge_in_0", "inflow_0", ":e_0", ":c_0"]
        edges_1 = ["merge_in_1", "inflow_1", ":g_2", ":a_0"]

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

        one lane: 
        [":a_1", "right", ":b_1", "top", ":c_1",
        "left", ":d_1", "bottom", "inflow_1",
        ":g_2", "merge_in_1", ":a_0", ":b_0",
        "merge_out_0", ":e_1", "outflow_0", "inflow_0",
        ":e_0", "merge_in_0", ":c_0", ":d_0",
        "merge_out_1", ":g_0", "outflow_1" ]

        """

        # Arrays holding veh_ids of vehicles before/after the rl_id.
        # Will always be sorted by distance to rl_id 
        k_tailway = [] 
        k_headway = []

        # Prep.
        route = ["top", "left", "bottom", "right"]
        if self.scenario.lane_num == 2:
            route = ["top", ":c_2", "left", ":d_2", "bottom", ":a_2", "right", ":b_2"] #two lane
        elif self.scenario.lane_num == 1:
            route = ["top", ":c_1", "left", ":d_1", "bottom", ":a_1", "right", ":b_1"]
        rl_edge = self.vehicles.get_edge(rl_id)
        if rl_edge == "":
            return [], []
        rl_index = route.index(rl_edge) 
        rl_x = self.get_x_by_id(rl_id)
        rl_pos = self.vehicles.get_position(rl_id)
        print(rl_pos)

        # Get preceding first.
        for i in range(rl_index, rl_index-3, -1): # Curr  edge and preceding edge
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
        for i in range(rl_index, rl_index+3):
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

    def roundabout_full(self):
        """
        A zero-padded state space with ROUNDABOUT_LENGTH / 5 (veh_len)
        spaces for observations. I suspect at least half of this array
        will be empty most of the time and it will never reach full
        capacity. This is so we can achieve a full state space. 

        state[0] = abs pos
        state[1] = vel
        """
        state = np.zeros((int(self.roundabout_length//5), 2))
        i = 0 # index of state to alter
        for edge in self.roundabout_edges:
            vehicles = sorted(self.vehicles.get_ids_by_edge(edge),
                              key=lambda x: self.get_x_by_id(x))
            for veh_id in vehicles:
                state[i][0] = self.get_x_by_id(veh_id)
                state[i][1] = self.vehicles.get_speed(veh_id)
                i += 1
        return state



    def roundabout_state(self): # this is variable length, is that okay? I could instead m
        """
        Need some way to pass a static state about this

        Dynamic (bc inflows): position, vel
        Static: edge density, cars one edge, avg velocity on edge 
            - obviously there   are a lot of problems with this but it's
              possible that since the ring is so small, this could work

        STATIC VERSION
        """
        # total num is 3 * 24
        # merge_edges = [":a_1", "right", ":b_1", "top", ":c_1",
        #                "left", ":d_1", "bottom", "inflow_1",
        #                ":g_2", "merge_in_1", ":a_0", ":b_0",
        #                "merge_out_0", ":e_1", "outflow_0", "inflow_0",
        #                ":e_0", "merge_in_0", ":c_0", ":d_0",
        #                "merge_out_1", ":g_0", "outflow_1" ] # len 24
        # import ipdb; ipdb.set_trace()
        states = []
        for edge in self.roundabout_edges:
            density = self._edge_density(edge) # No need to normalize, already under 0
            states.append(density)
            avg_velocity = self._edge_velocity(edge) 
            avg_velocity = avg_velocity / self.scenario.max_speed
            states.append(avg_velocity)
            num_veh = len(self.vehicles.get_ids_by_edge(edge)) / 10 # Works for now
            states.append(num_veh)
        # import ipdb; ipdb.set_trace()
        return states

        
    def _edge_density(self, edge):
        num_veh = len(self.vehicles.get_ids_by_edge(edge))
        length = self.scenario.edge_length(edge)
        return num_veh/length 

    def _edge_velocity(self, edge):
        vel = self.vehicles.get_speed(self.vehicles.get_ids_by_edge(edge))
        return np.mean(vel) if vel else 0

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

    def queue_length(self):
        queue_0 = len(self.vehicles.get_ids_by_edge(["inflow_0", ":e_0", "merge_in_0", ":c_0"]))
        queue_1 = len(self.vehicles.get_ids_by_edge(["inflow_1", ":g_2", "merge_in_1", ":a_0"]))
        return queue_0, queue_1

    def process(self, state, length=None, normalizer=1):
        """
        Takes in a list, returns a normalized version of the list
        with padded zeros at the end according to the length parameter
        """
        if length: # Truncation or padding
            if len(state) < length:
                state += [0.] * (length - len(state))
            else:
                state = state[:length]
        state = [x / normalizer for x in state]
        return state

    def circumference(self):
        """
        Counts the circumference on the circle, because
        annoyingly it's not 2*pi*r
        """
        # circ = 0
        # edges = [":a_1", "right", ":b_1", "top", ":c_1",
        #          "left", ":d_1", "bottom"]
        circ = sum([self.scenario.edge_length(e) for e in self.roundabout_edges])
        return circ
        
    def get_k_followers(self, veh_id, k):
        """
        Return the IDs of the k vehicles behind veh_id.
        Will not pad zeros.
        """
        curr = veh_id 
        tailways = []
        while k > 0 and curr.get_follower():
            tailways.append(curr.get_follower())
            curr = curr.get_follower()
            k -= 0
        return tailways

    def get_k_leaders(self, veh_id, k):
        """
        Return the IDs of the k vehicles leading veh_id.
        Will not pad zeros.
        """
        curr = veh_id 
        leaders = []
        while k > 0 and curr.get_leader():
            tailways.append(curr.get_leader())
            curr = curr.get_leader()
            k -= 0
        return leaders

    @property
    def roundabout_length(self):
        rl = sum([self.scenario.edge_length(e) for e in self.roundabout_edges])
        return rl

    @property
    def scenario_length(self):
        length = sum([self.scenario.edge_length(e) for e in self.all_edges])
        return length

    def additional_command(self):
        try: 
            self.velocities.append(np.mean(self.vehicles.get_speed(self.vehicles.get_ids())))
        except AttributeError:
            self.velocities = []
            self.velocities.append(np.mean(self.vehicles.get_speed(self.vehicles.get_ids())))
        
        # Curate rl_stack
        for veh_id in self.vehicles.get_rl_ids():
            if veh_id not in self.rl_stack:
                self.rl_stack.append(veh_id) # TODO also need step for removing it from the system



# # one lane: 
#         [":a_1", "right", ":b_1", "top", ":c_1",
#         "left", ":d_1", "bottom", "inflow_1",
#         ":g_2", "merge_in_1", ":a_0", ":b_0",
#         "merge_out_0", ":e_1", "outflow_0", "inflow_0",
#         ":e_0", "merge_in_0", ":c_0", ":d_0",
#         "merge_out_1", ":g_0", "outflow_1" ]

class CartesianEnv(RoundaboutEnv):
    def __init__(self, env_params, sumo_params, scenario):
        self.positions = {}
        super().__init__(env_params, sumo_params, scenario)

    @property
    def observation_space(self):
        # Vehicle position and velocity, normalized: 3
        # Queue length x 2
        # Roundabout state = len(MERGE_EDGES) * 3
        # Roundabout full = (ROUNDABOUT_LENGTH // 5)*3 # 2 cols
        # rl_pos_2, the pos in the roundabout: 1
        # + 1 to account for the new cartesian coord for the rl veh
        self.total_obs = self.n_obs_vehicles * 2 + 2 + \
                         int(self.roundabout_length // 5) * 3 + 1 + 1
                         
        box = Box(low=0.,
                  high=1,
                  shape=(self.total_obs,),
                  dtype=np.float32)          
        return box
    
    def get_state(self, **kwargs):
        """
        *************************
        The state space includes:
        *************************

        * distance, velocity of the RL-controlled vehicle
        * distance, velocity of all vehicles in the roundabout
        * distance, velocity of all vehicles closest to merge_0
        * distance, velocity of vehicles closest to merge_1
        * distance, velocity of vehicles leading and following
          the RL vehicle 
        * number of vehicles in each queue 

        The state space is returned in the form:

            [rl_pos, rl_pos_2, rl_vel,
            merge_dists_0, merge_0_vel,
            merge_dists_1, merge_1_vel,
            tailway_dists, tailway_vel,
            headway_dists, headway_vel,
            queue_0, queue_1,
            roundabout_full]

        ***********************************************
        Description of the variables in the state space
        ***********************************************

        * rl_pos: absolute position of the RL vehicle / total_scenario_length.
            0 if there is no RL vehicle in the system.
        * rl_pos_2: absolute position of the RL vehicle / roundabout_length.
            0 if there is no RL vehicle is outside the roundabout or if there
            are no RL vehicles in the system. 
        * rl_vel: velocity of the RL vehicle / max_speed
            0 if there is no RL vehicle in the system.
        * merge_dists_0: distance to the roundabout / merge_0_norm for the k
            vehicles closest to the roundabout on the Northern entry. Sorted by
            increasing distance to the roundabout
        * merge_0_vel: velocity / max_speed for the k vehicles closest to the
            roundabout on the Northern entry. Sorted by increasing distance to the
            roundabout
        * merge_dists_1: distance to the roundabout / merge_1_norm for the k
            vehicles closest to the roundabout on the Western entry. Sorted by
            increasing distance to the roundabout
        * merge_1_vel: velocity / max_speed for the k vehicles closest to the
            roundabout on the Western entry. Sorted by increasing distance to the
            roundabout
        * tailway_dists: absolute position / total_scenario_length of the RL vehicle's
            follower. 0 if there is no follower.
        * tailway_vel: absolute position / max_speed of the RL vehicle's
            follower. 0 if there is no follower.
        * headway_dists: absolute position / total_scenario_length of the RL vehicle's
            leader. 0 if there is no leader.
        * headway_vel: absolute position / max_speed of the RL vehicle's
            leader. 0 if there is no leader.
        * queue_0: number of vehicles on the Northern entry / queue_0_norm
        * queue_1: number of vehicles on the Western entry / queue_1_norm
        * roundabout_full: 26x2 2D array. The 0th column of this array consists of
            absolute positions / roundabout_length for veh_ids, and the 1st column consists of
            velocity / max_speed for veh_ids, where veh_ids denotes the vehicles in the
            roundabout. Sorted by increasing absolute distance. Pad in zeros to the end of this
            array to fill up all 26x2 entries. 

        *****************************************
        Constant values and normalization factors
        *****************************************
        * k = 6 [describes num vehicles observed for merge_* variables]
        * merge_0_norm = 49.32
        * merge_1_norm = 61.54
        * queue_0_norm = 11
        * queue_1_norm = 14
        * total_scenario_length = 342.90999999999997
        * roundabout_length = 130.98
        * max_speed = 15 

        """
        print('hollaaaaaa')
        rl_id = None
        
        # Get normalization factors 
        circ = self.circumference()
        max_speed = self.scenario.max_speed 
        merge_0_norm = self.scenario.edge_length('inflow_0') + \
                        self.scenario.edge_length(':e_0') + \
                        self.scenario.edge_length('merge_in_0') + \
                        self.scenario.edge_length(':c_0')
        merge_1_norm = self.scenario.edge_length('inflow_1') + \
                        self.scenario.edge_length(':g_2') + \
                        self.scenario.edge_length('merge_in_1') + \
                        self.scenario.edge_length(':a_0')
        queue_0_norm = ceil(merge_0_norm/5 + 1) # 5 is the car length
        queue_1_norm = ceil(merge_1_norm/5 + 1)

        # Get the RL-dependent info
        if self.rl_stack:
            # Get the rl_id
            rl_id = self.rl_stack[0]

            # rl_pos, rl_vel
            rl_pos = self.vehicles.get_2d_position(rl_id)
            if not rl_pos:
                rl_pos = [0, 0]
            rl_pos[0] = rl_pos[0] / self.scenario.generator.max_x
            rl_pos[1] = rl_pos[1] / self.scenario.generator.max_y

            rl_vel = [self.vehicles.get_speed(rl_id) / max_speed]
            if self.vehicles.get_edge(rl_id) in ROUNDABOUT_EDGES:
                rl_pos_2 = [self.get_x_by_id(rl_id) / self.roundabout_length]
            else: 
                rl_pos_2 = [0]

            # tailway_dists, tailway_vel
            # headway_dists, headway_vel
            tail_id = self.vehicles.get_follower(rl_id)
            head_id = self.vehicles.get_leader(rl_id)

            # This is kinda shitty coding, but I'm not that confident
            # in get_lane_tailways atm, Idrk how it works 
            if tail_id: 
                tailway_vel = [self.vehicles.get_speed(tail_id) / max_speed]
                tailway_dists = self.vehicles.get_lane_tailways(rl_id)
                if not tailway_vel:
                    tailway_vel = [0]
                if not tailway_dists or tailway_dists[0] == 1e+3:
                    tailway_dists = [0]
                else:
                    tailway_dists[0] = tailway_dists[0] / self.scenario_length
            else: # No 
                tailway_vel = [0]
                tailway_dists = [0]
            if head_id:
                headway_vel = [self.vehicles.get_speed(head_id) / max_speed]
                headway_dists = self.vehicles.get_lane_headways(rl_id)
                # print(headway_dists)
                if not headway_vel:
                    headway_vel = [0]
                if not headway_dists or headway_dists[0] == 1e+3:
                    headway_dists = [0]
                else:
                    headway_dists[0] = headway_dists[0] / self.scenario_length
                # print(headway_dists)
            else: # No leader
                headway_vel = [0]
                headway_dists = [0]

        else: # RL vehicle's not in the system. Pass in zeros here 
            rl_pos = [0, 0]
            rl_pos_2 = [0]
            rl_vel = [0]
            tailway_vel = [0]
            tailway_dists = [0]
            headway_vel = [0]
            headway_dists = [0]

        # DISTANCES
        # sorted by closest to farthest
        merge_id_0, merge_id_1 = self.k_closest_to_merge(self.n_merging_in)
        merge_dists_0 = self.process(self._dist_to_merge_0(merge_id_0),
                                    length=self.n_merging_in,
                                    normalizer=merge_0_norm)
        merge_dists_1 = self.process(self._dist_to_merge_1(merge_id_1),
                                    length=self.n_merging_in,
                                    normalizer=merge_1_norm)

        # OBSOLETE NOW 
        # # Get (ID, dist_from_RL) for the k vehicles closest to 
        # # the RL vehicle. 0 if there is no k_closest.
        # tailway, headway = self.k_closest_to_rl('rl_0', self.n_preceding) #todo
        # tailway_ids = [x[0] for x in tailway]
        # tailway_dists = [x[1] for x in tailway]
        # tailway_dists = self.process(tailway_dists,
        #                             length=self.n_preceding,
        #                             normalizer=circ)

        # headway_ids = [x[0] for x in headway]
        # headway_dists = [x[1] for x in headway]
        # headway_dists = self.process(headway_dists,
        #                             length=self.n_preceding,
        #                             normalizer=circ)
        # tailway_dists = [self.vehicles.get_follower('rl_0')]


        # VELOCITIES
        merge_0_vel = self.process(self.vehicles.get_speed(merge_id_0),
                                length=self.n_merging_in,
                                normalizer=max_speed)
        merge_1_vel = self.process(self.vehicles.get_speed(merge_id_1),
                                length=self.n_merging_in,
                                normalizer=max_speed)
        # tailway_vel = self.process(self.vehicles.get_speed(tailway_ids),
        #                         length=self.n_preceding,
        #                         normalizer=max_speed)
        # headway_vel = self.process(self.vehicles.get_speed(headway_ids),
        #                         length=self.n_following,
        #                         normalizer=max_speed)

        queue_0, queue_1 = self.queue_length()
        queue_0 = [queue_0 / queue_0_norm]
        queue_1 = [queue_1 / queue_1_norm]
        
        roundabout_full = self.roundabout_full()
        
        # Normalize the 0th and 1st  column by max_x and max_y respectively
        roundabout_full[:,0] = roundabout_full[:,0]/self.scenario.generator.max_x
        roundabout_full[:,1] = roundabout_full[:,1]/self.scenario.generator.max_y

        # Normalize the 1st column containing velocities
        roundabout_full[:,2] = roundabout_full[:,2]/max_speed
        roundabout_full = roundabout_full.flatten().tolist()

        state = np.array(np.concatenate([rl_pos, rl_pos_2, rl_vel,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        tailway_dists, tailway_vel,
                                        headway_dists, headway_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
        return state

    
    def roundabout_full(self):
        """
        A zero-padded state space with ROUNDABOUT_LENGTH / 5 (veh_len)
        spaces for observations. I suspect at least half of this array
        will be empty most of the time and it will never reach full
        capacity. This is so we can achieve a full state space. 

        state[0] = abs pos
        state[1] = vel
        """
        state = np.zeros((int(self.roundabout_length//5), 3))
        i = 0 # index of state to alter
        for edge in self.roundabout_edges:
            vehicles = sorted(self.vehicles.get_ids_by_edge(edge),
                              key=lambda x: self.get_x_by_id(x))
            for veh_id in vehicles:
                if not self.vehicles.get_2d_position(veh_id):
                    state[i][:2] = [0, 0]
                else:
                    state[i][:2] = self.vehicles.get_2d_position(veh_id)
                state[i][2] = self.vehicles.get_speed(veh_id)
                i += 1
        return state

    def additional_command_2(self):
        # try:

        # Curate rl_stack
        for veh_id in self.vehicles.get_rl_ids():
            if veh_id not in self.rl_stack:
                self.rl_stack.append(veh_id)

        # Curate rl_stack
        removal = [] 
        for rl_id in self.rl_stack:
            if rl_id not in self.vehicles.get_rl_ids():
                removal.append(rl_id)
        for rl_id in removal:
            self.rl_stack.remove(rl_id)