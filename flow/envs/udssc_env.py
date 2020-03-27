from flow.envs.base_env import Env
from flow.multiagent_envs import MultiEnv
from flow.core import rewards
from flow.core.params import InitialConfig, NetParams, InFlows
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoCarFollowingParams, SumoLaneChangeParams, VehicleParams  

# from gym.spaces.box import Box
# from gym.spaces.tuple_space import Tuple
from gym.spaces import Box, Tuple, Dict

from copy import deepcopy
from math import ceil
from collections import deque

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
            ['inflow_1', ':g_2', 'merge_in_1', ':a_0'
             , ':b_0', 'merge_out_0', ':e_1', 'outflow_0'
             , 'inflow_0', ':e_0', 'merge_in_0', ':c_0'
             , ':d_0', 'merge_out_1', ':g_0', 'outflow_1']

RAMP_0 = ["inflow_0", ":e_0", "merge_in_0", ":c_0"]
RAMP_1 = ["inflow_1", ":g_2", "merge_in_1", ":a_0"]

SHARED_ROUTE = ["left", ":d_0", "merge_out_1", ":g_0", "outflow_1"]
ROUTE_0 = ["inflow_0", ":e_0", "merge_in_0", ":c_0"] + SHARED_ROUTE
ROUTE_1 = ["inflow_1", ":g_2", "merge_in_1", ":a_0", "right", ":b_1", "top", ":c_1"] + SHARED_ROUTE
ROUTE_2 = ["inflow_1", ":g_2", "merge_in_1", ":a_0", "right", ":b_0", "merge_out_0", ":e_1", "outflow_0"] 

class UDSSCMergeEnv(Env):
    """
    Environment for training cooperative merging behavior in
    the SE corner of UDSSC.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * n_preceding: number of observable vehicles preceding the rl vehicle
    * n_following: number of observable vehicles following the rl vehicle
    * n_merging_in: number of observable merging-in vehicle from the larger
      loop

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
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
        self.rl_stack_2 = []
        
        # Maintain an array of actions to aid penalizing jerkiness
        self.past_actions = deque(maxlen=10)
        self.past_actions_2 = deque(maxlen=10)
        self.actions = []

        super().__init__(env_params, sim_params, scenario)


    @property
    def observation_space(self):
        # state = np.array(np.concatenate([rl_info, rl_info_2,
        #                                 merge_dists_0, merge_0_vel,
        #                                 merge_dists_1, merge_1_vel,
        #                                 queue_0, queue_1,
        #                                 roundabout_full]))

        # rl_info = rl_pos, rl_pos_2, rl_vel, tailway_vel, tailway_dists, headway_vel, headway_dists = 7
        # rl_info_2 = rl_pos, rl_pos_2, rl_vel, tailway_vel, tailway_dists, headway_vel, headway_dists = 7
        # merge_info = self.n_merging_in * 4 # 4 variables
        # queues = 2 # 2 queues
        # Roundabout state = len(MERGE_EDGES) * 3
        # roundabout_full = (ROUNDABOUT_LENGTH // 5) * 2 # 2 cols
        
        self.total_obs = 7 * 2 + \
                         self.n_merging_in * 4 + \
                         2 + \
                         int(self.roundabout_length // 5) * 2
                         
        box = Box(low=0.,
                  high=1,
                  shape=(self.total_obs,),
                  dtype=np.float32)          
        return box

    @property
    def action_space(self):
        """
        Actions
        Actions are a list of acceleration for the RL vehicle currently being
        controlled, bounded by the maximum accelerations and decelerations
        specified in EnvParams. 
        """
        return Box(low=-np.abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"],
                   shape=(2,),
                   dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """
        This includes the second step of RL stack curation, specifically
        focusing on the case where an RL vehicle was removed in the last
        time step and hasn't been recorded as removed yet.

        Remove RL vehicles that are no longer in the system.

        Apply rl_actions to the first vehicle in the stack.

        Notes: More efficient to keep a removal list than to resize
        continuously
        """
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
        self.curate_rl_stack() # This shouldn't be necessary given the new flow of events
        
        # self.actions.append(rl_actions[1])
        # Apply RL Actions
        if self.rl_stack:
            rl_id = self.rl_stack[0]
            if self.in_control(rl_id):
                self.k.vehicle.apply_acceleration([rl_id], rl_actions[:1])
                self.past_actions.append(rl_actions[0])

        if self.rl_stack_2:
            rl_id_2 = self.rl_stack_2[0]
            if self.in_control(rl_id_2):
                self.k.vehicle.apply_acceleration([rl_id_2], rl_actions[1:])
                self.past_actions_2.append(rl_actions[1])

    def compute_reward(self, rl_actions, **kwargs):
        """
        Current reward used is:
        - average velocity
        - penalizing standstill
        """
        if self.env_params.evaluate:
            return rewards.min_delay(self)
        penalty = rewards.penalize_standstill(self, gain=1)
        penalty_2 = rewards.penalize_near_standstill(self, thresh=0.2, gain=1)
        penalty_jerk = rewards.penalize_jerkiness(self, gain=0.1)
        penalty_speeding = rewards.penalize_speeding(self, gain=3, fail=kwargs['fail'])
        min_delay = rewards.min_delay(self)

        # Use a similar weighting of of the headway reward as the velocity
        # reward

        # print('avg_vel: %.2f, min_delay: %.2f, penalty: %.2f, penalty_2: %.2f, penalty_jerk: %.2f, penalty_speed: %.2f' % \
        #       (avg_vel, min_delay, penalty, penalty_2, penalty_jerk, penalty_speeding))
        return 2 * min_delay + penalty + penalty_2 + penalty_jerk + penalty_speeding
        # return 2 * min_delay + penalty_speeding
        # return min_delay + penalty_speeding 

    def get_state(self, **kwargs):
        """
        *************************
        The state space includes:
        *************************

        * absolute distance, velocity of the RL-controlled vehicle
        * absolute distance, velocity of all vehicles in the roundabout
        * relative distance, velocity of all vehicles closest to merge_0
        * relative distance, velocity of vehicles closest to merge_1
        * tailway and headway distance, velocity of vehicles leading and following
          the RL vehicle 
        * number of vehicles in each queue 

        The state space is returned in the form:
            [rl_info, rl_info_2,
            merge_dists_0, merge_0_vel,
            merge_dists_1, merge_1_vel,
            queue_0, queue_1,
            roundabout_full]

        ***********************************************
        Description of the variables in the state space
        ***********************************************
        * rl_info: the concatenation of [rl_pos, rl_pos_2, rl_vel, tailway_vel,
                   tailway_dists, headway_vel, headway_dists] for the Northern-origin
                   RL vehicle, variables described below
        * rl_info_2: the concatenation of [rl_pos, rl_pos_2, rl_vel, tailway_vel,
                   tailway_dists, headway_vel, headway_dists] for the Western-origin
                   RL vehicle, variables described below
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
        # speeds = []
        # for veh_id in self.k.vehicle.get_ids():
        #     speeds.append(self.k.vehicle.get_speed(veh_id))
        # print(speeds)

        self.curate_rl_stack()

        rl_id = None
        
        # Get normalization factors 
        circ = self.circumference()
        max_speed = self.k.scenario.max_speed()
        merge_0_norm = sum([self.k.scenario.edge_length(e) for e in RAMP_0])
        merge_1_norm = sum([self.k.scenario.edge_length(e) for e in RAMP_1])
        queue_0_norm = ceil(merge_0_norm/5 + 1) # 5 is the car length
        queue_1_norm = ceil(merge_1_norm/5 + 1)

        rl_info = self.rl_info(self.rl_stack)
        rl_info_2 = self.rl_info(self.rl_stack_2)

        # DISTANCES
        # sorted by closest to farthest
        merge_id_0, merge_id_1 = self.k_closest_to_merge(self.n_merging_in)
        merge_dists_0 = self.process(self._dist_to_merge_0(merge_id_0),
                                    length=self.n_merging_in,
                                    normalizer=merge_0_norm)
        merge_dists_1 = self.process(self._dist_to_merge_1(merge_id_1),
                                    length=self.n_merging_in,
                                    normalizer=merge_1_norm)


        # VELOCITIES
        merge_0_vel = self.process(self.k.vehicle.get_speed(merge_id_0),
                                length=self.n_merging_in,
                                normalizer=max_speed)
        merge_1_vel = self.process(self.k.vehicle.get_speed(merge_id_1),
                                length=self.n_merging_in,
                                normalizer=max_speed)

        queue_0, queue_1 = self.queue_length()
        queue_0 = [queue_0 / queue_0_norm]
        queue_1 = [queue_1 / queue_1_norm]
        
        roundabout_full = self.roundabout_full()
        
        # Normalize the 0th column containing absolute position
        roundabout_full[:,0] = roundabout_full[:,0]/self.roundabout_length

        # Normalize the 1st column containing velocities
        roundabout_full[:,1] = roundabout_full[:,1]/max_speed
        roundabout_full = roundabout_full.flatten().tolist()
        
        state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))

        if "state_noise" in self.env_params.additional_params:
            var = self.env_params.additional_params.get("state_noise")
            for i, st in enumerate(state):
                perturbation = np.random.normal(0, var) 
                state[i] = st + perturbation

        # Reclip
        if isinstance(self.observation_space, Box):
            state = np.clip(
                state,
                a_min=self.observation_space.low,
                a_max=self.observation_space.high)

        return state

    def get_state_test(self): 
        # number of vehicles on each edge
        self.curate_rl_stack()

        def edge_index(veh_id):
            try:
                edge_index = (ALL_EDGES.index(self.k.vehicle.get_edge(veh_id))+1)/len(ALL_EDGES)
                return edge_index
            except ValueError:
                return 0
      
        rl_info = [0] * 2
        rl_info_2 = [0] * 2

        if self.rl_stack:
            rl_id = self.rl_stack[0]
            rl_info[0] = self.k.vehicle.get_x_by_id(rl_id) / self.scenario_length
            rl_info[1] = edge_index(rl_id)


        if self.rl_stack_2:
            rl_id_2 = self.rl_stack_2[0]
            rl_info_2[0] = self.k.vehicle.get_x_by_id(rl_id_2) / self.scenario_length
            rl_info_2[1] = edge_index(rl_id_2)   

        edge_info = [len(self.k.vehicle.get_ids_by_edge(edge)) for edge in ALL_EDGES]
        state = np.array(np.concatenate([edge_info, rl_info, rl_info_2]))
        if isinstance(self.observation_space, Box):
            state = np.clip(
                state,
                a_min=self.observation_space.low,
                a_max=self.observation_space.high)
        return state
    
    def rl_info(self, stack):
        max_speed = self.k.scenario.max_speed() 
        # state = [] 
        if stack:
            # Get the rl_id
            # num_rl = min(1, len(stack))
            rl_id = stack[0]
            
            # rl_pos, rl_vel
            rl_pos = [self.k.vehicle.get_x_by_id(rl_id) / self.scenario_length]
            rl_vel = [self.k.vehicle.get_speed(rl_id) / max_speed] if \
                      self.k.vehicle.get_speed(rl_id)!= -1001 else [0]
            if self.k.vehicle.get_edge(rl_id) in ROUNDABOUT_EDGES:
                rl_pos_2 = [self.k.vehicle.get_x_by_id(rl_id) / self.roundabout_length]
            else: 
                rl_pos_2 = [0]

            # tailway_dists, tailway_vel
            # headway_dists, headway_vel
            tail_ids = self.k.vehicle.get_follower(rl_id)
            head_ids = self.k.vehicle.get_leader(rl_id)
            tail_ids = self.k.vehicle.get_lane_followers(rl_id) if not tail_ids else [tail_ids]
            head_ids = self.k.vehicle.get_lane_leaders(rl_id) if not head_ids else [head_ids]
            
            tailway_vel = []
            tailway_dists = []
            headway_vel = []
            headway_dists = []
            
            tailway_vel = [x / max_speed if x != -1001 else 0 for x in self.k.vehicle.get_speed(tail_ids)]
            tailway_dists = self.process([x / self.scenario_length for x in self.k.vehicle.get_lane_tailways(rl_id) \
                                          if x != 1000], length=1)
            tailway_vel = self.process(tailway_vel, length=1)
            tailway_dists = self.process(tailway_dists, length=1)

            headway_vel = [x / max_speed if x != -1001 else 0 for x in self.k.vehicle.get_speed(head_ids)]
            headway_dists = self.process([x / self.scenario_length for x in self.k.vehicle.get_lane_headways(rl_id) \
                                          if x != 1000], length=1)
            headway_vel = self.process(headway_vel, length=1)
            headway_dists = self.process(headway_dists, length=1)

            rl_info = np.concatenate([rl_pos, rl_pos_2, rl_vel, tailway_vel,
                        tailway_dists, headway_vel, headway_dists])

            # Pad
        else:
            rl_info = rl_info = [0] * 7

        
        return rl_info

    def get_tailway(self, v1, v2):
        # Iterative approach

        if self.k.vehicle.get_edge(v1) == self.k.vehicle.get_edge(v2): #they're on the same edge
            return self.k.vehicle.get_position(v1) - self.k.vehicle.get_position(v2) - self.k.vehicle.get_length(v1)# might have to add vehicle length
        
        route = self.k.vehicle.get_route(v1)
        prev = self.k.scenario.prev_edge(self.k.vehicle.get_edge(v1), 0)
        if not prev:
            return 
        prev = [x[0] for x in prev]
        
        v2_edge = self.k.vehicle.get_edge(v2)
        
        for p in prev: # iterate until you find the vehicle, reach the end, or go in a circle
            visited = [self.k.vehicle.get_edge(v1)]

            # queue contains tuples of (EDGE to check, accumulated tailway up to said EDGE)
            queue = [(p, self.k.vehicle.get_position(v1) - self.k.vehicle.get_length(v1))]
            while queue:
                curr, tailway = queue.pop(0)
                if v2_edge == curr:
                    return tailway + self.k.scenario.edge_length(curr) - self.k.vehicle.get_position(v2)
                if curr in visited or (not curr.startswith(':') and curr not in route):
                    continue
                visited.append(curr)
                
                previous = self.k.scenario.prev_edge(curr, 0)
                previous = [x[0] for x in previous]
                for pr in previous:
                    queue.append((pr, tailway + self.k.scenario.edge_length(curr)))

        return None

    # env.vehicles.get_route(self.veh_id)
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

        close_0 = sorted(self.k.vehicle.get_ids_by_edge(edges_0), 
                         key=lambda veh_id:
                         -self.k.vehicle.get_x_by_id(veh_id))

        close_1 = sorted(self.k.vehicle.get_ids_by_edge(edges_1), 
                         key=lambda veh_id:
                         -self.k.vehicle.get_x_by_id(veh_id))

        if len(close_0) > k:
            close_0 = close_0[:k]

        if len(close_1) > k:
            close_1 = close_1[:k]
        
        return close_0, close_1

    def k_closest_to_rl(self, rl_id, k):
        """ 
        NOT USED IN INFLOWS VERSION.

        Return a list of ids and  distances to said rl vehicles

        In the form:
        [(veh_id, dist), (veh_id, dist)]

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
        if self.k.scenario.lane_num == 2:
            route = ["top", ":c_2", "left", ":d_2", "bottom", ":a_2", "right", ":b_2"] #two lane
        elif self.k.scenario.lane_num == 1:
            route = ["top", ":c_1", "left", ":d_1", "bottom", ":a_1", "right", ":b_1"]
        rl_edge = self.k.vehicle.get_edge(rl_id)
        if rl_edge == "":
            return [], []
        rl_index = route.index(rl_edge) 
        rl_x = self.k.vehicle.get_x_by_id(rl_id)
        rl_pos = self.k.vehicle.get_position(rl_id)

        # Get preceding first.
        for i in range(rl_index, rl_index-3, -1): # Curr  edge and preceding edge
            if i == rl_index: # Same edge as rl_id
                veh_ids = [(v, rl_pos - self.k.vehicle.get_position(v)) 
                           for v in self.k.vehicle.get_ids_by_edge(route[i]) 
                           if self.k.vehicle.get_position(v) < rl_pos]
            else: # Preceding edge 
                edge_len = self.k.scenario.edge_length(route[i])
                veh_ids = [(v, rl_pos + (edge_len - self.k.vehicle.get_position(v))) 
                           for v in self.k.vehicle.get_ids_by_edge(route[i])]
            sorted_vehs = sorted(veh_ids, key=lambda v: self.k.vehicle.get_x_by_id(v[0]))
            # k_tailway holds veh_ids in decreasing order of closeness 
            k_tailway = sorted_vehs + k_tailway

        # Get headways second.
        for i in range(rl_index, rl_index+3):
            i = i % len(route)
            # If statement is to cover the case of overflow in get_x 
            if i == rl_index: # Same edge as rl_id
                veh_ids = [(v, self.k.vehicle.get_position(v) - rl_pos) 
                           for v in self.k.vehicle.get_ids_by_edge(route[i]) 
                           if self.k.vehicle.get_position(v) > rl_pos]
            else:
                rl_dist = self.k.scenario.edge_length(rl_edge) - \
                          self.k.vehicle.get_position(rl_id)
                veh_ids = [(v, self.k.vehicle.get_position(v) + rl_dist)
                           for v in self.k.vehicle.get_ids_by_edge(route[i])]
            # The following statement is safe because it's being done
            # by one edge at a time
            sorted_vehs = sorted(veh_ids, key=lambda v: self.k.vehicle.get_x_by_id(v[0]))
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
        for edge in ROUNDABOUT_EDGES:
            vehicles = sorted(self.k.vehicle.get_ids_by_edge(edge),
                              key=lambda x: self.k.vehicle.get_x_by_id(x))
            for veh_id in vehicles:
                state[i][0] = self.k.vehicle.get_x_by_id(veh_id)
                state[i][1] = self.k.vehicle.get_speed(veh_id)
                i += 1
        return state

    def roundabout_state(self): 
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
        states = []
        for edge in ROUNDABOUT_EDGES:
            density = self._edge_density(edge) # No need to normalize, already under 0
            states.append(density)
            avg_velocity = self._edge_velocity(edge) 
            avg_velocity = avg_velocity / self.k.scenario.max_speed()
            states.append(avg_velocity)
            num_veh = len(self.k.vehicle.get_ids_by_edge(edge)) / 10 # Works for now
            states.append(num_veh)
        return states

    def _edge_density(self, edge):
        num_veh = len(self.k.vehicle.get_ids_by_edge(edge))
        length = self.k.scenario.edge_length(edge)
        return num_veh/length 

    def _edge_velocity(self, edge):
        vel = self.k.vehicle.get_speed(self.k.vehicle.get_ids_by_edge(edge))
        return np.mean(vel) if vel else 0

    def _dist_to_merge_1(self, veh_id):
        reference = self.k.scenario.total_edgestarts_dict[":a_0"] + \
                    self.k.scenario.edge_length(":a_0")
        # reference = self.k.scenario.total_edgestarts_dict["merge_in_1"] + \
        #             self.k.scenario.edge_length("merge_in_1")
        distances = [reference - self.k.vehicle.get_x_by_id(v)
                     for v in veh_id]
        return distances

    def _dist_to_merge_0(self, veh_id):
        reference = self.k.scenario.total_edgestarts_dict[":c_0"] + \
                    self.k.scenario.edge_length(":c_0")
        # reference = self.k.scenario.total_edgestarts_dict["merge_in_0"] + \
        #             self.k.scenario.edge_length("merge_in_0")
        distances = [reference - self.k.vehicle.get_x_by_id(v)
                     for v in veh_id]
        return distances
    
    def _dist_to_end(self, veh_id):
        dist = 0
        edge = self.k.vehicle.get_edge(veh_id)
        if edge:
            dist += self.scenario.edge_length(edge) - self.k.vehicle.get_position(veh_id)
            for edge in self._get_edges_left(self.k.vehicle.get_edge(veh_id)):
                dist += self.scenario.edge_length(edge)
        return dist

    def _get_edges_left(self, edge):
        # edges = self.scenario.specify_absolute_order()
        if edge in ROUTE_0:
            edges = ROUTE_0
        elif edge in ROUTE_1:
            edges = ROUTE_1
        elif edge in ROUTE_2:
            edges = ROUTE_2
        index_cur = edges.index(edge)
        return edges[index_cur+1:]

    def queue_length(self):
        queue_0 = len(self.k.vehicle.get_ids_by_edge(["inflow_0", ":e_0", "merge_in_0", ":c_0"]))
        queue_1 = len(self.k.vehicle.get_ids_by_edge(["inflow_1", ":g_2", "merge_in_1", ":a_0"]))
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
        edges = [":a_1", "right", ":b_1", "top", ":c_1",
                 "left", ":d_1", "bottom"]
        circ = sum([self.k.scenario.edge_length(e) for e in edges])
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

    def in_control(self, veh_id):
        """
        Return True if VEH_ID is in the controlled part of the system
        """
        if "control_length" not in self.env_params.additional_params:
            return True 
        control_length = self.env_params.additional_params["control_length"] # this is a percentage
        if self.k.vehicle.get_edge(veh_id) in RAMP_0:
            merge_position = self.k.vehicle.get_x_by_id(veh_id) - \
                             self.scenario.es[RAMP_0[0]]
            merge_len = sum([self.k.scenario.edge_length(e) for e in RAMP_0])
            return False if merge_position <= control_length * merge_len else True
        elif self.k.vehicle.get_edge(veh_id) in RAMP_1:
            merge_position = self.k.vehicle.get_x_by_id(veh_id) - \
                             self.scenario.es[RAMP_1[0]]
            merge_len = sum([self.k.scenario.edge_length(e) for e in RAMP_1])
            return False if merge_position <= control_length * merge_len else True
        else:
            return True


    @property
    def roundabout_length(self):
        rl = sum([self.k.scenario.edge_length(e) for e in ROUNDABOUT_EDGES])
        return rl

    @property
    def scenario_length(self):
        length = sum([self.k.scenario.edge_length(e) for e in ALL_EDGES])
        return length

    @property
    def max_route_length(self):
        return sum([self.k.scenario.edge_length(edge) for edge in ROUTE_1])

    def curate_rl_stack(self):

        # Add to stacks
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in self.rl_stack and self.k.vehicle.get_edge(veh_id) == "inflow_0":
                self.rl_stack.append(veh_id)
            elif veh_id not in self.rl_stack_2 and self.k.vehicle.get_edge(veh_id) == "inflow_1":
                self.rl_stack_2.append(veh_id)

        # Remove from stacks
        removal = [] 
        removal_2 = []
        for rl_id in self.rl_stack:
            if rl_id not in self.k.vehicle.get_rl_ids():
                removal.append(rl_id)
        for rl_id in self.rl_stack_2:
            if rl_id not in self.k.vehicle.get_rl_ids():
                removal_2.append(rl_id)
        for rl_id in removal:
            self.rl_stack.remove(rl_id)
        for rl_id in removal_2:
            self.rl_stack_2.remove(rl_id)

    def additional_command(self):
        if isinstance(self, UDSSCMergeEnvReset):
            self.counter += 1
        try: 
            self.velocities.append(np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids())))
        except AttributeError:
            self.velocities = []
            self.velocities.append(np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids())))

        # Color RL vehicles
        rl_control = self.rl_stack[:min(1, len(self.rl_stack))]
        rl_control_2 = self.rl_stack_2[:min(1, len(self.rl_stack_2))]

        try:
            for veh_id in rl_control + rl_control_2:
                self.traci_connection.vehicle.setColor(
                            vehID=veh_id, color=(0, 255, 255, 255))
        except:
            pass
        

class UDSSCMergeEnvReset(UDSSCMergeEnv):
    """
    Creating this class for the purpose of having an experimental
    reset function.
    """
    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        self.range_inflow_0 = env_params.additional_params['range_inflow_0']
        self.range_inflow_1 = env_params.additional_params['range_inflow_1']
        self.max_inflow = max(self.range_inflow_0 + self.range_inflow_1)
        self.batch_size = env_params.additional_params["batch_size"]
        self.counter = 0
        self.max_actions = max(self.range_inflow_0, self.range_inflow_1)

        super().__init__(env_params, sim_params, scenario)


    @property
    def action_space(self):
        """
        Actions
        Actions are a list of acceleration for the RL vehicle currently being
        controlled, bounded by the maximum accelerations and decelerations
        specified in EnvParams. 
        """
        return Box(low=-np.abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"],
                   shape=(self.max_actions,),
                   dtype=np.float32)
        

    @property
    def observation_space(self):
        
        self.total_obs = 7 * 2 + \
                         self.n_merging_in * 4 + \
                         2 + \
                         int(self.roundabout_length // 5) * 2 + \
                         2
    
        ret = Dict({
            "action_mask": Box(0, 1, shape=(self.max_actions, )),
            "avail_actions": Box(-10, 10, shape=(self.max_actions, 2)),
            "cart": self.wrapped.observation_space,
        })
        box = Box(low=0.,
                  high=1,
                  shape=(self.total_obs,),
                  dtype=np.float32)          
        return box

    def get_state(self, internal=False, **kwargs):
        state_dict = {}

        self.curate_rl_stack()

        rl_id = None
        
        # Get normalization factors 
        circ = self.circumference()
        max_speed = self.k.scenario.max_speed() 
        merge_0_norm = sum([self.k.scenario.edge_length(e) for e in RAMP_0])
        merge_1_norm = sum([self.k.scenario.edge_length(e) for e in RAMP_1])
        queue_0_norm = ceil(merge_0_norm/5 + 1) # 5 is the car length
        queue_1_norm = ceil(merge_1_norm/5 + 1)

        rl_info = self.rl_info(self.rl_stack)
        rl_info_2 = self.rl_info(self.rl_stack_2)
        state_dict['rl_info'] = rl_info
        state_dict['rl_info_2'] = rl_info_2

        # DISTANCES
        # sorted by closest to farthest
        merge_id_0, merge_id_1 = self.k_closest_to_merge(self.n_merging_in)
        merge_dists_0 = self.process(self._dist_to_merge_0(merge_id_0),
                                    length=self.n_merging_in,
                                    normalizer=merge_0_norm)
        merge_dists_1 = self.process(self._dist_to_merge_1(merge_id_1),
                                    length=self.n_merging_in,
                                    normalizer=merge_1_norm)

        state_dict['merge_dists_0'] = merge_dists_0
        state_dict['merge_dists_1'] = merge_dists_1


        # VELOCITIES
        merge_0_vel = self.process(self.k.vehicle.get_speed(merge_id_0),
                                length=self.n_merging_in,
                                normalizer=max_speed)
        merge_1_vel = self.process(self.k.vehicle.get_speed(merge_id_1),
                                length=self.n_merging_in,
                                normalizer=max_speed)

        state_dict['merge_0_vel'] = merge_0_vel
        state_dict['merge_1_vel'] = merge_1_vel

        queue_0, queue_1 = self.queue_length()
        queue_0 = [queue_0 / queue_0_norm]
        queue_1 = [queue_1 / queue_1_norm]

        state_dict['queue_0'] = queue_0
        state_dict['queue_1'] = queue_1
        
        roundabout_full = self.roundabout_full()
        
        # Normalize the 0th column containing absolute position
        roundabout_full[:,0] = roundabout_full[:,0]/self.roundabout_length

        # Normalize the 1st column containing velocities
        roundabout_full[:,1] = roundabout_full[:,1]/max_speed
        roundabout_full = roundabout_full.flatten().tolist()
        state_dict['roundabout_full'] = roundabout_full
        
        state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))

        len_inflow_0 = self.len_inflow_0 / self.max_inflow
        len_inflow_1 = self.len_inflow_1 / self.max_inflow
        state_dict['len_inflow_0'] = [len_inflow_0]
        state_dict['len_inflow_1'] = [len_inflow_1]

        state = np.concatenate([state, [len_inflow_0, len_inflow_1]])
        
        state_dict_keys = ['rl_info', 'rl_info_2',
                           'merge_dists_0', 'merge_0_vel',
                           'merge_dists_1', 'merge_1_vel',
                           'queue_0', 'queue_1',
                           'roundabout_full', 'len_inflow_0', 'len_inflow_1']
        if internal:
            return state_dict, state_dict_keys


        if "state_noise" in self.env_params.additional_params:
            std = self.env_params.additional_params.get("state_noise")
            for i, st in enumerate(state):
                perturbation = np.random.normal(0, std) 
                if "merge_norm_noise" in self.env_params.additional_params \
                    and ((14 <= i < 20) or (26 <= i < 32)): # 14 and 20 are the indices of merge_dists_0
                    merge_norm_noise = self.env_params.additional_params.get("merge_norm_noise")
                    perturbation = np.random.normal(0, merge_norm_noise)
                if "scenario_length_noise" in self.env_params.additional_params \
                    and (i in [0, 4, 6, 7, 11, 13]): # indices of those affected by self.scenario_length
                    scenario_length_noise = self.env_params.additional_params.get("scenario_length_noise")
                    perturbation = np.random.normal(0, scenario_length_noise)
                if "no_inflow_noise" in self.env_params.additional_params \
                    and (i in [92, 93]):
                    perturbation = 0 
                state[i] = st + perturbation

        # Reclip
        if isinstance(self.observation_space, Box):
            state = np.clip(
                state,
                a_min=self.observation_space.low,
                a_max=self.observation_space.high)
        return state 

    # def reset(self):
    #     """See parent class.

    #     The sumo instance is reset with a new ring length, and a number of
    #     steps are performed with the rl vehicle acting as a human vehicle.

    #     Works agnostic of RESTART_INSTANCE value.
    #     Designed to reset inflow lengths every iteration
    #     Base_env's restart_simulation portion is commented out to avoid
    #     resetting inflows every rollout, which is too much, and also to avoid
    #     discrepancy between self.len_inflow_k vs. the reset value
    #     That commented out portion is copied here to restart based on 
    #     inflows agnostic of RESTART_INSTANCE value
    #     """
    #     if self.counter % self.batch_size == 0: # Restarts after every iteration
    #         # Add variable number of inflows here.
    #         inflow = InFlows()
    #         # inflow.add(veh_type="rl", edge="inflow_0", name="rl", vehs_per_hour=50)
    #         inflow.add(veh_type="rl", edge="inflow_1", name="rl", vehs_per_hour=50)
    #         self.len_inflow_0 = np.random.randint(self.range_inflow_0[0], self.range_inflow_0[1]+1)
    #         self.len_inflow_1 = np.random.randint(self.range_inflow_1[0], self.range_inflow_1[1]+1)
    #         # print('############################')
    #         # print('############################')
    #         # print('len_inflow_0: %d' % self.len_inflow_0)
    #         # print('len_inflow_1: %d' % self.len_inflow_1)
    #         # print('############################')
    #         # print('############################')

            
    #         # Forcing the default
    #         if np.random.random() <= 1:
    #             self.len_inflow_0 = 3
    #             self.len_inflow_1 = 3

    #         # for _ in range(self.len_inflow_0):
    #         #     inflow.add(veh_type="rl", edge="inflow_0", name="rl", vehs_per_hour=50)
    #         # for _ in range(self.len_inflow_1):
    #         #     inflow.add(veh_type="rl", edge="inflow_1", name="rl", vehs_per_hour=50)

    #         # update the scenario\
    #         net_params = self.net_params
    #         net_params.inflows = inflow

    #         self.scenario = self.scenario.__class__(
    #             self.scenario.orig_name, self.scenario.vehicles, 
    #             net_params, self.scenario.initial_config)
    #         #------------------------------------------------
    #         # issue a random seed to induce randomness  into the next rollout
    #         # self.sim_params.seed = np.random.randint(0, 1e5)

    #         # self.k.vehicle = deepcopy(self.initial_vehicles)
    #         # self.k.vehicle.master_kernel = self.k
    #         # # restart the sumo instance
    #         # self.restart_simulation(self.sim_params)

    #     # perform the generic reset function
    #     observation = super().reset()
    #     return observation

class MultiAgentUDSSCMerge(UDSSCMergeEnvReset, MultiEnv):
    """Non-adversarial multi-agent env.

    One example of this is: both AVs in the platoon are running separate
    policies. This can later extend to other vehicles in the system, should
    we try a fully autonomous case.
    """
    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super(MultiAgentUDSSCMerge, self).__init__(env_params, sim_params, scenario, simulator='traci')

    @property
    def action_space(self):
        """
        Actions
        Actions are a list of acceleration for the RL vehicle currently being
        controlled, bounded by the maximum accelerations and decelerations
        specified in EnvParams. 
        """
        return Box(low=-np.abs(self.env_params.additional_params["max_decel"]),
                high=self.env_params.additional_params["max_accel"],
                shape=(1,),
                dtype=np.float32)

    @property
    def observation_space(self):
        # self.total_obs = 7 * 2 + \
        #                  self.n_merging_in * 4 + \
        #                  2 + \
        #                  int(self.roundabout_length // 5) * 2 + \
        #                  2
        # self.total_obs = self.n_obs_vehicles * 2 + 2 + \
        #                  int(self.roundabout_length // 5) * 2

        self.total_obs = 2
                         
        box = Box(low=0.,
                  high=1,
                  shape=(self.total_obs,), # -7 for the experiments that only have a single rl_info
                  dtype=np.float32)          
        return box

    def get_state(self, **kwargs):
        """
        The state space is not gucci. Need RL info for every individual RL vehicle

        I think I can actually rewrite the state. Whate we want: distance and vel of each vehicle? Idk if
        absolute velocity will be great, but might as well do this anyway.
        """

        # state_dict, state_dict_keys = super(MultiAgentUDSSCMerge, self).get_state(internal=True, **kwargs)
        # state_dict_keys = ['rl_info', 'rl_info_2',
        #                 'merge_dists_0', 'merge_0_vel',
        #                 'merge_dists_1', 'merge_1_vel',
        #                 'queue_0', 'queue_1',
        #                 'roundabout_full', 'len_inflow_0', 'len_inflow_1']

        # state = np.concatenate([state_dict[key] for key in state_dict_keys])
        # state = np.clip(
        #     state,
        #     a_min=self.observation_space.low,
        #     a_max=self.observation_space.high)

        rl_ids = self.k.vehicle.get_rl_ids()
        ret = {}
        for rl_id in rl_ids:


            max_speed = self.k.scenario.max_speed() 
            
            # rl_pos, rl_vel
            rl_pos = [self.k.vehicle.get_x_by_id(rl_id) / self.scenario_length]
            rl_vel = [self.k.vehicle.get_speed(rl_id) / max_speed] if \
                    self.k.vehicle.get_speed(rl_id)!= -1001 else [0]
            rl_headway = [self.k.vehicle.get_headway(rl_id) / self.scenario_length]
            dist_to_end = [self._dist_to_end(rl_id) / self.max_route_length]

            if self.k.vehicle.get_edge(rl_id) in ROUNDABOUT_EDGES:
                rl_pos_2 = [self.k.vehicle.get_x_by_id(rl_id) / self.roundabout_length]
            else: 
                rl_pos_2 = [0]
            # state = np.concatenate([rl_pos, rl_vel, rl_headway])
            # state = np.concatenate([dist_to_end, rl_vel, rl_headway])
            state = np.concatenate([dist_to_end, rl_vel])
            state = np.clip(
                state,
                a_min=self.observation_space.low,
                a_max=self.observation_space.high)
            # state = np.concatenate([rl_pos, rl_vel, rl_pos_2])
            ret[rl_id] = state
        
        return ret

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        accels = []
        valid_ids = []

        for rl_id, a in rl_actions.items():
            if self.in_control(rl_id):
                accels.append(a)
                valid_ids.append(rl_id)

        # # Apply RL Actions
        # if self.rl_stack:
        #     rl_id = self.rl_stack[0]
        #     if self.in_control(rl_id):
        #         accels.append(rl_action_0)
        #         valid_ids.append(rl_id)

        # if self.rl_stack_2:
        #     rl_id_2 = self.rl_stack_2[0]
        #     if self.in_control(rl_id_2):
        #         accels.append(rl_action_1)
        #         valid_ids.append(rl_id_2)

        # TODO(@evinitsky) why is the human perturbation the wrong size????
        self.k.vehicle.apply_acceleration(valid_ids, np.array(accels))

    def compute_reward(self, rl_actions, **kwargs):
        reward = super(MultiAgentUDSSCMerge, self).compute_reward(rl_actions, **kwargs)
        ret = {}
        #####
        reward = rewards.min_delay(self)
        #####
        for rl_id in self.k.vehicle.get_rl_ids():
            ###
            reward = self.k.vehicle.get_speed(rl_id)
            ###
            ret[rl_id] = reward
        return ret




class MultiAgentUDSSCMergeHumanAdversary(UDSSCMergeEnvReset, MultiEnv):
    """Adversarial multi-agent env.
    Multi-agent env for UDSSC with an adversarial agent perturbing
    the accelerations of the autonomous vehicle. There is also an adversary perturbing the accelerations
    of each of the human drivers
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super(MultiAgentUDSSCMergeHumanAdversary, self).__init__(env_params, sim_params, scenario, simulator='traci')

    @property
    def adv_action_space(self):
        # self.total_obs = 7 * 2 + \
        #                  self.n_merging_in * 4 + \
        #                  2 + \
        #                  int(self.roundabout_length // 5) * 2 + \
        #                  2
        actions = 2
        selective_state = 20

        box = Box(low=-1.0,
                  high=1.0,
                  shape=(selective_state + actions,),
                  dtype=np.float32)
        return box

    @property
    def human_adv_action_space(self):
        # These are just accelerations that can be provided to every human vehicle

        box = Box(low=-1.0,
                  high=1.0,
                  shape=(1,),
                  dtype=np.float32)
        return box

    @property
    def human_adv_obs_space(self):
        # TODO(@evinitsky) what should they actually observe?
        box = Box(low=-3.0,
                  high=3.0,
                  shape=(2,),
                  dtype=np.float32)
        return box

    # <-- ORIGINAL. Commenting out temporarily
    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        av_action = rl_actions['av']
        if self.env_params.additional_params['action_adversary']:
            self.adv_actions = rl_actions['action_adversary']
            adv_action = rl_actions['action_adversary']
            adv_action_weight = 0
            if 'adv_action_weight' in self.env_params.additional_params:
                adv_action_weight = self.env_params.additional_params['adv_action_weight']

            rl_action_0 = av_action[0] + adv_action_weight * adv_action[0]
            rl_action_1 = av_action[1] + adv_action_weight * adv_action[1]

            rl_action_0 = np.clip(rl_action_0,
                                  -self.env_params.additional_params["max_decel"],
                                  self.env_params.additional_params["max_accel"])
            rl_action_1 = np.clip(rl_action_1,
                                  -self.env_params.additional_params["max_decel"],
                                  self.env_params.additional_params["max_accel"])

        else:
            self.adv_actions = np.zeros(self.adv_action_space.shape[0])
            rl_action_0 = av_action[0]
            rl_action_1 = av_action[1]

        # Curation
        removal = []
        removal_2 = []
        for rl_id in self.rl_stack:
            if rl_id not in self.k.vehicle.get_rl_ids():
                removal.append(rl_id)
        for rl_id in self.rl_stack_2:
            if rl_id not in self.k.vehicle.get_rl_ids():
                removal_2.append(rl_id)
        for rl_id in removal:
            self.rl_stack.remove(rl_id)
        for rl_id in removal_2:
            self.rl_stack_2.remove(rl_id)

        accels = []
        valid_ids = []

        # Apply RL Actions
        if self.rl_stack:
            rl_id = self.rl_stack[0]
            if self.in_control(rl_id):
                accels.append(rl_action_0)
                valid_ids.append(rl_id)

        if self.rl_stack_2:
            rl_id_2 = self.rl_stack_2[0]
            if self.in_control(rl_id_2):
                accels.append(rl_action_1)
                valid_ids.append(rl_id_2)

        # Now go through the humans in the scene and perturb all of their actions
        for veh_id, accel in rl_actions.items():
            if veh_id != 'action_adversary' and veh_id != 'av':
                base_accel = self.k.vehicle.get_acc_controller(veh_id).get_accel(self)
                accels.append(base_accel + accel[0])
                valid_ids.append(veh_id)

        # TODO(@evinitsky) why is the human perturbation the wrong size????
        self.k.vehicle.apply_acceleration(valid_ids, np.array(accels))


    def compute_reward(self, rl_actions, **kwargs):
        """The agent receives the class definition reward,
        the adversary recieves the negative of the agent reward
        """
        reward = super(MultiAgentUDSSCMergeHumanAdversary, self).compute_reward(rl_actions, **kwargs)
        # human_dict = {veh_id: -reward for veh_id in self.k.vehicle.get_human_ids()}
        reward_dict = {'av': reward}
        # reward_dict.update(human_dict)
        if self.env_params.additional_params['action_adversary']:
            reward_dict['action_adversary'] = -reward
        # Go through the human drivers and add zeros if the vehicles have left as a final observation
        # left_vehicles_dict = {veh_id: 0 for veh_id
        #                       in self.k.vehicle.get_arrived_ids()}
        # reward_dict.update(left_vehicles_dict)

        return reward_dict

    def get_state(self, **kwargs):
        """See class definition for the state. Both adversary and
        agent receive the same state
        """
        state = super(MultiAgentUDSSCMergeHumanAdversary, self).get_state(**kwargs)
        # quick defensive check
        # assert super(MultiAgentUDSSCMergeHumanAdversary, self).get_state == UDSSCMergeEnvReset.get_state

        adv_state_weight = 0
        if 'adv_state_weight' in self.env_params.additional_params:
            adv_state_weight = self.env_params.additional_params['adv_state_weight']
        try:
            perturb = self.adv_actions[2:] * adv_state_weight
        except:
            perturb = 0

        if perturb is not 0:
            state[0:3] += perturb[:3]
            state[7:10] += perturb[3:6]
            state[14:20] += perturb[6:12]
            state[26:32] += perturb[12:18]

        state = np.clip(
            state,
            a_min=self.observation_space.low,
            a_max=self.observation_space.high)

        state_dict = {}
        state_dict['av'] = state
        if self.env_params.additional_params['action_adversary']:
            state_dict['action_adversary'] = state
        # the adversary driving the human cars
        # human_ids = self.k.vehicle.get_human_ids()
        # human_state_dict = {human_id: np.array([np.clip(self.k.vehicle.get_headway(human_id) / 1000.0, 0, 1),
        #                      np.clip(self.k.vehicle.get_speed(human_id) / 60.0, 0, 1)]) for human_id in human_ids}
        # state_dict.update(human_state_dict)

        # # Go through the human drivers and add zeros if the vehicles have left as a final observation
        # left_vehicles_dict = {veh_id: np.zeros(self.human_adv_obs_space.shape[0]) for veh_id
        #                       in self.k.vehicle.get_arrived_ids()}
        # state_dict.update(left_vehicles_dict)
        
        
        return state_dict 

    def reset(self, new_inflow_rate=None):
        """See parent class.
        The sumo instance is reset with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        super(MultiAgentUDSSCMergeHumanAdversary, self).reset()
        observation = self.get_state()
        return observation
