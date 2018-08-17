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
            1 + self.n_preceding + self.n_following + self.n_merging_in

        self.obs_var_labels = \
            ["speed", "pos", "queue_length", "velocity_stats"]

        super().__init__(env_params, sumo_params, scenario)

    @property
    def observation_space(self):
        # speed = Box(low=0, high=np.inf, shape=(self.n_obs_vehicles,),
        #             dtype=np.float32)
        # absolute_pos = Box(low=0., high=np.inf, shape=(self.n_obs_vehicles,),
        #                    dtype=np.float32)
        # queue_length = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        # vel_stats = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # return Tuple((speed, absolute_pos, queue_length, vel_stats))
        speed = Box(low=0, high=np.inf, shape=(self.n_obs_vehicles,),
                    dtype=np.float32)
        relative_pos = Box(low=0., high=np.inf, shape=(self.n_obs_vehicles,),
                           dtype=np.float32) # Normalized by ring circumference 
        return Tuple((speed, relative_pos))

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
        ## changes start
        # import ipdb; ipdb.set_trace()

        # TODO: NORMALIZE EVERYTHING
        rl_pos = self.get_x_by_id('rl_0')
        rl_vel = self.vehicles.get_speed('rl_0')
        close_0 = np.zeros(self.n_merging_in)
        close_1 = np.zeros(self.n_merging_in)

        merge_0, merge_1 = self.k_closest_to_merge(k)
        tailway, headway = self.k_closest_to_rl('rl_0', k)
        
        velocities = self.vehicles.get_speed([merge_ids] + [close_ids])
        rl_distances = [item[1] for item in [tailway] + [headway]]
        merge_distances = self._dist_to_merge_0(merge_0) + 
                          self._dist_to_merge_1(merge_1)
        state = np.array([rl_pos, rl_vel, velocities,
                          rl_distances, merge_distances])
        return state
        
        # Calculate distances. 
        


        ## changes end

        ### OLD STUFF STARTS
        # sorted_data = self.sorted_extra_data
        # merge_len = self.scenario.intersection_length

        # # Merge stretch is pos 0.0-25.5 (ish), so actively merging vehicles
        # # are sorted at the front of the list. Otherwise, vehicles closest to
        # # the merge are at the end of the list (effectively reverse sorted).
        # if self.get_x_by_id(sorted_data[0]) < merge_len and self.get_x_by_id(
        #         sorted_data[1]) < merge_len:
        #     if not sorted_data[0].startswith("merge") and \
        #             not sorted_data[1].startswith("merge"):
        #         vid1 = sorted_data[-1]
        #         vid2 = sorted_data[-2]
        #     elif not sorted_data[0].startswith("merge"):
        #         vid1 = sorted_data[1]
        #         vid2 = sorted_data[-1]
        #     elif not sorted_data[1].startswith("merge"):
        #         vid1 = sorted_data[0]
        #         vid2 = sorted_data[-1]
        #     else:
        #         vid1 = sorted_data[1]
        #         vid2 = sorted_data[0]
        # elif self.get_x_by_id(sorted_data[0]) < merge_len:
        #     vid1 = sorted_data[0]
        #     vid2 = sorted_data[-1]
        # else:
        #     vid1 = sorted_data[-1]
        #     vid2 = sorted_data[-2]
        # pos[-2] = self.get_x_by_id(vid1)
        # pos[-1] = self.get_x_by_id(vid2)
        # vel[-2] = self.vehicles.get_speed(vid1)
        # vel[-1] = self.vehicles.get_speed(vid2)

        # # find and eliminate all the vehicles on the outer ring
        # num_inner = len(sorted_data)

        # rl_vehID = self.vehicles.get_rl_ids()[0]
        # rl_srtID, = np.where(sorted_data == rl_vehID)
        # rl_srtID = rl_srtID[0]

        # # FIXME(cathywu) hardcoded for self.num_preceding = 2
        # lead_id1 = sorted_data[(rl_srtID + 1) % num_inner]
        # lead_id2 = sorted_data[(rl_srtID + 2) % num_inner]
        # # FIXME(cathywu) hardcoded for self.num_following = 2
        # follow_id1 = sorted_data[(rl_srtID - 1) % num_inner]
        # follow_id2 = sorted_data[(rl_srtID - 2) % num_inner]
        # vehicles = [rl_vehID[0], lead_id1, lead_id2, follow_id1, follow_id2]

        # vel[:self.n_obs_vehicles - self.n_merging_in] = np.array(
        #     self.vehicles.get_speed(vehicles))
        # pos[:self.n_obs_vehicles - self.n_merging_in] = np.array(
        #     [self.get_x_by_id(veh_id) for veh_id in vehicles])

        # # normalize the speed
        # # FIXME(cathywu) can divide by self.max_speed
        # normalized_vel = np.array(vel) / self.scenario.max_speed

        # # normalize the position
        # normalized_pos = np.array(pos) / self.scenario.length

        # # Compute number of vehicles in the outer ring
        # queue_length = np.zeros(1)
        # queue_length[0] = len(sorted_data) - num_inner

        # # Compute mean velocity on inner and outer rings
        # # Note: merging vehicles count towards the inner ring stats
        # vel_stats = np.zeros(2)
        # vel_all = self.vehicles.get_speed(sorted_data)
        # vel_stats[0] = np.mean(vel_all[:num_inner])
        # vel_stats[1] = np.mean(vel_all[num_inner:])
        # vel_stats = np.nan_to_num(vel_stats)

        # return np.array([normalized_vel, normalized_pos, queue_length,
        #                  vel_stats]).T
        ### OLD STUFF ENDS 

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
        close_0 = np.zeros(self.n_merging_in)
        close_1 = np.zeros(self.n_merging_in)

        # both dists_0 and dists_1 increase the closer you get to the intersection
        # dists_0 = self.vehicles.get_x_by_id(self.vehicles.get_ids_by_edge(["merge_in_0", "inflow_0"]))
        # dists_1 = self.vehicles.get_x_by_id(self.vehicles.get_ids_by_edge(["merge_in_1", "inflow_1"]))

        # total_len = self.scenario.edge_length("merge_in_0") +
        #             self.scenario.edge_length("inflow_0")

        # return self.total_edgestarts_dict[edge] + position


        dists_0 = sorted(self.vehicles.get_ids_by_edge(["merge_in_0", "inflow_0"]), 
                         key=lambda veh_id:
                         -self.vehicles.get_x_by_id(veh_id))

        dists_1 = sorted(self.vehicles.get_ids_by_edge(["merge_in_1", "inflow_1"]), 
                         key=lambda veh_id:
                         -self.vehicles.get_x_by_id(veh_id))

        for i in range(min(k, len(dists_0))):
            close_0[i] = dists_0

        for i in range(min(k, len(dists_1))):
            close_1[i] = dists_1
        
        return close_0, close_1

    def k_closest_to_rl(self, rl_id, k):
        """
        Return a list of ids and  distances to said rl vehicles
        """

        # Arrays holding veh_ids of vehicles before/after the rl_id.
        # Will always be sorted by distance to rl_id 
        k_tailway = [] 
        k_headway = []

        # Prep.
        route = ["top", "left", "bottom", "right"]
        rl_edge = self.vehicles.get_edge(rl_id)
        rl_index = route.index(rl_edge) 
        rl_x = self.get_x_by_id(rl_id)
        rl_pos = self.scenario.get_position(rl_id)

        # Get preceding first.
        for i in range(rl_index, rl_index-2, -1): # Curr  edge and preceding edge
            if i == rl_index: # Same edge as rl_id
                veh_ids = [(v, rl_pos - self.scenario.get_position(v)) 
                           for v in self.vehicles.get_ids_by_edge(route[i]) 
                           if self.scenario.get_position(v) < rl_pos]
            else: # Preceding edge 
                edge_len = self.scenario.edge_length(route[i])
                veh_ids = [(v, rl_pos + (edge_len - self.vehicles.get_position(v))) 
                           for v in self.vehicles.get_ids_by_edge(route[i])]
            sorted_vehs = sorted(veh_ids, lambda v: self.get_x_by_id(v))
            # k_tailway holds veh_ids in decreasing order of closeness 
            k_tailway = sorted_vehs + k_tailway

        # Get headways second.
        for i in range(rl_index, rl_index+2):
            # If statement is to cover the case of overflow in get_x 
            if i == rl_index: # Same edge as rl_id
                veh_ids = [(v, self.scenario.get_position(v) - rl_pos) 
                           for v in self.vehicles.get_ids_by_edge(route[i]) 
                           if self.scenario.get_position(v) > rl_pos]
            else:
                rl_dist = self.scenario.edge_length(rl_edge) 
                          - self.scenario.get_position(rl_id)
                veh_ids = [(v, self.vehicles.get_position(v) + rl_dist)
                           for self.vehicles.get_ids_by_edge(route[i])]
            # The following statement is safe because it's being done
            # by one edge at a time
            sorted_vehs = sorted(veh_ids, lambda v: self.get_x_by_id(v))
            k_headway += sorted_vehs

        # After this step, truncate if necessary, or pad with zeros.
        if len(k_tailway) > k:
            k_tailway = k[-k:]
        else:
            k_tailway = [0] * (k - len(k_tailway))
                
        if len(k_headway) > k:
            k_headway = k[:k]
        else: 
            k_headway += [0] * (k - len(k_headway))

        return k_tailway, k_headway

    def _dist_to_merge_1(self, veh_id):
        # distances = []
        reference = self.scenario.total_edgestarts_dict["merge_in_1"] +
                    self.scenario.edge_length("merge_in_1")
        distances = [reference - self.vehicles.get_x_by_id(v)
                     for v in veh_id]
        return distances

    def _dist_to_merge_0(self, veh_id):
        reference = self.scenario.total_edgestarts_dict["merge_in_0"] +
                    self.scenario.edge_length("merge_in_0")
        distances = [reference - self.vehicles.get_x_by_id(v)
                     for v in veh_id]
        return distances

    def _dist_to_rl(self, rl_id, veh_id):
        rl_edge = self.vehicles.get_edge(rl_id)
        route = 
        distances = 
        for veh_