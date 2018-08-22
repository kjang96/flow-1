from flow.core.params import SumoLaneChangeParams
import logging

LC_MODES = {
    # execute all TraCI commands
    "aggressive": 0,
    # collision avoidance and safety-gap enforcement
    "no_lat_collide": 512,
    # the laneChangeModel may execute all changes unless in conflict with TraCI
    "strategic": 1621
}


class BaseLaneChangeController:

    def __init__(self,
                 veh_id,
                 lane_change_mode="no_lat_collide",
                 sumo_lc_params=None):
        """Base class for lane-changing controllers.

        Instantiates a controller and forces the user to pass a
        lane_changing duration to the controller. Provides the method
        get_safe_lane_change_action to ensure that lane-changes do
        not cause crashes.

        Attributes
        ----------
        veh_id: string
            ID of the vehicle this controller is used for
        lane_change_mode: str or int, optional
            may be one of the following:

            * "no_lat_collide" (default): Human cars will not make lane
              changes, RL cars can lane change into any space, no matter how
              likely it is to crash
            * "strategic": Human cars make lane changes in accordance with SUMO
              to provide speed boosts
            * "aggressive": RL cars are not limited by sumo with regard to
              their lane-change actions, and can crash longitudinally
            * int values may be used to define custom lane change modes for the
              given vehicles, specified at:
              http://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#lane_change_mode_.280xb6.29

        sumo_lc_params: flow.core.params.SumoLaneChangeParams type
            Params object specifying attributes for Sumo lane changing model.
        """
        if sumo_lc_params is None:
            sumo_lc_params = SumoLaneChangeParams()

        self.veh_id = veh_id
        self.sumo_lc_params = sumo_lc_params

        # adjust the lane change mode value
        if isinstance(lane_change_mode, str) and lane_change_mode in LC_MODES:
            lane_change_mode = LC_MODES[lane_change_mode]
        elif not (isinstance(lane_change_mode, int)
                  or isinstance(lane_change_mode, float)):
            logging.error("Setting lane change mode of {0} to "
                          "default.".format(veh_id))
            lane_change_mode = LC_MODES["no_lat_collide"]

        self.lane_change_mode = lane_change_mode

        # min_gap defines the minimum gap (in meters) that a car is willing
        # to accept in front and behind it to enter a gap
        self.min_gap = sumo_lc_params.controller_params.get("min_gap", 0.1)

    def get_lane_change_action(self, env):
        """Specifies the lane change action to be performed.

        If discrete lane changes are being performed, the action is a direction
         -1: lane change right
         0: no lane change
         1: lane change left

        Parameters
        ----------
        env: Env type
            state of the environment at the current time step

        Returns
        -------
        lc_action: float or int
            requested lane change action
        """
        raise NotImplementedError

    def get_action(self, env):
        """Returns the action of the lane change controller

        Modifies the lane change action to ensure safety, if requested.

        Returns
        -------
        lc_action: float or int
            lane change action
        """
        lc_action = self.get_lane_change_action(env)
        # TODO(ak): add failsafe

        return lc_action

    def get_safe_lane_change_action(self, env, target_lane):
        """Determines whether a collision will occur if a vehicle enters the
        target lane.

        Parameters
        ----------
        env: Env type
            state of the environment at the current time step
        target_lane: int
            requested target lane by the controller

        Returns
        -------
        safe_lane: int
            the safe target lane (requested target lane if action is safe,
            current lane if the action is not)
        """
        current_lane = env.vehicles.get_lane(self.veh_id)

        # if no lane change is being performed, there is no need to check for
        # safety
        if current_lane == target_lane:
            return target_lane

        # if the target lane is not within the range of lanes available, return
        # the current lane
        if target_lane < 0 or target_lane > env.scenario.lanes - 1:
            return current_lane

        lead_id = env.get_leading_car(self.veh_id, target_lane)
        trail_id = env.get_trailing_car(self.veh_id, target_lane)

        # if there is only one vehicle in the environment, or there are no
        # vehicles in the target lane, then lane changing to the target lane
        # is safe
        if (lead_id is None) or (env.vehicles.num_vehicles == 1):
            return target_lane

        lead_pos = env.get_x_by_id(lead_id)
        lead_length = env.vehicles.get_length(lead_id)

        trail_pos = env.get_x_by_id(trail_id)
        trail_vel = env.vehicles.get_speed(trail_id)

        this_pos = env.get_x_by_id(self.veh_id)
        this_vel = env.vehicles.get_speed(self.veh_id)
        this_length = env.vehicles.get_length(self.veh_id)

        lead_gap = (lead_pos - this_pos) % env.scenario.length - lead_length
        trail_gap = (this_pos - trail_pos) % env.scenario.length - this_length

        time_step = env.time_step

        max_acc = env.env_params.max_acc
        max_trail_vel = trail_vel + max_acc * time_step
        max_this_vel = this_vel + max_acc * time_step

        # if vehicle may collide into a vehicle in the target lane in the lane
        # change is performed, opt out of lane change
        if lead_gap - max_this_vel * time_step < self.min_gap or \
                trail_gap - max_trail_vel * time_step < self.min_gap:
            return current_lane
        else:
            return target_lane
