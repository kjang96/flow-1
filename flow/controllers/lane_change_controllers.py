from flow.controllers.base_lane_changing_controller import \
    BaseLaneChangeController


class SumoLaneChangeController(BaseLaneChangeController):
    """A controller used to enforce sumo lane-change dynamics on a vehicle."""

    def __init__(self,
                 veh_id,
                 lane_change_mode="no_lat_collide",
                 sumo_lc_params=None):
        super().__init__(veh_id,
                         lane_change_mode=lane_change_mode,
                         sumo_lc_params=sumo_lc_params)
        self.SumoController = True

    def get_lane_change_action(self, env):
        return None


class StaticLaneChanger(BaseLaneChangeController):
    """A lane-changing model used to keep a vehicle in the same lane."""

    def get_lane_change_action(self, env):
        return 0
