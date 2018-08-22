from flow.controllers.base_controller import BaseController


class RLController(BaseController):

    def __init__(self,
                 veh_id,
                 speed_mode='right_of_way',
                 sumo_car_following_params=None):
        """Instantiates an RL Controller.

        Vehicles with this controller are provided with actions by an rl agent,
        and perform their actions accordingly.

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        speed_mode : str
            see parent class
        sumo_car_following_params : SumoCarFollowingParams
            see parent class
        """
        BaseController.__init__(
            self, veh_id,
            speed_mode=speed_mode,
            sumo_car_following_params=sumo_car_following_params)
