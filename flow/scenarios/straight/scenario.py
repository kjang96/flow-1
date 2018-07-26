import numpy as np

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights

ADDITIONAL_NET_PARAMS = {
    # max speed limit of the network
    "speed_limit": 30,
    "length": 500
}


class StraightScenario(Scenario):

    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initializes a merge scenario.

        Requires from net_params:
        - merge_length: length of the merge edge
        - pre_merge_length: length of the highway leading to the merge
        - post_merge_length: length of the highway past the merge
        - merge_lanes: number of lanes in the merge
        - highway_lanes: number of lanes in the highway
        - speed_limit: max speed limit of the network

        See Scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        edgestarts = [
            ("main", 0)
        ]

        return edgestarts

    def specify_internal_edge_starts(self):
        internal_edgestarts = []

        return internal_edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """

        elif self.initial_config.spacing == "custom":
                startpositions, startlanes = self.gen_custom_start_pos(
                    self.initial_config, num_vehicles, **kwargs)

        ipdb> startpositions
        [('main', 0), ('main', 500.0)]
        ipdb> startlanes
        [0, 0]

        Assumptions: index 0 is the rl vehicle, index 1 is idm 
        Add some noise to the spacing 
        """
        startlanes = [0, 0]
        idm_spacing = 10 + np.random.normal(0, 2)
        if idm_spacing < 6:
            idm_spacing = 6 # minimum
        startpositions = [('main', 0), ('main', float(idm_spacing))]

        return startpositions, startlanes
          
