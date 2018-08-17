import traci.constants as tc

# DEFAULTS
PROGRAM_ID = 1
MAX_GAP = 3.0
DETECTOR_GAP = 0.8
SHOW_DETECTORS = True
""" ORIGINAL ROUTE
# rts = {"top": ["top", "left", "bottom", "right"],
#        "left": ["left", "bottom", "right", "top"],
#        "bottom": ["bottom", "right", "top", "left"],
#        "right": ["right", "top", "left", "bottom"],
#        "inflow_1": ["inflow_1", "merge_in_1", "right", "top", "left", "bottom"],
#        "merge_in_1": ["merge_in_1", "right", "top", "left", "bottom"],
#        "inflow_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
#        "merge_in_0": ["merge_in_0", "left", "merge_out_1", "outflow_1"],
#        "merge_out_1": ["merge_out_1", "outflow_1"],
#        "outflow_1": ["outflow_1"],
#        }
"""
""" DISTRIBUTION ROUTE
rts =  {"top": {"top": ["top", "left", "bottom", "right"]},
        "left": {"left": ["left", "bottom", "right", "top"]},
        "bottom": {"bottom": ["bottom", "right", "top", "left"]},
        "right": {"right": ["right", "top", "left", "bottom"]},

        "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"],
                    "inflow_1_1": ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"]}, # added
        "inflow_0": {"inflow_0_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
                    "inflow_1_1": ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"]},
        "outflow_1": {"outflow_1": ["outflow_1"]},
        "outflow_0": {"outflow_0": ["outflow_0"]}
        }
"""

class Routes:

    def __init__(self, distributed=False):
        """Base route.

        description here

        Parameters
        ----------
        baseline: bool
        """
        self.routes = {}
        self.distributed = distributed

    def add(self, start, route=[], prob=1):
        """Adds a route to the network.

        for the base route case 

        Parameters
        ----------
        node_id : str
            name of the node with traffic lights
        tls_type : str, optional
            type of the traffic light (see Note)

        Note
        ----
        For information on defining traffic light properties, see:
        http://sumo.dlr.de/wiki/Simulation/Traffic_Lights#Defining_New_TLS-Programs
        """

        self.routes[start] = 

    def update(self, tls_subscriptions):
        """Updates the states and phases of the traffic lights to match current
        traffic light data.

        Parameters
        ----------
        tls_subscriptions : dict
            sumo traffic light subscription data
        """
        self.__tls = tls_subscriptions.copy()

    def get_ids(self):
        """Returns the names of all nodes with traffic lights."""
        return self.__ids

    def get_properties(self):
        """Returns traffic light properties. This is meant to be used by the
        generator to import traffic light data to the .net.xml file"""
        return self.__tls_properties

    def set_state(self, node_id, state, env, link_index="all"):
        """Sets the state of the traffic lights on a specific node.

        Parameters
        ----------
        node_id : str
            name of the node with the controlled traffic lights
        state : str
            requested state(s) for the traffic light
        env : flow.envs.base_env.Env type
            the environment at the current time step
        link_index : int, optional
            index of the link whose traffic light state is meant to be changed.
            If no value is provided, the lights on all links are updated.
        """
        if link_index == "all":
            # if lights on all lanes are changed
            env.traci_connection.trafficlight.setRedYellowGreenState(
                tlsID=node_id, state=state)
        else:
            # if lights on a single lane is changed
            env.traci_connection.trafficlight.setLinkState(
                tlsID=node_id, tlsLinkIndex=link_index, state=state)

    def get_state(self, node_id):
        """Returns the state of the traffic light(s) at the specified node

        Parameters
        ----------
        node_id: str
            name of the node

        Returns
        -------
        state : str
            Index = lane index
            Element = state of the traffic light at that node/lane
        """
        return self.__tls[node_id][tc.TL_RED_YELLOW_GREEN_STATE]

