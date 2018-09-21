from flow.core.generator import Generator
from flow.core.routes import Routes

from numpy import pi, sin, cos, linspace, sqrt


class RoundaboutGenerator(Generator):
    """
    Generator for a two-loop network in which both loops merge into a common
    lane.
    """
    def __init__(self, net_params, base):
        """
        See parent class
        """
        radius = net_params.additional_params["ring_radius"]
        self.inner_lanes = net_params.additional_params["inner_lanes"]
        self.outer_lanes = net_params.additional_params["outer_lanes"]

        super().__init__(net_params, base)

        # self.name = "%s-%dr%dl" % (base, radius,
        #                            self.inner_lanes + self.outer_lanes)

    def specify_nodes(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]
        m = self.net_params.additional_params["merge_length"]

        roundabout_type = "priority"
        default = "priority"

        nodes = [{"id": "a",   "x": repr(0),  "y": repr(-r), "type": roundabout_type},
                 {"id": "b",   "x": repr(0.5 * r),  "y": repr(sqrt(3)/2 * r), "type": roundabout_type},
                 {"id": "c",   "x": repr(-0.5 * r),  "y": repr(sqrt(3)/2 * r), "type": roundabout_type},
                 {"id": "d",   "x": repr(-r), "y": repr(0), "type": roundabout_type},
                 {"id": "e",   "x": repr(0), "y": repr(r + m), "type": default},
                 {"id": "f",   "x": repr(0), "y": repr(r + m + x), "type": default},
                 {"id": "g",   "x": repr(-r - m), "y": repr(-r - 0.1*r), "type": default},
                 {"id": "h",   "x": repr(-r - m - x), "y": repr(-r - 0.2*r), "type": default},
                ]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]
        circumference = 2 * pi * r
        lanes = repr(net_params.additional_params["lane_num"])
        
        resolution = net_params.additional_params["resolution"]

        length = net_params.additional_params["length"]
        # edgelen = length / 4.
        circ = 2 * pi * r
        twelfth = circ / 12
        edges = [
            {"id": "bottom",
             "type": "edgeType_hi",
             "from": "d",
             "to": "a",
             "numLanes": lanes,
             "length": repr(twelfth * 3),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(-pi, -pi/2 , resolution)])},

            {"id": "right",
             "type": "edgeType_hi",
             "from": "a",
             "to": "b",
             "numLanes": lanes,
             "length": repr(twelfth * 5),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(-pi / 2,pi/3, resolution)])},

            {"id": "top",
             "type": "edgeType_hi",
             "from": "b",
             "to": "c",
             "numLanes": lanes,
             "length": repr(twelfth * 2),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(pi/3, 2*pi/3, resolution)])},

            {"id": "left",
             "type": "edgeType_hi",
             "from": "c",
             "to": "d", 
             "numLanes": lanes,
             "length": repr(twelfth * 2),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(2*pi/3, pi, resolution)])},

            {"id": "merge_out_0",
             "type": "edgeType_lo",
             "from": "b",
             "to": "e",
             "numLanes": lanes,
            },

            {"id": "merge_in_0",
             "type": "edgeType_lo",
             "from": "e",
             "to": "c",
             "numLanes": lanes,
            },

            {"id": "outflow_0",
             "type": "edgeType_lo",
             "from": "e",
             "to": "f",
             "numLanes": lanes,
            },

            {"id": "inflow_0",
             "type": "edgeType_lo",
             "from": "f",
             "to": "e",
             "numLanes": lanes,
            },

            {"id": "merge_out_1",
             "type": "edgeType_lo",
             "from": "d",
             "to": "g",
             "numLanes": lanes,
            },
            
            {"id": "merge_in_1",
             "type": "edgeType_lo",
             "from": "g",
             "to": "a",
             "numLanes": lanes,
            },

            {"id": "outflow_1",
             "type": "edgeType_lo",
             "from": "g",
             "to": "h",
             "numLanes": lanes,
            },

            {"id": "inflow_1",
             "type": "edgeType_lo",
             "from": "h",
             "to": "g",
             "numLanes": lanes,
            },
        ]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        types = [{"id": "edgeType_hi",
                  "speed": repr(net_params.additional_params.get("roundabout_speed_limit")),
                  "priority": repr(2)},
                 {"id": "edgeType_lo",
                  "speed": repr(net_params.additional_params.get("outside_speed_limit")),
                  "priority": repr(1)}]
        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """

        # rts = {"top": {"top": ["top", "left", "bottom", "right"]},
        #        "left": {"left": ["left", "bottom", "right", "top"]},
        #        "bottom": {"bottom": ["bottom", "right", "top", "left"]},
        #        "right": {"right": ["right", "top", "left", "bottom"]},

        #        "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"]}, # added
        #     #    "inflow_0": {"inflow_1_1": ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"]},

        #     #    "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"],
        #     #                 "inflow_1_1": ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"]}, # added
        #        "inflow_0": {"inflow_0_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
        #                     "inflow_1_1": ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"]},

        #        "outflow_1": {"outflow_1": ["outflow_1"]},
        #        "outflow_0": {"outflow_0": ["outflow_0"]}
        #        }

        routes = Routes()
        routes.add("top_0", ["top", "left", "bottom", "right"])
        routes.add("left_0", ["left", "bottom", "right", "top"])
        routes.add("bottom_0", ["bottom", "right", "top", "left"])
        routes.add("right_0", ["right", "top", "left", "bottom"])
        routes.add("inflow_1_0", ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"])
        # routes.add("inflow_1_1", ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"])
        routes.add("inflow_0_0", ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"])
        # routes.add("inflow_0_1", ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"])
        routes.add("outflow_1", ["outflow_1"])
        routes.add("outflow_0", ["outflow_0"])
        return routes

