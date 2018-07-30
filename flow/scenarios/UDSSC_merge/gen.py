from flow.core.generator import Generator

from numpy import pi, sin, cos, linspace


class UDSSCMergingGenerator(Generator):
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

        self.name = "%s-%dr%dl" % (base, radius,
                                   self.inner_lanes + self.outer_lanes)

    def specify_nodes(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        # KJ what's the point of them all being priority?
        nodes = [{"id": "first", "x": repr(0), "y": repr(0),
                  "type": "priority"},
                 {"id": "second", "x": repr(x), "y": repr(0),
                  "type": "priority"},
                 {"id": "third", "x": repr(x+r), "y": repr(r),
                  "type": "priority"},  
                 {"id": "fourth", "x": repr(x+r), "y": repr(x+r),
                  "type": "priority"},
                ]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]
        circumference = 2 * pi * r
        # ring_edgelen = pi * r
        resolution = 40

        edges = [
            # originally bottom_left to top_left
            {"id": "circus_right", "from": "second", "to": "third",
             "type": "edgeType", "length": repr(circumference / 4),
             "priority": "46", "shape": " ".join(
                 ["%.2f,%.2f" % (r * cos(t) + x, r * sin(t) + r)
                  for t in linspace(- pi / 2, 0, resolution)]),
             "numLanes": str(self.inner_lanes)},

            # {"id": "center", "from": "bottom_left", "to": "top_left",
            #  "type": "edgeType", "length": repr(ring_edgelen),
            #  "priority": "46", "shape": " ".join(
            #      ["%.2f,%.2f" % (r * cos(t), r * sin(t))
            #       for t in linspace(- pi / 2, pi / 2, resolution)]),
            #  "numLanes": str(self.inner_lanes)},

            {"id": "left", "from": "first", "to": "second",
             "type": "edgeType", "length": repr(x), "priority": "46",
             "numLanes": str(self.outer_lanes)},

            # {"id": "top", "from": "top_right", "to": "top_left",
            #  "type": "edgeType", "length": repr(x), "priority": "46",
            #  "numLanes": str(self.outer_lanes)},

            {"id": "right", "from": "third", "to": "fourth",
             "type": "edgeType", "length": repr(x),
             "numLanes": str(self.outer_lanes)},

            # {"id": "bottom", "from": "bottom_left", "to": "bottom_right",
            #  "type": "edgeType", "length": repr(x),
            #  "numLanes": str(self.outer_lanes)},

            {"id": "circus_left", "from": "third", "to": "second",
             "type": "edgeType", "length": repr(3 / 4 * circumference ),
             "shape": " ".join(
                 ["%.2f,%.2f" % (r * cos(t) + x, r * sin(t) + r)
                  for t in linspace(0, 3 * pi / 2, resolution)]),
             "numLanes": str(self.inner_lanes)},

            # {"id": "left", "from": "top_left", "to": "bottom_left",
            #  "type": "edgeType", "length": repr(ring_edgelen),
            #  "shape": " ".join(
            #      ["%.2f,%.2f" % (r * cos(t), r * sin(t))
            #       for t in linspace(pi / 2, 3 * pi / 2, resolution)]),
            #  "numLanes": str(self.inner_lanes)},

            # {"id": "right", "from": "bottom_right", "to": "top_right",
            #  "type": "edgeType", "length": repr(ring_edgelen),
            #  "shape": " ".join(
            #      ["%.2f,%.2f" % (x + r * cos(t), r * sin(t))
            #       for t in linspace(- pi / 2, pi / 2, resolution)]),
            #  "numLanes": str(self.outer_lanes)},

        ]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{"id": "edgeType", "speed": repr(speed_limit)}]
        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """
        rts = {
               "circus_right": ["circus_right", "circus_left"]
            #    "left": ["left", "center", "left"],
            #    "center": ["center", "left", "center"]
              }

        return rts
