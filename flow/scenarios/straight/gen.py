from flow.core.generator import Generator

from numpy import pi, sin, cos

START = 0
END = 200

class StraightGenerator(Generator):
    """Generator for a straight road network."""

    def __init__(self, net_params, base):
        self.name = "delaware"
        self.length = net_params.additional_params.get("length")
        super().__init__(net_params, base)

    def specify_nodes(self, net_params):
        nodes = [
            {"id": "start", "x": repr(START), "y": repr(0)},
            {"id": "end", "x": repr(self.length), "y": repr(0)},
        ]

        return nodes

    def specify_edges(self, net_params):
        edges = [
            {"id": "main", "type": "horizontal",
             "from": "start", "to": "end", "length": repr(self.length-START) }
        ]

        return edges

    def specify_types(self, net_params):
        speed = net_params.additional_params["speed_limit"]
        types = [{"id": "horizontal", "numLanes": repr(1),
                  "speed": repr(speed)}]

        return types

    def specify_routes(self, net_params):
        rts = {"main": ["main"]}

        return rts
