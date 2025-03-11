from __future__ import annotations
from sumo_rl.models.commons import Point
from sumo_rl.models.commons import indentation

class Lane:
  def __init__(self, id: str, index: int, speed: float, length: float) -> None:
    self.id: str = id
    self.index: int = index
    self.speed: float = speed
    self.length: float = length

  def to_xml(self, indent: int = 0) -> str:
    return (
        indentation(indent) +
        "<lane id=\"%s\" index=\"%s\" speed=\"%s\" length=\"%s\" shape=\"\"/>" % (
          self.id, self.index, self.speed, self.length
        ))

  def __repr__(self) -> str:
    return self.to_xml(0)

  @staticmethod
  def name(edge_id: str, lane_id: int) -> str:
    return "%s_%s" % (edge_id, lane_id)

class Edge:
  def __init__(self, id: str, from_junction: str, to_junction: str, shape: list[Point], lanes: list[Lane]) -> None:
    self.id: str = id
    self.from_junction: str = from_junction
    self.to_junction: str = to_junction
    self.shape: list[Point] = shape
    self.lanes: list[Lane] = lanes

  def to_xml(self, indent: int = 0) -> str:
    header = (
      indentation(indent) +
      "<edge id=\"%s\" from=\"%s\" to=\"%s\" priority=\"-1\" spreadType=\"center\" shape=\"%s\">" % (
        self.id, self.from_junction, self.to_junction, " ".join([p.to_str() for p in self.shape])
      )
    )
    children = [
      lane.to_xml(indent + 1) for lane in self.lanes
    ]
    footer = indentation(indent) + "</edge>"
    return "\n".join([header] + children + [footer])

  def real_lane_index(self, lane_index: int) -> int:
    """
    Since SUMO is right-handed and CityFlow is left-handed i have to invert lane_index numbers in connections
    """
    return len(self.lanes) - 1 - lane_index

  def __repr__(self) -> str:
    return self.to_xml(0)

class InternalEdge:
  def __init__(self, id: str, lanes: list[Lane]) -> None:
    self.id: str = id
    self.lanes: list[Lane] = lanes

  def to_xml(self, indent: int = 0) -> str:
    header = (
      indentation(indent) +
      "<edge id=\"%s\" function=\"internal\">" % (
        self.id
      )
    )
    children = [
      lane.to_xml(indent + 1) for lane in self.lanes
    ]
    footer = indentation(indent) + "</edge>"
    return "\n".join([header] + children + [footer])

  def __repr__(self) -> str:
    return self.to_xml(0)

  @staticmethod
  def name(junction_id: str, connection_id: int) -> str:
    return "%s_%s" % (junction_id, connection_id)

class Junction:
  def __init__(self, id: str, kind: str, point: Point, incoming_lanes: list[str], into_lanes: list[str], requests: list[Request]) -> None:
    self.id: str = id
    self.kind = kind
    self.point = point
    self.incoming_lanes = incoming_lanes
    self.into_lanes = into_lanes
    self.requests: list[Request] = requests

  def to_xml(self, indent: int = 0) -> str:
    header = (
      indentation(indent) +
      "<junction id=\"%s\" type=\"%s\" x=\"%s\" y=\"%s\" incLanes=\"%s\" intLanes=\"%s\" >" % (
        self.id, self.kind, self.point.x, self.point.y,
        " ".join(self.incoming_lanes),
        " ".join(self.into_lanes)
      )
    )
    children = [
      request.to_xml(indent + 1)
      for request in self.requests
    ]
    footer = indentation(indent) + "</junction>"
    return "\n".join([header] + children + [footer])

  def __repr__(self) -> str:
    return self.to_xml(0)

class Request:
  def __init__(self, index: int, response: str, foes: str) -> None:
    self.index: int = index
    self.response: str = response
    self.foes: str = foes

  def to_xml(self, indent: int = 0) -> str:
    return (
        indentation(indent) +
        "<request index=\"%s\" response=\"%s\" foes=\"%s\" cont=\"0\"/>" % (
          self.index, self.response, self.foes
        ))

  def __repr__(self) -> str:
    return self.to_xml(0)

  @staticmethod
  def name(edge_id: str, lane_id: int) -> str:
    return "%s_%s" % (edge_id, lane_id)

class InternalConnection:
  def __init__(self, from_edge: str, to_edge: str, from_lane: int, to_lane: int, direction: str) -> None:
    self.from_edge: str = from_edge
    self.to_edge: str = to_edge
    self.from_lane: int = from_lane
    self.to_lane: int = to_lane
    self.direction: str = direction

  def to_xml(self, indent: int = 0) -> str:
    return (
      indentation(indent) +
      '<connection from="%s" to="%s" fromLane="%s" toLane="%s" dir="%s" state="M"/>' % (
        self.from_edge, self.to_edge, self.from_lane, self.to_lane, self.direction
      )
    )

  def __repr__(self) -> str:
    return self.to_xml(0)

class ViaConnection:
  def __init__(self, from_edge: str, to_edge: str, from_lane: int, to_lane: int, direction: str, index: int, via_junction_lane: str, junction_id: str) -> None:
    self.from_edge: str = from_edge
    self.to_edge: str = to_edge
    self.from_lane: int = from_lane
    self.to_lane: int = to_lane
    self.direction: str = direction
    self.index: int = index
    self.via_junction_lane: str = via_junction_lane
    self.junction_id: str = junction_id

  def to_xml(self, indent: int = 0) -> str:
    return (
      indentation(indent) +
      '<connection from="%s" to="%s" fromLane="%s" toLane="%s" dir="%s" state="M" linkIndex=\"%s\" via=\"%s\" tl=\"%s\"/>' % (
        self.from_edge, self.to_edge, self.from_lane, self.to_lane, self.direction, self.index, self.via_junction_lane, self.junction_id
      )
    )

  def __repr__(self) -> str:
    return self.to_xml(0)

class TLLogic:
  def __init__(self, id: str, phases: list[Phase]) -> None:
    self.id: str = id
    self.phases: list[Phase] = phases

  def to_xml(self, indent: int = 0) -> str:
    header = (
        indentation(indent) +
        "<tlLogic id=\"%s\" type=\"static\" programID=\"0\" offset=\"0\">" % (
          self.id
        ))
    children = [
      phase.to_xml(indent + 1)
      for phase in self.phases
    ]
    footer = indentation(indent) + "</tlLogic>"
    return "\n".join([header] + children + [footer])

  def __repr__(self) -> str:
    return self.to_xml(0)

class Phase:
  def __init__(self, duration: float, state: str) -> None:
    self.duration: float = duration
    self.state: str = state

  def to_xml(self, indent: int = 0) -> str:
    return (
        indentation(indent) +
        "<phase duration=\"%s\" state=\"%s\"/>" % (
          self.duration, self.state
        ))

  def __repr__(self) -> str:
    return self.to_xml(0)

  @staticmethod
  def name(edge_id: str, lane_id: int) -> str:
    return "%s_%s" % (edge_id, lane_id)

class Network:
  def __init__(self,
               road_edges: list[Edge],
               junctions: list[Junction],
               via_connections: list[ViaConnection],
               internal_connections: list[InternalConnection],
               junction_edges: list[InternalEdge],
               tllogics: list[TLLogic]) -> None:
    self.road_edges: list[Edge] = road_edges
    self.junctions: list[Junction] = junctions
    self.via_connections: list[ViaConnection] = via_connections
    self.internal_connections: list[InternalConnection] = internal_connections
    self.junction_edges: list[InternalEdge] = junction_edges
    self.tllogics: list[TLLogic] = tllogics

  def to_xml(self, indent: int = 0) -> str:
    lines = []
    lines.append(indentation(indent) + '<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(indentation(indent) + '<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">')
    for child in self.junction_edges:
      lines.append(child.to_xml(indent + 1))
    for child in self.road_edges:
      lines.append(child.to_xml(indent + 1))
    for child in self.tllogics:
      lines.append(child.to_xml(indent + 1))
    for child in self.junctions:
      lines.append(child.to_xml(indent + 1))
    for child in self.via_connections:
      lines.append(child.to_xml(indent + 1))
    for child in self.internal_connections:
      lines.append(child.to_xml(indent + 1))
    lines.append('</net>')
    return "\n".join(lines)

  def __repr__(self) -> str:
    return self.to_xml(0)

class Route:
  def __init__(self, id: str, edges: list[str]) -> None:
    self.id: str = id
    self.edges: list[str] = edges

  def to_xml(self, indent: int = 0) -> str:
    return (
      indentation(indent) + "<route id=\"%s\" edges=\"%s\"/>" % (
        self.id, " ".join(self.edges)
      )
    )

  @staticmethod
  def name(route_index: int) -> str:
    return "route_%s" % route_index

  def __repr__(self) -> str:
    return self.to_xml(0)

class Vehicle:
  def __init__(self, id: str, departure_time: float, route_id: str) -> None:
    self.id: str = id
    self.departure_time: float = departure_time
    self.route_id: str = route_id

  def to_xml(self, indent: int = 0) -> str:
    return (
      indentation(indent) + "<vehicle id=\"%s\" depart=\"%s\" route=\"%s\"/>" % (
        self.id, self.departure_time, self.route_id
      )
    )

  def __repr__(self) -> str:
    return self.to_xml(0)

  @staticmethod
  def name(vehicle_index: int) -> str:
    return "vehicle_%s" % vehicle_index

class Routes:
  def __init__(self, routes: list[Route], vehicles: list[Vehicle], flows: list[Flow]) -> None:
    self.routes: list[Route] = routes
    self.vehicles: list[Vehicle] = vehicles
    self.flows: list[Flow] = flows

  def to_xml(self, indent: int = 0) -> str:
    lines = []
    lines.append(indentation(indent) + '<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(indentation(indent) + '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    for route in self.routes:
      lines.append(route.to_xml(indent + 1))
    for vehicle in self.vehicles:
      lines.append(vehicle.to_xml(indent + 1))
    for flow in self.flows:
      lines.append(flow.to_xml(indent + 1))
    lines.append('</routes>')
    return "\n".join(lines)

  def __repr__(self) -> str:
    return self.to_xml(0)

class TAZ:
  def __init__(self, id: str, shape: list[Point], edges: list[str]) -> None:
    self.id = id
    self.shape = shape
    self.edges = edges

  def to_xml(self, indent: int = 0) -> str:
    lines = [
        indentation(indent) + '<taz id="%s" shape="%s" color="blue" edges="%s"/>' % (
          self.id,
          " ".join([point.to_xml() for point in self.shape] + [self.shape[0].to_xml()]),
          " ".join(self.edges)
        )
    ]
    return "\n".join(lines)

class Flow:
  def __init__(self, id, begin, end, fromTaz, toTaz, vehsPerHour) -> None:
    self.id = id
    self.begin = begin
    self.fromTaz = fromTaz
    self.toTaz = toTaz
    self.end = end
    self.vehsPerHour = vehsPerHour
    pass

  def to_xml(self, indent: int = 0) -> str:
    return indentation(indent) + '<flow id="%s" begin="%s" fromTaz="%s" toTaz="%s" end="%s" vehsPerHour="%s"/>' % (
      self.id, self.begin, self.fromTaz, self.toTaz, self.end, self.vehsPerHour
    )

class Additions:
  def __init__(self, tazs: list[TAZ]) -> None:
    self.tazs = tazs

  def to_xml(self, indent: int = 0) -> str:
    lines = [
      indentation(indent) + '<?xml version="1.0" encoding="UTF-8"?>',
      indentation(indent) + '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">'
    ]
    for taz in self.tazs:
      lines.append(taz.to_xml(indent + 1))
    lines += [
      indentation(indent) + '</additional>'
    ]
    return "\n".join(lines)

class Simulation:
  def __init__(self, network: Network, routes: Routes, additions: Additions) -> None:
    self.network: Network = network
    self.routes: Routes = routes
    self.additions: Additions = additions

  def to_xml(self, indent: int = 0) -> str:
    lines = []
    lines.append(indentation(indent) + '<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(indentation(indent) + '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">')
    lines.append(indentation(indent) + '    <input>')
    lines.append(indentation(indent) + '        <net-file value="network.net.xml"/>')
    lines.append(indentation(indent) + '        <route-files value="routes.rou.xml"/>')
    lines.append(indentation(indent) + '        <addition-files value="additions.add.xml"/>')
    lines.append(indentation(indent) + '    </input>')
    lines.append(indentation(indent) + '</configuration>')
    return "\n".join(lines)

  def __repr__(self) -> str:
    return self.to_xml(0)
