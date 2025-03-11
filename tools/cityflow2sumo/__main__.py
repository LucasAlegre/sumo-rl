"""
Tools::CityFlow2SUMO

Converts RoadNet and FlowNet format json files to SUMO xml files
"""

from __future__ import annotations
import json
import hashlib
import argparse
import sys
import os
from sumo_rl.models.sumo import *

# UTILITY

def load_network_json(path: str) -> dict:
  with open(path, "r") as file:
    return json.load(file)

def load_routes_json(path: str) -> list:
  with open(path, "r") as file:
    return json.load(file)

# CONVERSION

def translate_road(json_road: dict) -> Edge:
  points = [Point(p['x'], p['y']) for p in json_road['points']]
  length = sum([points[i].distance(points[i - 1]) for i in range(1, len(points))])
  lanes = [
    Lane(
      id=Lane.name(json_road['id'], json_lane_idx),
      index=json_lane_idx,
      speed=json_lane['maxSpeed'],
      length=length
    )
    for json_lane_idx, json_lane in enumerate(json_road['lanes'])
  ]
  return Edge(
    id=json_road['id'],
    from_junction=json_road['startIntersection'],
    to_junction=json_road['endIntersection'],
    shape=points,
    lanes=lanes,
  )

def translate_direction(json_direction: str) -> str:
  directions = {
    'go_straight': 's',
    'turn_left': 'l',
    'turn_right': 'r'
  }
  return directions[json_direction]

def simplify_phases(phases: list[Phase]) -> list[Phase]:
  new_phases = [phases[0]]
  for phase in phases[1:]:
    if phase.state == new_phases[-1].state:
      new_phases[-1].duration += phase.duration
    else:
      new_phases.append(phase)
  return new_phases

def translate_tl_intersection(json_intersection: dict, edge_map: dict[str, Edge]) -> tuple[Junction, list[ViaConnection], list[InternalConnection], list[InternalEdge], TLLogic]:
  junction_id = json_intersection['id']
  incoming_lanes = {}
  into_lanes = {}
  via_connections: list[ViaConnection] = []
  internal_connections: list[InternalConnection] = []
  roadToLaneLinks: dict[int, list[int]] = {}
  junction_edges: list[InternalEdge] = []

  for roadLink_index, roadLink in enumerate(json_intersection['roadLinks']):
    from_edge = roadLink['startRoad']
    to_edge = roadLink['endRoad']
    direction = translate_direction(roadLink['type'])
    roadToLaneLinks[roadLink_index] = []
    for laneLink in roadLink['laneLinks']:
      from_lane = laneLink['startLaneIndex']
      to_lane = laneLink['endLaneIndex']
      index = len(via_connections)
      edge_name = InternalEdge.name(junction_id, index)
      lane_name = Lane.name(edge_name, 0)
      
      via_connection = ViaConnection(from_edge, to_edge, edge_map[from_edge].real_lane_index(from_lane), edge_map[to_edge].real_lane_index(to_lane), direction, index, lane_name, junction_id)
      internal_connection = InternalConnection(edge_name, to_edge, 0, edge_map[to_edge].real_lane_index(to_lane), direction)
      junction_lane = Lane(id=lane_name, index=0, speed=3.93, length=2.19)
      junction_edge = InternalEdge(id=edge_name, lanes=[junction_lane])

      roadToLaneLinks[roadLink_index].append(index)
      incoming_lanes[Lane.name(via_connection.from_edge, via_connection.from_lane)] = 0
      into_lanes[Lane.name(via_connection.to_edge, via_connection.to_lane)] = 0
      via_connections.append(via_connection)
      internal_connections.append(internal_connection)
      junction_edges.append(junction_edge)
  n_of_via_connections = len(via_connections)

  phases: list[Phase] = []
  previous_phase_state: str|None = None
  for tl_phase in json_intersection['trafficLight']['lightphases']:
    gyr_map = {c:'r' for c in range(n_of_via_connections)}
    for greenRoadLink_index in tl_phase["availableRoadLinks"]:
      for greenLaneLink_index in roadToLaneLinks[greenRoadLink_index]:
        if previous_phase_state is None or previous_phase_state[greenLaneLink_index] == 'G':
          gyr_map[greenLaneLink_index] = 'G'
        else:
          gyr_map[greenLaneLink_index] = 'g'
    green_phase_state = "".join([gyr_map[c] for c in range(n_of_via_connections)])
    green_phase_duration = tl_phase["time"]
    phases.append(Phase(duration=green_phase_duration, state=green_phase_state))

    yellow_phase_state = green_phase_state.replace('g', 'y')
    yellow_phase_duration = 3.0
    phases.append(Phase(duration=yellow_phase_duration, state=yellow_phase_state))
    previous_phase_state = green_phase_state

  phases = simplify_phases(phases)
  if len(phases) == 1:
    green_phase_1 = phases[0]
    yellow_phase_1 = Phase(duration=3, state=green_phase_1.state.replace('G', 'y').replace('g', 'y'))
    green_phase_2 = Phase(duration=green_phase_1.duration, state=yellow_phase_1.state.replace('r', 'G').replace('y', 'r'))
    yellow_phase_2 = Phase(duration=3, state=green_phase_2.state.replace('G', 'y').replace('g', 'y'))
    phases = [green_phase_1, yellow_phase_1, green_phase_2, yellow_phase_2]
    phases = simplify_phases(phases)
    # If it has a single green phase, then i'll duplicate it to avoid problems with sumo_rl
    if len([0 for phase in phases if ('g' in phase.state or 'G' in phase.state)]) < 2:
      phases = phases + phases
  tllogic = TLLogic(id=junction_id, phases=phases)

  junction = Junction(
    id=junction_id,
    kind='traffic_light',
    point=Point(json_intersection['point']['x'], json_intersection['point']['y']),
    incoming_lanes=list(incoming_lanes),
    into_lanes=list(into_lanes),
    requests=[],
  )

  return (junction, via_connections, internal_connections, junction_edges, tllogic)

def translate_virtual_intersection(json_intersection: dict, incoming_map: dict[str, list[str]], into_map: dict[str, list[str]]) -> tuple[Junction]:
  junction_id = json_intersection['id']
  incoming_lanes = (incoming_map.get(junction_id) or [])
  into_lanes = (into_map.get(junction_id) or [])

  junction = Junction(
    id=junction_id,
    kind='dead_end',
    point=Point(json_intersection['point']['x'], json_intersection['point']['y']),
    incoming_lanes=incoming_lanes,
    into_lanes=into_lanes,
    requests=[],
  )

  return (junction,)

def map_of_edges(edges: list[Edge]) -> dict[str, Edge]:
  return {edge.id:edge for edge in edges}

def map_incoming_into_junction_edges(edges: list[Edge]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
  raw_incoming_map: dict[str, dict[str, int]] = {}
  raw_into_map: dict[str, dict[str, int]] = {}

  for edge in edges:
    if edge.to_junction not in raw_incoming_map:
      raw_incoming_map[edge.to_junction] = {}
    if edge.from_junction not in raw_into_map:
      raw_into_map[edge.from_junction] = {}
    for lane in edge.lanes:
      raw_incoming_map[edge.to_junction][lane.id] = 0
      raw_into_map[edge.from_junction][lane.id] = 0

  incoming_map: dict[str, list[str]] = {junction:list(raw_incoming_map[junction].keys()) for junction in raw_incoming_map}
  into_map: dict[str, list[str]] = {junction:list(raw_into_map[junction].keys()) for junction in raw_into_map}

  return incoming_map, into_map

def map_of_adiacency_of_edges(via_connections: list[ViaConnection]) -> dict[str, dict[str, bool]]:
  adiacency: dict[str, dict[str, bool]] = {}
  for via_connection in via_connections:
    source, target = via_connection.from_edge, via_connection.to_edge
    if source not in adiacency:
      adiacency[source] = {}
    adiacency[source][target] = True
  return adiacency

def translate_network(json_network: dict) -> Network:
  json_roads = json_network['roads']
  json_intersections = json_network['intersections']

  road_edges = [translate_road(json_road) for json_road in json_roads]
  edge_map = map_of_edges(road_edges)
  junction_incoming_map, junction_into_map = map_incoming_into_junction_edges(road_edges)

  junctions = []
  via_connections = []
  internal_connections = []
  junction_edges = []
  tllogics = []
  for json_intersection in json_intersections:
    if json_intersection['virtual']:
      _junction, = translate_virtual_intersection(json_intersection, junction_incoming_map, junction_into_map)
      junctions.append(_junction)
    else:
      _junction, _via_connections, _internal_connections, _junction_edges, tllogic = translate_tl_intersection(json_intersection, edge_map)
      junctions.append(_junction)
      internal_connections += _internal_connections
      via_connections += _via_connections
      junction_edges += _junction_edges
      tllogics.append(tllogic)

  return Network(road_edges, junctions, via_connections, internal_connections, junction_edges, tllogics)

def valid_route(route: list[str], adiacency_map: dict[str, dict[str, bool]]) -> bool:
  for i in range(1, len(route)):
    source, target = route[i - 1], route[i]
    if source not in adiacency_map or target not in adiacency_map[source] or not adiacency_map[source][target]:
      return False
  return True

def fix_route(route: list[str], adiacency_map: dict[str, dict[str, bool]]) -> None:
  pass

def translate_routes(json_routes: list, network: Network) -> Routes:
  adiacency_map: dict[str, dict[str, bool]] = map_of_adiacency_of_edges(network.via_connections)
  # print(json.dumps(adiacency_map))

  raw_routes: dict[str, Route] = {}
  route_validity: dict[str, bool] = {}
  vehicles: list[Vehicle] = []

  for json_route in json_routes:
    route_hash = hashlib.sha256("/".join(json_route['route']).encode()).digest().hex()
    if route_hash in route_validity and not route_validity[route_hash]:
      print("WARNING", "Skipping vehicle", json_route, "since it uses the reclaimed route", route_hash)
      continue
    if route_hash not in raw_routes:
      # Check Route
      edges = json_route['route']
      if not valid_route(edges, adiacency_map):
        route_validity[route_hash] = False
        print("WARNING", "Skipping route", edges, "since it is broken")
        continue
      # Add Route
      route_index = len(raw_routes)
      route_id = Route.name(route_index)
      route = Route(id=route_id, edges=edges)
      raw_routes[route_hash] = route
      route_validity[route_hash] = True
    route = raw_routes[route_hash]
    # Add Vehicle
    vehicle_index = len(vehicles)
    vehicle_id = Vehicle.name(vehicle_index)
    vehicle = Vehicle(id=vehicle_id, departure_time=json_route['startTime'], route_id=route.id)
    vehicles.append(vehicle)

  routes = list(raw_routes.values())
  return Routes(routes=routes, vehicles=vehicles)

if __name__ == "__main__":
  argument_parser = argparse.ArgumentParser("Cityflow2SUMO", description="Converts CityFlow RoadNet/FlowNet format to SUMO XML files")
  argument_parser.add_argument("network_file", type=str, help="Input network file in JSON CityFlow format")
  argument_parser.add_argument("routes_file", type=str, help="Input routes file in JSON CityFlow format")
  argument_parser.add_argument("-o", "--output", type=str, default="./output", help="Output directory for SUMO project")
  cli_args = argument_parser.parse_args(sys.argv[1:])

  print("Using network=%s AND routes=%s" % (cli_args.network_file, cli_args.routes_file))

  json_network = load_network_json(cli_args.network_file)
  json_routes = load_routes_json(cli_args.routes_file)

  network: Network = translate_network(json_network)
  routes: Routes = translate_routes(json_routes, network)
  additions = Additions([])
  simulation: Simulation = Simulation(network, routes, additions)

  if not os.path.exists(cli_args.output):
    os.makedirs(cli_args.output)

  with open("%s/network.net.xml" % (cli_args.output,), "w") as file:
    file.write(network.to_xml())
  with open("%s/routes.rou.xml" % (cli_args.output,), "w") as file:
    file.write(routes.to_xml())
  with open("%s/simulation.sumocfg" % (cli_args.output,), "w") as file:
    file.write(simulation.to_xml())
