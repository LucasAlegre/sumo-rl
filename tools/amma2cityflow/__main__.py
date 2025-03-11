from __future__ import annotations
import json
import sqlite3
import pandas
import matplotlib.pyplot
import re
import abc
import argparse
import sys
import os

from sumo_rl.models.citiflow import *

class DBReader(abc.ABC):
  def __init__(self, path: str) -> None:
    self.path = path
    self.connection: sqlite3.Connection|None = None
    self.cursor: sqlite3.Cursor|None = None

  def _is_open(self) -> bool:
    return (self.connection is not None) and (self.cursor is not None)

  def _is_closed(self) -> bool:
    return (self.connection is None) and (self.cursor is None)

  def open(self) -> None:
    assert self._is_closed()
    self.connection = sqlite3.connect(self.path)
    self.cursor = self.connection.cursor()

  def close(self) -> None:
    assert self._is_open()
    self.cursor = None
    self.connection.close()
    self.connection = None

  def __enter__(self) -> DBReader:
    self.open()
    return self

  def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
    self.close()

  def identify_table(self):
    assert self._is_open()
    table = self.cursor.execute('select * from sqlite_master limit 1').fetchone()
    constructor = table[-1]
    regex = re.compile(r'\[[^\[\]]+\]')
    matches = regex.findall(constructor)
    return matches[0][1:-1], [_[1:-1] for _ in matches[1:]]

  @abc.abstractmethod
  def read(self) -> pandas.DataFrame:
    """Read data to df"""

class NodesDBReader(DBReader):
  @staticmethod
  def Read(path: str) -> pandas.DataFrame:
    with __class__(path) as dbreader:
      return dbreader.read()

  def read(self) -> pandas.DataFrame:
    assert self._is_open()
    table_name, table_fields = self.identify_table()
    rows = self.cursor.execute('select * from %s' % table_name).fetchall()
    return pandas.DataFrame(data=rows, columns=table_fields)

class EdgesDBReader(DBReader):
  @staticmethod
  def Read(path: str) -> pandas.DataFrame:
    with __class__(path) as dbreader:
      return dbreader.read()

  def read(self) -> pandas.DataFrame:
    assert self._is_open()
    table_name, table_fields = self.identify_table()
    rows = self.cursor.execute('select * from %s' % table_name).fetchall()
    return pandas.DataFrame(data=rows, columns=table_fields)

def linktype_id_to_name(id: int) -> str:
  map = {1: 'autostrade', 2: 'strade primarie', 3: 'strade secondarie', 4: 'strade locali'}
  return map[id]

def linktype_id_to_number_of_lanes(id: int) -> int:
  map = {1: 4, 2: 3, 3: 2, 4: 1}
  return map[id]

def curve_of_capacity_over_distance(edges_df: pandas.DataFrame):
  figure = matplotlib.pyplot.figure(figsize=(20,10))
  
  min_linktype, max_linktype = min(edges_df['linktype']), max(edges_df['linktype'])
  for linktype_id in range(min_linktype, max_linktype + 1):
    closure = edges_df.where(edges_df.linktype.eq(linktype_id))
    matplotlib.pyplot.scatter(x=closure['distance'], y=closure['capacity'], label=linktype_id_to_name(linktype_id))
  matplotlib.pyplot.title('curve_of_capacity_over_distance')
  matplotlib.pyplot.legend()
  matplotlib.pyplot.savefig('curve_of_capacity_over_distance.png')
  matplotlib.pyplot.close()

def transform(nodes_df: pandas.DataFrame, edges_df: pandas.DataFrame):
  node_index: dict[str, dict] = {}
  edge_index: dict[str, dict] = {}

  for _, row in nodes_df.iterrows():
    ID = 'N' + str(int(row['n']))
    node_index[ID] = {
      'id': ID,
      'point': Point(x = row['x'],
                     y = row['y']),
      'edges': set({})
    }

  for _, row in edges_df.iterrows():
    ID = 'E' + str(int(row['id']))
    from_ID = 'N' + str(int(row['a']))
    to_ID = 'N' + str(int(row['b']))
    edge_index[ID] = {
      'id': ID,
      'from': from_ID,
      'to': to_ID,
      'number_of_lanes': linktype_id_to_number_of_lanes(row['linktype']),
      'max_speed': row['speed']
    }
    node_index[from_ID]['edges'].add(ID)
    node_index[to_ID]['edges'].add(ID)

  road_index: dict[str, Road] = {}
  intersection_index: dict[str, Intersection] = {}
  
  for edge_ID in edge_index:
    edge = edge_index[edge_ID]
    from_node = node_index[edge['from']]
    to_node = node_index[edge['to']]
    road = Road(
      id=edge_ID,
      from_intersection_ID=from_node['id'],
      to_intersection_ID=to_node['id'],
      points=[from_node['point'], to_node['point']],
      number_of_lanes=edge['number_of_lanes'],
      max_speed=edge['max_speed']
    )
    road_index[road.id] = road

  for node_ID in node_index:
    node = node_index[node_ID]
    road_links = []
    for edge_AID in node['edges']:
      edge_A = edge_index[edge_AID]
      for edge_BID in node['edges']:
        edge_B = edge_index[edge_BID]
        if edge_AID != edge_BID:
          lane_links = []
          for lane_AIDX in range(edge_A['number_of_lanes']):
            lane_BIDX = lane_AIDX
            if lane_BIDX >= edge_B['number_of_lanes']:
              lane_BIDX = edge_B['number_of_lanes'] - 1
            lane_links.append(LaneLink(lane_AIDX, lane_BIDX))
          road_link = RoadLink(
            from_road_ID=edge_AID,
            to_road_ID=edge_BID,
            lane_links=lane_links
          )
          road_links.append(road_link)
    intersection = Intersection(
        id=node_ID,
        point=node['point'],
        road_IDs=list(node['edges']),
        road_links=road_links
    )
    intersection_index[intersection.id] = intersection
  network = Network(list(road_index.values()), list(intersection_index.values()))
  return network

if __name__ == "__main__":
  cli = argparse.ArgumentParser(description="Converts data from Agenzia Milanese Mobilita' e Ambiente into Citiflow format, so that it can be translated back to SUMO with citiflow2sumo")
  cli.add_argument('nodes', type=str, help='nodes file in sqlite3 format')
  cli.add_argument('edges', type=str, help='edges file in sqlite3 format')
  cli.add_argument('-o', '--output', type=str, default='.', help='output directory (default = .)')
  cli_args = cli.parse_args(sys.argv[1:])

  if not os.path.exists(cli_args.output):
    os.makedirs(cli_args.output)

  nodes_df = NodesDBReader.Read(cli_args.nodes)
  edges_df = EdgesDBReader.Read(cli_args.edges)

  network = transform(nodes_df, edges_df)
  with open(os.path.join(cli_args.output, "network.json"), "w") as network_file:
    json.dump(network.to_dict(), network_file)
