from __future__ import annotations
import sqlite3
import pickle
import time
import typing
import pandas
import re
import math
import argparse
import sys
import os
import json
import shapefile
import matplotlib.pyplot

class Cache:
  def __init__(self):
    self.index: dict[str, typing.Any] = {}

  def query(self, ID: str) -> typing.Any:
    if ID in self.index:
      return self.index[ID]
    path = (ID + ".cache.pickle")
    if os.path.exists(path):
      data: typing.Any
      with open(path, "rb") as file:
        data = pickle.load(file)
      self.index[ID] = data
      return data
    return None

  def store(self, ID: str, data: typing.Any):
    path = (ID + ".cache.pickle")
    with open(path, "wb") as file:
      pickle.dump(data, file)
    self.index[ID] = data

class Timer:
  def __init__(self, indent: int = 0) -> None:
    self.clock = time.time()
    self.indent = indent

  def branch(self) -> Timer:
    return Timer(self.indent + 1)

  def round(self, msg: str):
    nclock = time.time()
    diff = nclock - self.clock
    print("%s| %s | Elapsed %s ms" % ("  " * self.indent, msg, diff))
    self.clock = time.time()

  def clear(self):
    self.clock = time.time()

class Point:
  def __init__(self, x: float, y: float) -> None:
    self.x: float = x
    self.y: float = y

  def distance(self, o: Point) -> float:
    return math.sqrt(math.pow(self.x - o.x, 2) + math.pow(self.y - o.y, 2))

  def to_dict(self) -> dict:
    return {'x': self.x, 'y': self.y}

  def to_xml(self, indent: int = 0) -> str:
    return "  " * indent + "%s,%s" % (self.x, self.y)

class JSONReader:
  def __init__(self, path: str) -> None:
    self.path = path
    self.file = None

  def _is_open(self) -> bool:
    return (self.file is not None)

  def _is_closed(self) -> bool:
    return (self.file is None)

  def open(self) -> None:
    assert self._is_closed()
    self.file = open(self.path, "r")

  def close(self) -> None:
    assert self._is_open()
    self.file.close()
    self.file = None

  def __enter__(self) -> DBReader:
    self.open()
    return self

  def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
    self.close()

  @staticmethod
  def Read(path: str) -> dict:
    with __class__(path) as dbreader:
      return dbreader.read()

  def read(self) -> dict:
    assert self._is_open()
    return json.load(self.file)

class DBReader:
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

  @staticmethod
  def Read(path: str) -> pandas.DataFrame:
    with __class__(path) as dbreader:
      return dbreader.read()

  def read(self) -> pandas.DataFrame:
    assert self._is_open()
    table_name, table_fields = self.identify_table()
    rows = self.cursor.execute('select * from %s' % table_name).fetchall()
    return pandas.DataFrame(data=rows, columns=table_fields)

class SHPReader:
  def __init__(self, path: str) -> None:
    self.path = path
    self.reader: shapefile.Reader|None = None

  def _is_open(self) -> bool:
    return (self.reader is not None)

  def _is_closed(self) -> bool:
    return (self.reader is None)

  def open(self) -> None:
    assert self._is_closed()
    self.reader = shapefile.Reader(self.path)

  def close(self) -> None:
    assert self._is_open()
    self.reader.close()
    self.reader = None

  def __enter__(self) -> DBReader:
    self.open()
    return self

  def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
    self.close()

  @staticmethod
  def Read(path: str) -> None:
    with __class__(path) as dbreader:
      return dbreader.read()

  def read(self) -> None:
    assert self._is_open()
    columns = [_[0] for _ in self.reader.fields[1:]]
    records = []

    for record in self.reader.records():
      obj = {}
      for key, value in zip(columns, record):
        obj[key] = value
      records.append(obj)

    for index, shape in enumerate(self.reader.shapes()):
      records[index]['shape'] = shape.bbox

    return pandas.DataFrame(records)

class TAZ:
  def __init__(self, id: str, shape: list[Point], edges: list[str]) -> None:
    self.id = id
    self.shape = shape
    self.edges = edges

  def to_xml(self, indent: int = 0) -> str:
    lines = [
        "  " * indent + '<taz id="%s" shape="%s" color="blue" edges="%s">' % (
          self.id,
          " ".join([point.to_xml() for point in self.shape] + [self.shape[0].to_xml()]),
          " ".join(self.edges)
        ),
        "  " * indent + '</taz>'
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
    return '<flow id="%s" begin="%s" fromTaz="%s" toTaz="%s" end="%s" vehsPerHour="%s"/>' % (
      self.id, self.begin, self.fromTaz, self.toTaz, self.end, self.vehsPerHour
    )

class Additions:
  def __init__(self, tazs: list[TAZ]) -> None:
    self.tazs = tazs

  def to_xml(self, indent: int = 0) -> str:
    lines = [
      "  " * indent + '<?xml version="1.0" encoding="UTF-8"?>',
      "  " * indent + '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">'
    ]
    for taz in self.tazs:
      lines.append(taz.to_xml(indent + 1))
    lines += [
      "  " * indent + '</additional>'
    ]
    return "\n".join(lines)

class Routes:
  def __init__(self, flows: list[Flow]) -> None:
    self.flows = flows

  def to_xml(self, indent: int = 0) -> str:
    lines = [
      "  " * indent + '<?xml version="1.0" encoding="UTF-8"?>',
      "  " * indent + '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">'
    ]
    for flow in self.flows:
      lines.append(flow.to_xml(indent + 1))
    lines += [
      "  " * indent + '</routes>'
    ]
    return "\n".join(lines)

def compute_box(shape: tuple[float, float, float, float]) -> tuple[Point, Point, Point, Point]:
  Ax, Ay, Cx, Cy = shape
  assert Ax < Cx
  assert Ay < Cy
  Bx, By = Cx, Ay
  Dx, Dy = Ax, Cy

  return (
    Point(Ax, Ay),
    Point(Bx, By),
    Point(Cx, Cy),
    Point(Dx, Dy),
  )

def plot_zones(zones_index: dict[str, dict]):
  matplotlib.pyplot.figure(figsize=(30,30))
  for zone in zones_index.values():
    shape = zone['shape']
    Xs = [_.x for _ in shape]
    Ys = [_.y for _ in shape]
    matplotlib.pyplot.plot(Xs, Ys, marker='o')
  matplotlib.pyplot.savefig('zones.png')
  matplotlib.pyplot.close()

def plot_roads(roads_index: dict[str, dict]):
  matplotlib.pyplot.figure(figsize=(30,30))
  for road in roads_index.values():
    shape = road['points']
    Xs = [_.x for _ in shape]
    Ys = [_.y for _ in shape]
    matplotlib.pyplot.plot(Xs, Ys)
  matplotlib.pyplot.savefig('roads.png')
  matplotlib.pyplot.close()

def plot_zones_and_roads(zones_index: dict[str, dict], roads_index: dict[str, dict]):
  matplotlib.pyplot.figure(figsize=(100,100))
  for zone in zones_index.values():
    shape = zone['shape']
    Xs = [_.x for _ in shape] + [shape[0].x]
    Ys = [_.y for _ in shape] + [shape[0].y]
    matplotlib.pyplot.plot(Xs, Ys, marker='o')
  for road in roads_index.values():
    shape = road['points']
    Xs = [_.x for _ in shape]
    Ys = [_.y for _ in shape]
    matplotlib.pyplot.plot(Xs, Ys, color='black')
  matplotlib.pyplot.savefig('AMMA.png')
  matplotlib.pyplot.close()

def is_point_inside_rect(p, rect):
    """Check if point p is inside the rectangle defined by two opposite corners."""
    (p1, p2) = rect  # Assume p1 is bottom-left, p2 is top-right
    return p1.x <= p.x <= p2.x and p1.y <= p.y <= p2.y

def cross_product(a, b):
    """Compute the cross product of vectors OA and OB (used for orientation tests)."""
    return a.x * b.y - a.y * b.x

def direction(p1, p2, p3):
    """Find the orientation of the triplet (p1, p2, p3)."""
    return cross_product(Point(p3.x - p1.x, p3.y - p1.y), Point(p2.x - p1.x, p2.y - p1.y))

def segments_intersect(seg1, seg2):
    """Check if two segments intersect using orientation and cross product tests."""
    p1, q1 = seg1
    p2, q2 = seg2

    d1 = direction(p2, q2, p1)
    d2 = direction(p2, q2, q1)
    d3 = direction(p1, q1, p2)
    d4 = direction(p1, q1, q2)

    if d1 * d2 < 0 and d3 * d4 < 0:
        return True  # Proper intersection

    return False  # No intersection

def segment_in_rectangle(segment, rect):
    """Check if segment is inside or intersects the rectangle."""
    p1, p2 = segment
    rect_p1, rect_p2 = rect  # Bottom-left and top-right

    # Step 1: Check if both points are inside the rectangle
    if is_point_inside_rect(p1, rect) and is_point_inside_rect(p2, rect):
        return True  # Fully contained

    # Step 2: Define rectangle edges as segments
    rect_edges = [
        (rect_p1, Point(rect_p2.x, rect_p1.y)),  # Bottom edge
        (Point(rect_p2.x, rect_p1.y), rect_p2),  # Right edge
        (rect_p2, Point(rect_p1.x, rect_p2.y)),  # Top edge
        (Point(rect_p1.x, rect_p2.y), rect_p1)   # Left edge
    ]

    # Step 3: Check if segment intersects any rectangle edge
    for edge in rect_edges:
        if segments_intersect(segment, edge):
            return True

    return False  # Neither inside nor intersecting

def intersects(road_points: tuple[Point, Point],
               zone_shape: tuple[Point, Point, Point, Point]) -> bool:
  road_A, road_B = road_points
  zone_A, _, zone_C, _ = zone_shape
  return segment_in_rectangle((road_A, road_B), (zone_A, zone_C))

def transform(timer: Timer, od_matrix_dfs: list[pandas.DataFrame], zones_df: pandas.DataFrame, network: dict):
  cache = Cache()
  zones_index: dict[str, dict] = {}
  if cache.query("zones_index") is None:
    for _, row in zones_df.iterrows():
      ID = 'Z' + str(int(row['Z_CUBE']))
      zones_index[ID] = {'id': ID, 'shape': compute_box(row['shape'])}
    timer.round("Initialized zones_index")
    cache.store("zones_index", zones_index)
  else:
    timer.round("Load zones_index")
    zones_index = cache.query("zones_index")

  roads_index: dict[str, dict] = {}
  if cache.query("roads_index") is None:
    for road in network['roads']:
      roads_index[road['id']] = {'id': road['id'], 'points': (
        Point(road['points'][0]['x'], road['points'][0]['y']),
        Point(road['points'][1]['x'], road['points'][1]['y'])
      )}
    timer.round("Initialized road_index")
    cache.store("roads_index", roads_index)
  else:
    timer.round("Loaded road_index")
    roads_index = cache.query("roads_index")

  timer2 = timer.branch()
  for zone_ID in zones_index:
    included_roads = []
    shape = zones_index[zone_ID]['shape']
    if cache.query('roads-' + zone_ID) is None:
      for road in roads_index.values():
        if intersects(road['points'], shape):
          included_roads.append(road['id'])
      cache.store('roads-' + zone_ID, included_roads)
      timer2.round("Computed for zone %s" % zone_ID)
    else:
      included_roads = cache.query('roads-' + zone_ID)
      timer2.round("Loaded for zone %s" % zone_ID)
    zones_index[zone_ID]['roads'] = included_roads
  timer.round("Built zone<->road join")

  tazs = []
  if cache.query("tazs") is None:
    for zone in zones_index.values():
      tazs.append(TAZ(zone['id'], zone['shape'], (zone.get('roads') or ['E8173'])))
    cache.store("tazs", tazs)
    timer.round("Built TAZs")
  else:
    tazs = cache.query("tazs")
    timer.round("Loaded TAZs")
  additions = Additions(tazs)

  with open("zones.add.xml", "w") as file:
    file.write(additions.to_xml())
  timer.round("Wrote zones.add.xml")

  od_matrix_idx = 0
  for od_matrix_df in od_matrix_dfs:
    flows: list[Flow] = []
    flow_idx = 0
    print("vehsPerHour", sum(od_matrix_df['veq_priv']))
    for _, row in od_matrix_df.iterrows():
      orig_ID = 'Z' + str(int(row['orig_urb']))
      dest_ID = 'Z' + str(int(row['dest_urb']))
      if orig_ID in zones_index and dest_ID in zones_index:
        flows.append(Flow('F' + str(flow_idx), 0, 3600, orig_ID, dest_ID, row['veq_priv']))
        flow_idx += 1
    routes = Routes(flows)
    od_matrix_idx += 1
    timer.round("Built od_index %s" % od_matrix_idx)
    with open("routes.%s.rou.xml" % od_matrix_idx, "w") as file:
      file.write(routes.to_xml())
    timer.round("Wrote routes.%s.rou.xml" % od_matrix_idx)

if __name__ == "__main__":
  cli = argparse.ArgumentParser(description="Converts data from Agenzia Milanese Mobilita' e Ambiente into Citiflow format, so that it can be translated back to SUMO with citiflow2sumo")
  cli.add_argument('mat_od_07_10', type=str, help='OD Matrix for 07-10 Traffic in sqlite3 format')
  cli.add_argument('mat_od_10_16', type=str, help='OD Matrix for 10-16 Traffic in sqlite3 format')
  cli.add_argument('mat_od_16_20', type=str, help='OD Matrix for 16-20 Traffic in sqlite3 format')
  cli.add_argument('zones', type=str, help='zones file in SHP format')
  cli.add_argument('network', type=str, help='network file in Cityflow JSON format')
  cli.add_argument('-o', '--output', type=str, default='.', help='output directory (default = .)')
  cli_args = cli.parse_args(sys.argv[1:])

  if not os.path.exists(cli_args.output):
    os.makedirs(cli_args.output)

  timer = Timer()
  od_matrix_dfs = [DBReader.Read(cli_args.mat_od_07_10),
                   DBReader.Read(cli_args.mat_od_10_16),
                   DBReader.Read(cli_args.mat_od_16_20)]
  timer.round("Loaded OD Matrices")
  zones_df = SHPReader.Read(cli_args.zones)
  timer.round("Loaded Zones")
  network = JSONReader.Read(cli_args.network)
  timer.round("Loaded Network")

  transform(timer.branch(), od_matrix_dfs, zones_df, network)
  timer.round("Transformed")
