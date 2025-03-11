from __future__ import annotations
import math
import typing
import pickle
import time
import os
import os.path

class Point:
  def __init__(self, x: float, y: float) -> None:
    self.x: float = x
    self.y: float = y

  def distance(self, o: Point) -> float:
    return math.sqrt(math.pow(self.x - o.x, 2) + math.pow(self.y - o.y, 2))

  def to_dict(self) -> dict:
    return {'x': self.x, 'y': self.y}

  def to_str(self) -> str:
    return "Point(%s,%s)" % (self.x, self.y)

  def to_xml(self) -> str:
    return "%s,%s" % (self.x, self.y)

  def __repr__(self) -> str:
    return self.to_str()

def indentation(indent: int = 0):
  return "  " * indent

class Cache:
  def __init__(self):
    self.index: dict[str, typing.Any] = {}

  def path(self, ID) -> str:
    directory = ".cache"
    if not os.path.exists(directory):
      os.makedirs(directory)
    return os.path.join(directory, ID + ".pickle")

  def query(self, ID: str) -> typing.Any:
    if ID in self.index:
      return self.index[ID]
    path = self.path(ID)
    if os.path.exists(path):
      data: typing.Any
      with open(path, "rb") as file:
        data = pickle.load(file)
      self.index[ID] = data
      return data
    return None

  def store(self, ID: str, data: typing.Any):
    path = self.path(ID)
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
