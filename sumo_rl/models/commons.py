from __future__ import annotations
import math

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
