from __future__ import annotations
from sumo_rl.models.commons import Point

class LaneLink:
  def __init__(self, from_lane_idx: int, to_lane_idx: int) -> None:
    self.from_lane_idx: int = from_lane_idx
    self.to_lane_idx: int = to_lane_idx

  def to_dict(self) -> dict:
    return {
      "startLaneIndex": self.from_lane_idx,
      "endLaneIndex": self.to_lane_idx,
      "points": []
    }

class RoadLink:
  def __init__(self, from_road_ID: str, to_road_ID: str, lane_links: list[LaneLink]) -> None:
    self.from_road_ID = from_road_ID
    self.to_road_ID = to_road_ID
    self.lane_links = lane_links

  def to_dict(self) -> dict:
    return {
      "type": "go_straight",
      "startRoad": self.from_road_ID,
      "endRoad": self.to_road_ID,
      "direction": 0,
      "laneLinks": [
        lane_link.to_dict() for lane_link in self.lane_links
      ]
    }

class Intersection:
  def __init__(self, id: str, point: Point, road_IDs: list[str], road_links: list[RoadLink]):
    self.id: str = id
    self.point: Point = point
    self.road_IDs: list[str] = road_IDs
    self.road_links: list[RoadLink] = road_links

  def to_dict(self) -> dict:
    return {
        "id": self.id,
        "point": self.point.to_dict(),
        "width": 11,
        "roads": self.road_IDs,
        "roadLinks": [
          road_link.to_dict() for road_link in self.road_links
          ],
        "trafficLight": {
          "roadLinkIndices": [
            _ for _ in range(len(self.road_links))
            ],
          "lightphases": [
            {
              "time": 5,
              "availableRoadLinks": [
                _ for _ in range(len(self.road_links))
                ]
            } for _ in range(2)
          ]
        }, "virtual": False,
      }

class Road:
  def __init__(self, id: str, from_intersection_ID: str, to_intersection_ID: str, points: list[Point], number_of_lanes: int, max_speed: int):
    self.id: str = id
    self.from_intersection_ID: str = from_intersection_ID
    self.to_intersection_ID: str = to_intersection_ID
    self.points: list[Point] = points
    self.number_of_lanes: int = number_of_lanes
    self.max_speed: int = max_speed
  
  def to_dict(self) -> dict:
    return {
        "id": self.id,
        "points": [point.to_dict() for point in self.points],
        "lanes": [
          {"width": 3, "maxSpeed": self.max_speed} for _ in range(self.number_of_lanes)
        ], "startIntersection": self.from_intersection_ID,
        "endIntersection": self.to_intersection_ID
     }

class Network:
  def __init__(self, roads: list[Road], intersections: list[Intersection]) -> None:
    self.roads = roads
    self.intersections = intersections

  def to_dict(self) -> dict:
    return {
        'roads': [road.to_dict() for road in self.roads],
        'intersections': [intersection.to_dict() for intersection in self.intersections]
      }

