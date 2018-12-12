import traci


class TrafficSignal:

    NS = 0
    EW = 3
    min_green = 10

    def __init__(self, ts_id):
        self.id = ts_id
        self.time_on_phase = 0
        self.edges = self._compute_edges()
        self.ns_stopped = [0, 0]
        self.ew_stopped = [0, 0]
        phases = [
            traci.trafficlight.Phase(35000, 35000, 35000, "GGGrrr"),   # north-south -> 0
            traci.trafficlight.Phase(2000, 2000, 2000, "yyyrrr"),
            traci.trafficlight.Phase(1, 1, 1, "rrrrrr"),
            traci.trafficlight.Phase(35000, 35000, 35000, "rrrGGG"),   # east-west -> 3
            traci.trafficlight.Phase(2000, 2000, 2000, "rrryyy"),
            traci.trafficlight.Phase(1, 1, 1, "rrrrrr")
        ]
        logic = traci.trafficlight.Logic("new-program", 0, 0, 0, phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def keep(self, time_keep):
        self.time_on_phase += time_keep
        traci.trafficlight.setPhaseDuration(self.id, time_keep)

    def change(self):
        if self.time_on_phase < self.min_green:  # min green time => do not change
            self.time_on_phase += 5
        else:
            self.time_on_phase = 3  # delta time - yellow time
            traci.trafficlight.setPhaseDuration(self.id, 0)

    def _compute_edges(self):
        """
        :return: Dict green phase to edge id
        """
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # remove duplicates and keep order
        return {self.NS: lanes[:2], self.EW: lanes[2:]}  # two lanes per edge

    def get_occupancy(self):
        ns_occupancy = sum([traci.lane.getLastStepOccupancy(lane) for lane in self.edges[self.NS]]) / len(self.edges[self.NS])
        ew_occupancy = sum([traci.lane.getLastStepOccupancy(lane) for lane in self.edges[self.EW]]) / len(self.edges[self.EW])
        return ns_occupancy, ew_occupancy

    def get_stopped_vehicles_num(self):
        self.ns_stopped[1], self.ew_stopped[1] = self.ns_stopped[0], self.ew_stopped[0]
        self.ns_stopped[0] = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.NS]])
        self.ew_stopped[0] = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.EW]])

        return self.ns_stopped[0], self.ew_stopped[0]
