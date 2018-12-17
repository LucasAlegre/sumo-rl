import traci


class TrafficSignal:

    NS = 0
    EW = 2

    def __init__(self, ts_id, delta_time, min_green, custom_phases):
        self.id = ts_id
        self.time_on_phase = 0
        self.delta_time = delta_time
        self.min_green = min_green
        self.edges = self._compute_edges()
        self.edges_capacity = self._compute_edges_capacity()
        if custom_phases is not None:
            logic = traci.trafficlight.Logic("new-program", 0, 0, 0, custom_phases)
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def keep(self):
        self.time_on_phase += self.delta_time
        traci.trafficlight.setPhaseDuration(self.id, self.delta_time)

    def change(self):
        if self.time_on_phase < self.min_green:  # min green time => do not change
            self.keep()
        else:
            self.time_on_phase = self.delta_time
            traci.trafficlight.setPhaseDuration(self.id, 0)

    def _compute_edges(self):
        """
        :return: Dict green phase to edge id
        """
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # remove duplicates and keep order
        return {self.NS: lanes[:2], self.EW: lanes[2:]}  # two lanes per edge

    def _compute_edges_capacity(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return {
            self.NS: sum([traci.lane.getLength(lane) for lane in self.edges[self.NS]]) / vehicle_size_min_gap,
            self.EW: sum([traci.lane.getLength(lane) for lane in self.edges[self.EW]]) / vehicle_size_min_gap
        }

    def get_density(self):
        ns_density = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.edges[self.NS]]) / self.edges_capacity[self.NS]
        ew_density = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.edges[self.EW]]) / self.edges_capacity[self.EW]
        return ns_density, ew_density

    def get_stopped_vehicles_num(self):
        ns_stopped = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.NS]])
        ew_stopped = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.EW]])
        #print(ns_stopped, ew_stopped)
        return ns_stopped, ew_stopped

    def get_mean_waiting_time(self):
        ns_wait = sum([traci.lane.getWaitingTime(lane) for lane in self.edges[self.NS]])
        ew_wait = sum([traci.lane.getWaitingTime(lane) for lane in self.edges[self.EW]])
        n = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.NS]])
        e = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.EW]])
        if n != 0:
            ns_wait /= n
        if e != 0:
            ew_wait /= e
        #print(ns_stopped, ew_stopped)
        return ns_wait, ew_wait
