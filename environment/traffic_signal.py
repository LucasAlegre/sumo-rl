import traci


class TrafficSignal:

    NS = 0
    EW = 2

    def __init__(self, env, ts_id, delta_time, min_green, max_green, phases):
        self.id = ts_id
        self.env = env
        self.time_on_phase = 0
        self.delta_time = delta_time
        self.min_green = min_green
        self.max_green = max_green
        self.edges = self._compute_edges()
        self.edges_capacity = self._compute_edges_capacity()
        
        logic = traci.trafficlight.Logic("new-program", 0, 0, 0, phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)
        self.phases = [p for p in range(len(phases))]
        
    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def set_phase(self, new_phase):
        """
        
        """
        new_phase *= 2
        #print(new_phase, self.phase)
        if self.phase == new_phase or self.time_on_phase < self.min_green:
            self.time_on_phase += self.delta_time
            traci.trafficlight.setPhase(self.id, self.phase) 
        else:
            self.time_on_phase = self.delta_time
            traci.trafficlight.setPhase(self.id, self.phases[new_phase-1])
            

    def keep(self):
        if self.time_on_phase >= self.max_green:
            self.change()
        else:
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
        #print(len(traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]._phases))
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

    def get_stopped_density(self):
        ns_stopped = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.NS]]) / self.edges_capacity[self.NS]
        ew_stopped = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.EW]]) / self.edges_capacity[self.EW]
        return ns_stopped, ew_stopped

    def get_stopped_vehicles_num(self):
        ns_stopped = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.NS]])
        ew_stopped = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[self.EW]])
        return ns_stopped, ew_stopped

    def get_waiting_time(self):
        ls = traci.lane.getLastStepVehicleIDs(self.edges[self.NS][0]) + traci.lane.getLastStepVehicleIDs(self.edges[self.NS][1])
        ns_wait = 0.0
        for veh in ls:
            veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
            acc = traci.vehicle.getAccumulatedWaitingTime(veh)
            if veh not in self.env.vehicles:
                self.env.vehicles[veh] = {veh_lane: acc}
            else:
                self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
            ns_wait += self.env.vehicles[veh][veh_lane]

        ls = traci.lane.getLastStepVehicleIDs(self.edges[self.EW][0]) + traci.lane.getLastStepVehicleIDs(self.edges[self.EW][1])
        ew_wait = 0.0
        for veh in ls:
            veh_lane = traci.vehicle.getLaneID(veh)
            acc = traci.vehicle.getAccumulatedWaitingTime(veh)
            if veh not in self.env.vehicles:
                self.env.vehicles[veh] = {veh_lane: acc}
            else:
                self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
            ew_wait += self.env.vehicles[veh][veh_lane]
            #print(self.vehicles[veh], traci.vehicle.getWaitingTime(veh), traci.vehicle.getAccumulatedWaitingTime(veh))
            #print(veh_lane, self.get_edge_id(veh_lane))

        return ns_wait, ew_wait

    @staticmethod
    def get_edge_id(lane):
        ''' Get edge Id from lane Id
        :param lane: id of the lane
        :return: the edge id of the lane
        '''
        return lane[:-2]
