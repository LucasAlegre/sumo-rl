from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci  # noqa
import traci.constants as tc  # noqa

junctionID = '0'
traci.start(['sumo-gui', "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    traci.vehicle.subscribeContext("right_120", tc.CMD_GET_VEHICLE_VARIABLE, 0.0, [tc.VAR_SPEED])
    traci.vehicle.addSubscriptionFilterLanes('2o_0', noOpposite=True, downstreamDist=100, upstreamDist=50)

    for step in range(3):
        print("step", step)
        traci.simulationStep()
        print(traci.vehicle.getContextSubscriptionResults("right_120"))

traci.close()

"""
results:

"""
