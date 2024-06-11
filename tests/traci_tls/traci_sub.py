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

vehID = 'right_20'
traci.start(['sumo-gui', "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    present_vehicle_ids = traci.vehicle.getIDList()
    vehID = present_vehicle_ids[0]
    traci.vehicle.subscribe(vehID, (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))
    print(traci.vehicle.getSubscriptionResults(vehID))
    for step in range(3):
        print("step", step)
        traci.simulationStep()
        print(traci.vehicle.getSubscriptionResults(vehID))

traci.close()


"""
results:

{80: '52o', 86: 5.1}
step 0
{80: '52o', 86: 5.88593565840274}
step 1
{80: '52o', 86: 7.448970263078808}
step 2
{80: '52o', 86: 9.625052044261245}
{80: '2i', 86: 2.372327181510625}
step 0
{80: '2i', 86: 5.6942515280097705}
step 1
{80: '2i', 86: 9.447699875384567}
step 2
{80: '2i', 86: 13.829500807542352}
{80: '2i', 86: 18.69277858454734}
step 0
{80: '2i', 86: 24.086797169595958}
step 1
{80: '2i', 86: 30.01640003416687}
step 2
{80: '2i', 86: 36.63956250082701}
{80: '2i', 86: 43.85558764087036}

"""

# 80,86是什么意思？
# 52O 是edge id
# 5.1 是vehID在lane上的位置
