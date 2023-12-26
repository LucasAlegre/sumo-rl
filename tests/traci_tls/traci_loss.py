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


def _compute_loss():
    traci.start(['sumo', "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    # pick an arbitrary junction
    junctionID = traci.junction.getIDList()[0]
    # subscribe around that junction with a sufficiently large
    # radius to retrieve the speeds of all vehicles in every step
    traci.junction.subscribeContext(
        junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 1000000,
        [tc.VAR_SPEED, tc.VAR_ALLOWED_SPEED]
    )
    stepLength = traci.simulation.getDeltaT()

    total_time_loss = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        scResults = traci.junction.getContextSubscriptionResults(junctionID)
        halting = 0
        if scResults:
            relSpeeds = [d[tc.VAR_SPEED] / d[tc.VAR_ALLOWED_SPEED] for d in scResults.values()]
            # compute values corresponding to summary-output
            running = len(relSpeeds) # 进入路口范围的车辆数
            halting = len([1 for d in scResults.values() if d[tc.VAR_SPEED] < 0.1])  # 停车等候的车辆数
            meanSpeedRelative = sum(relSpeeds) / running # 平均车数
            timeLoss = (1 - meanSpeedRelative) * running * stepLength # 时间步长内的等车时间：时间损失
            print("timeLoss:", traci.simulation.getTime(), timeLoss, halting)
            total_time_loss += timeLoss
    traci.close()

    print("total_time_loss:", total_time_loss)


_compute_loss()

"""
results:

"""
