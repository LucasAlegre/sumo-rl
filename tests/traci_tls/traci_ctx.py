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
    traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 42, [tc.VAR_SPEED, tc.VAR_WAITING_TIME])
    print(traci.junction.getContextSubscriptionResults(junctionID))
    for step in range(3):
        print("step", step)
        traci.simulationStep()
        print(traci.junction.getContextSubscriptionResults(junctionID))

traci.close()

"""
results:

step 0
{'left_11': {64: 0.0, 122: 15.0}, 'left_12': {64: 0.0, 122: 8.0}, 'left_7': {64: 0.0, 122: 27.0}, 'left_8': {64: 0.0, 122: 23.0}, 'left_9': {64: 0.0, 122: 18.0}, 'right_10': {64: 0.0, 122: 17.0}, 'right_13': {64: 0.0, 122: 6.0}, 'right_14': {64: 9.419887719229363, 122: 0.0}}
step 1
{'left_11': {64: 0.0, 122: 16.0}, 'left_12': {64: 0.0, 122: 9.0}, 'left_7': {64: 0.0, 122: 28.0}, 'left_8': {64: 0.0, 122: 24.0}, 'left_9': {64: 0.0, 122: 19.0}, 'right_10': {64: 0.0, 122: 18.0}, 'right_13': {64: 0.0, 122: 7.0}, 'right_14': {64: 4.919887719229363, 122: 0.0}}
step 2
{'left_11': {64: 0.0, 122: 17.0}, 'left_12': {64: 0.0, 122: 10.0}, 'left_7': {64: 0.0, 122: 29.0}, 'left_8': {64: 0.0, 122: 25.0}, 'left_9': {64: 0.0, 122: 20.0}, 'right_10': {64: 0.0, 122: 19.0}, 'right_13': {64: 0.0, 122: 8.0}, 'right_14': {64: 0.4511045606067332, 122: 0.0}}
{'left_11': {64: 0.0, 122: 18.0}, 'left_12': {64: 0.0, 122: 11.0}, 'left_7': {64: 0.0, 122: 30.0}, 'left_8': {64: 0.0, 122: 26.0}, 'left_9': {64: 0.0, 122: 21.0}, 'right_10': {64: 0.0, 122: 20.0}, 'right_13': {64: 0.0, 122: 9.0}, 'right_14': {64: 0.2421018344346625, 122: 0.0}}

step 0
{'left_11': {64: 0.0, 122: 19.0}, 'left_12': {64: 0.0, 122: 12.0}, 'left_7': {64: 0.0, 122: 31.0}, 'left_8': {64: 0.0, 122: 27.0}, 'left_9': {64: 0.0, 122: 22.0}, 'right_10': {64: 0.0, 122: 21.0}, 'right_13': {64: 0.0, 122: 10.0}, 'right_14': {64: 0.1308803152337154, 122: 0.0}}
step 1
{'left_11': {64: 0.0, 122: 20.0}, 'left_12': {64: 0.0, 122: 13.0}, 'left_7': {64: 0.0, 122: 32.0}, 'left_8': {64: 0.0, 122: 28.0}, 'left_9': {64: 0.0, 122: 23.0}, 'right_10': {64: 0.0, 122: 22.0}, 'right_13': {64: 0.0, 122: 11.0}, 'right_14': {64: 0.004026011551458923, 122: 1.0}, 'right_15': {64: 13.041279833865206, 122: 0.0}}
step 2
{'left_11': {64: 0.0, 122: 21.0}, 'left_12': {64: 0.0, 122: 14.0}, 'left_7': {64: 0.5211062472313643, 122: 0.0}, 'left_8': {64: 0.0, 122: 29.0}, 'left_9': {64: 0.0, 122: 24.0}, 'right_10': {64: 0.4953724293038249, 122: 0.0}, 'right_13': {64: 0.0, 122: 12.0}, 'right_14': {64: 0.001905857631083883, 122: 2.0}, 'right_15': {64: 8.584235030554419, 122: 0.0}}
{'left_11': {64: 0.0, 122: 22.0}, 'left_12': {64: 0.0, 122: 15.0}, 'left_7': {64: 1.1550911675207318, 122: 0.0}, 'left_8': {64: 0.340549574556149, 122: 0.0}, 'left_9': {64: 0.0, 122: 25.0}, 'right_10': {64: 0.9728885198011994, 122: 0.0}, 'right_13': {64: 0.2548953974263138, 122: 0.0}, 'right_14': {64: 0.000295809128535074, 122: 3.0}, 'right_15': {64: 4.461698138248777, 122: 0.0}}

"""

# 64，122是什么意思？
# left_11, right_13 是vehicle id
# 'left_8': {64: 0.0, 122: 29.0}; 'left_7': {64: 1.1550911675207318, 122: 0.0} 其中，哪个是速度，哪个是等待时间？
