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


class ExampleListener(traci.StepListener):
    def step(self, t):
        # do something after every call to simulationStep
        print("ExampleListener called with parameter %s." % t)
        # indicate that the step listener should stay active in the next step
        return True


junctionID = '0'
traci.start(['sumo-gui', "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])

listener = ExampleListener()
traci.addStepListener(listener)

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    print(listener.getID())
    for step in range(3):
        print("step", step)
        traci.simulationStep()

traci.close()

"""
results:

"""
