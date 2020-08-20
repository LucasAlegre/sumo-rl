import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from gym import Env
import traci.constants as tc
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd

from .traffic_signal import TrafficSignal


class SumoEnvironment(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param time_to_load_vehicles: (int) Number of simulation seconds ran before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(self, net_file, route_file, phases, out_csv_name=None, use_gui=False, num_seconds=20000, max_depart_delay=100000,
                 time_to_teleport=-1, time_to_load_vehicles=0, delta_time=5, yellow_time=2, min_green=5, max_green=50, single_agent=False):

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.single_agent = single_agent
        self.ts_ids = traci.trafficlight.getIDList()
        self.lanes_per_ts = len(set(traci.trafficlight.getControlledLanes(self.ts_ids[0])))
        self.traffic_signals = dict()
        self.phases = phases
        self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.vehicles = dict()
        self.last_measure = dict()  # used to reward function remember last measure
        self.last_reward = {i: 0 for i in self.ts_ids}
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles  # number of simulation seconds ran in reset() before learning starts
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        """
        Default observation space is a vector R^(#greenPhases + 2 * #lanes)
        s = [current phase one-hot encoded, density for each lane, queue for each lane]
        You can change this by modifing self.observation_space and the method _compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds
        """
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 2*self.lanes_per_ts), high=np.ones(self.num_green_phases + 2*self.lanes_per_ts))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                         # Green Phase
            #spaces.Discrete(self.max_green//self.delta_time),               # Elapsed time of phase
            *(spaces.Discrete(10) for _ in range(2*self.lanes_per_ts))      # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = ''

        self.radix_factors = [s.n for s in self.discrete_observation_space.spaces]
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name

        traci.close()
        
    def reset(self):
        if self.run != 0:
            traci.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(self.time_to_teleport),
                     '--random']
        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green, self.phases)
            self.last_measure[ts] = 0.0

        self.vehicles = dict()

        # Load vehicles
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()

        else:
            self._apply_actions(action)

            for _ in range(self.yellow_time):
                self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update_phase()
            for _ in range(self.delta_time - self.yellow_time):
                self._sumo_step()

        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        info = self._compute_step_info()
        self.metrics.append(info)
        self.last_reward = reward

        if self.single_agent:
            return observation[self.ts_ids[0]], reward[self.ts_ids[0]], done['__all__'], {}
        else:
            return observation, reward, done, {}

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].set_next_phase(action)

    def _compute_observations(self):
        """
        Return the current observation for each traffic signal
        """
        observations = {}
        for ts in self.ts_ids:
            phase_id = [1 if self.traffic_signals[ts].phase//2 == i else 0 for i in range(self.num_green_phases)]  #one-hot encoding
            #elapsed = self.traffic_signals[ts].time_on_phase / self.max_green
            density = self.traffic_signals[ts].get_lanes_density()
            queue = self.traffic_signals[ts].get_lanes_queue()
            observations[ts] = phase_id + density + queue
        return observations

    def _compute_rewards(self):
        return self._waiting_time_reward()
        #return self._pressure_reward()
        #return self._queue_reward()
        #return self._waiting_time_reward2()
        #return self._queue_average_reward()
    
    def _pressure_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = -self.traffic_signals[ts].get_pressure()
        return rewards

    def _queue_average_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            new_average = np.mean(self.traffic_signals[ts].get_stopped_vehicles_num())
            rewards[ts] = self.last_measure[ts] - new_average
            self.last_measure[ts] = new_average
        return rewards

    def _queue_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - (sum(self.traffic_signals[ts].get_stopped_vehicles_num()))**2
        return rewards

    def _waiting_time_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time_per_lane())
            rewards[ts] = self.last_measure[ts] - ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _waiting_time_reward2(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            self.last_measure[ts] = ts_wait
            if ts_wait == 0:
                rewards[ts] = 1.0
            else:
                rewards[ts] = 1.0/ts_wait
        return rewards

    def _waiting_time_reward3(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            rewards[ts] = -ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': self.last_reward[self.ts_ids[0]],
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(self.last_measure[ts] for ts in self.ts_ids)
            #'total_wait_time': sum([sum(self.traffic_signals[ts].get_waiting_time()) for ts in self.ts_ids])
        }

    def close(self):
        traci.close()
    
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state):
        phase = state[:self.num_green_phases].index(1)
        #elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self._discretize_density(d) for d in state[self.num_green_phases:]]
        return self.radix_encode([phase] + density_queue)

    def _discretize_density(self, density):
        return min(int(density*10), 9)

    def _discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green
        for i in range(self.max_green//self.delta_time):
            if elapsed <= self.delta_time + i*self.delta_time:
                return i
        return self.max_green//self.delta_time -1

    def radix_encode(self, values):
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

    def radix_decode(self, value):
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res

    
