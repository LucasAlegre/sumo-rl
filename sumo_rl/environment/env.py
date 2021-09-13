import os
import sys
from pathlib import Path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.envs.registration import EnvSpec
import numpy as np
import pandas as pd

from .traffic_signal import TrafficSignal

from gym.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector


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
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(self, net_file, route_file, out_csv_name=None, use_gui=False, num_seconds=20000, max_depart_delay=100000,
                 time_to_teleport=-1, delta_time=5, yellow_time=2, min_green=5, max_green=50, single_agent=False):

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.single_agent = single_agent
        self.ts_ids = traci.trafficlight.getIDList()
        self.traffic_signals = {ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green) for ts in self.ts_ids}
        self.vehicles = dict()

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = EnvSpec('SUMORL-v0')

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

        self.traffic_signals = {ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green) for ts in self.ts_ids}

        self.vehicles = dict()

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
                if self.sim_step % 5 == 0:
                    info = self._compute_step_info()
                    self.metrics.append(info)
        else:
            self._apply_actions(action)

            time_to_act = False
            while not time_to_act:
                self._sumo_step()

                for ts in self.ts_ids:
                    self.traffic_signals[ts].update()
                    if self.traffic_signals[ts].time_to_act:
                        time_to_act = True

                if self.sim_step % 5 == 0:
                    info = self._compute_step_info()
                    self.metrics.append(info)

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        dones.update({'__all__': self.sim_step > self.sim_max_time})

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], dones['__all__'], {}
        else:
            return observations, rewards, dones, {}

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

    def _compute_dones(self):
        return {ts_id: False for ts_id in self.ts_ids}
    
    def _compute_observations(self):
        return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space
    
    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space
    
    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': self.traffic_signals[self.ts_ids[0]].last_reward,
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids)
        }

    def close(self):
        traci.close()
    
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        #elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase] + density_queue)

    def _discretize_density(self, density):
        return min(int(density*10), 9)

    def _discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green
        for i in range(self.max_green//self.delta_time):
            if elapsed <= self.delta_time + i*self.delta_time:
                return i
        return self.max_green//self.delta_time - 1


class SumoEnvironmentPZ(AECEnv, EzPickle):
    metadata = {'render.modes': ['human', "rgb_array"], 'name': "sumo"}  # fix

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_space for a in self.agents}
        self.observation_spaces = {a: self.env.observation_space for a in self.agents}

        print(self.agents)
        print(self.action_spaces)
        print(self.observation_spaces)

        # dicts
        self.observations = {}
        self.rewards = {a: 0 for a in self.agents}
        self.dones = {a: False for a in self.agents}  # fix for last
        self.infos = {a: '' for a in self.agents}

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)
        self.env = SumoEnvironment(**self._kwargs)

    def reset(self):
        self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = self.env._compute_rewards()
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.dones = self.env._compute_dones()  # fix for last
        self.infos = {a: '' for a in self.agents}

    def observe(self, agent):
        # Can this be done faster?
        observations = self.env._compute_observations()
        return observations[agent]

    def state(self):
        state = self.env.state()
        return state

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def step(self, action):
        # do
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception('Action for agent {} must be in Discrete({}).'
                            'It is currently {}'.format(agent, self.action_spaces[agent].n, action))

        self.env.step(action, agent)
        # select next agent and observe
        self.agent_selection = self._agent_selector.next()
        self.rewards = self.env._compute_rewards()
        self.dones = self.env.dones
        self.infos = self.env.infos

        self.score = self.env.score

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
