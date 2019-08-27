# SUMO-RL

SUMO-RL provides a simple interface to instantiate Reinforcement Learning environments with [SUMO v.1.2.0](https://github.com/eclipse/sumo) for Traffic Signal Control. 

The main class [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/master/environment/env.py) inherits [MultiAgentEnv](https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py) from [RLlib](https://github.com/ray-project/ray/tree/master/python/ray/rllib).  
If instantiated with parameter 'single-agent=True', it behaves like a regular [Gym Env](https://github.com/openai/gym/blob/master/gym/core.py) from [OpenAI](https://github.com/openai).  
[TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/master/environment/traffic_signal.py) is responsible for retrieving information and actuating on traffic lights using [TraCI](https://sumo.dlr.de/wiki/TraCI) API.

Goals of this repository:
- Provide a simple interface to work with Reinforcement Learning for Traffic Signal Control using SUMO.
- Support Multiagent RL.
- Compatibility with Gym Env and popular RL libraries like openAI baselines and RLlib.
- Easy customisation: state and reward definitions are easily modifiable.

## Install

### To install SUMO v1.2.0:

```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc 
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

### To install sumo_rl package:
```
pip3 install -e .
```

## Examples

Check [experiments](https://github.com/LucasAlegre/sumo-rl/tree/master/experiments) to see how to instantiate a SumoEnvironment and use it with your RL algorithm.

### [Q-learning](https://github.com/LucasAlegre/sumo-rl/blob/master/agents/ql_agent.py) in a one-way single intersection:
```
python3 experiments/ql_single-intersection.py 
```

### [RLlib A3C](https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/a3c) multiagent in a 4x4 grid:
```
python3 experiments/a3c_4x4grid.py
```

### [stable-baselines A2C](https://stable-baselines.readthedocs.io/en/master/modules/a2c.html) in a 2-way single intersection:
```
python3 experiments/a2c_2way-single-intersection.py
```

### To plot results:
```
python3 outputs/plot.py -f outputs/2way-single-intersection/a3c 
```
![alt text](https://github.com/LucasAlegre/sumo-rl/blob/master/outputs/result.png)

## Cite
If you use this repository in your research, please cite:
```
@misc{sumorl,
    author = {Lucas N. Alegre},
    title = {SUMO-RL},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}
```

