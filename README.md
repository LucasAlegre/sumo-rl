# Reinforcement Learning - SUMO

Environments inheriting OpenAI Gym Env and RL algorithms to control Traffic Signal controllers on SUMO.

# Install

## To install SUMO:

```
sudo apt-get install sumo sumo-tools sumo-doc
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

## To install required packages:
```
sudo python3 setup.py install
```

# Examples

Use one of the created experiments on /experiments or create your own experiment:

## To run ql_single-intersection:
```
python3 experiments/ql_single-intersection.py 
```
## To plot results:
```
python3 outputs/plot.py -f outputs/single-intersection_rewardqueue_run1.csv
```
