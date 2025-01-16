"""Environments from RESCO: https://github.com/jault/RESCO, paper https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf ."""

import os

import sumo_rl
from sumo_rl import env, parallel_env


PATH = os.path.dirname(sumo_rl.__file__)


def grid4x4(parallel=True, **kwargs):
    """Grid 4x4 network.

    Number of agents = 16
    Number of actions = 4
    Agents have the same observation and action space
    """
    kwargs.update(
        {
            "net_file": PATH + "/nets/RESCO/grid4x4/grid4x4.net.xml",
            "route_file": PATH + "/nets/RESCO/grid4x4/grid4x4_1.rou.xml",
            "num_seconds": 3600,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)


def arterial4x4(parallel=True, **kwargs):
    """Arterial 4x4 network.

    Number of agents = 16
    Number of actions = 5
    Agents have the same observation and action space
    """
    kwargs.update(
        {
            "net_file": PATH + "/nets/RESCO/arterial4x4/arterial4x4.net.xml",
            "route_file": PATH + "/nets/RESCO/arterial4x4/arterial4x4_1.rou.xml",
            "num_seconds": 3600,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)


def cologne1(parallel=True, **kwargs):
    """Cologne 1 network.

    Number of agents: 1
    Number of actions: 4
    """
    kwargs.update(
        {
            "net_file": PATH + "/nets/RESCO/cologne1/cologne1.net.xml",
            "route_file": PATH + "/nets/RESCO/cologne1/cologne1.rou.xml",
            "begin_time": 25200,
            "num_seconds": 28800 - 25200,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)


def cologne3(parallel=True, **kwargs):
    """Cologne 3 network.

    Number of agents: 3
    Number of actions: 2 agents with 4 actions and 1 agent with 3 actions
    2 agents have the same observation and action space and 1 has different spaces
    """
    kwargs.update(
        {
            "net_file": PATH + "/nets/RESCO/cologne3/cologne3.net.xml",
            "route_file": PATH + "/nets/RESCO/cologne3/cologne3.rou.xml",
            "begin_time": 25200,
            "num_seconds": 28800 - 25200,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)


def cologne8(parallel=True, **kwargs):
    """Cologne 8 network.

    Number of agents: 8
    Number of actions: variable
    """
    kwargs.update(
        {
            "net_file": PATH + "/nets/RESCO/cologne8/cologne8.net.xml",
            "route_file": PATH + "/nets/RESCO/cologne8/cologne8.rou.xml",
            "begin_time": 25200,
            "num_seconds": 28800 - 25200,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)


def ingolstadt1(parallel=True, **kwargs):
    """Ingolstadt 1 network.

    Number of agents: 1
    Number of actions: 3
    """
    kwargs.update(
        {
            "net_file": PATH + "/nets/RESCO/ingolstadt1/ingolstadt1.net.xml",
            "route_file": PATH + "/nets/RESCO/ingolstadt1/ingolstadt1.rou.xml",
            "begin_time": 57600,
            "num_seconds": 61200 - 57600,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)


def ingolstadt7(parallel=True, **kwargs):
    """Ingolstadt 7 network.

    Number of agents: 7
    Number of actions: variable
    """
    kwargs.update(
        {
            "net_file": PATH + "/nets/RESCO/ingolstadt7/ingolstadt7.net.xml",
            "route_file": PATH + "/nets/RESCO/ingolstadt7/ingolstadt7.rou.xml",
            "begin_time": 57600,
            "num_seconds": 61200 - 57600,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)


def ingolstadt21(parallel=True, **kwargs):
    """Ingolstadt 21 network.

    Number of agents: 21
    Number of actions: variable
    """
    kwargs.update(
        {
            "net_file": PATH + "/nets/RESCO/ingolstadt21/ingolstadt21.net.xml",
            "route_file": PATH + "/nets/RESCO/ingolstadt21/ingolstadt21.rou.xml",
            "begin_time": 57600,
            "num_seconds": 61200 - 57600,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)
