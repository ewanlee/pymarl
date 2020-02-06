from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .mpe_scenarios.environment import MultiAgentEnv as MPEMultiAgentEnv
from . import mpe_scenarios
import sys
import os

def mpe_env_fn(scenario_name, **kwargs) -> MPEMultiAgentEnv:
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from current directory to be Returns
                            (without the .py extension)
        **kwargs: For compatibility with pymarl
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    # load scenario from script
    scenario = mpe_scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    env = MPEMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, discrete_action=True)
    return env

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["fullobs_collect_treasure"] = partial(mpe_env_fn, scenario_name="fullobs_collect_treasure")
REGISTRY["multi_speaker_listener"] = partial(mpe_env_fn, scenario_name="multi_speaker_listener")

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
