from __future__ import annotations
import numpy as np
import gym
from gym import spaces
from rlbrainmaturation.tasks.task import Task
from typing import List, Tuple, Dict


class Environment(gym.Env):
    """Environment class that simulate the environement of the brain maturation experiments.
    
    The environment assumes the observation is the 2D screen with 1 color channel. 
    The task of the brain maturation experiments is configged via the dictionary of env_config.
    """

    def __init__(self, env_config: spaces.Dict[str, Task]) -> None:
        """Inits the env based on the env_config dictionary. 
        Args:
            env_config: Dict[str, Task]
                Environment config dictionary, i.e. env_config = {"task": Task}
        """
        self.task = env_config.get("task")  # Brain maturation task
        height = self.task.height  # height_of_screen
        width = self.task.width  # width_of_screen

        # action space is defined based on where the object would be moved to
        self.action_space = spaces.Tuple(
            (spaces.Discrete(width), spaces.Discrete(height))
        )
        # observation space is constructed as an image with only 1 color channel
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(width, height, 1), dtype=np.uint8,
        )
        # zero background that nothing shows on the screen
        self.zero_background = np.zeros((height, width, 1), dtype=np.uint8)
        self.time = 0  # intial frame
        self.observation = self._update_observation(self.time)  # screen output

    def step(
        self, action: spaces.Tuple[int, int]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Updates the env state/observation based on the action taken by the agent.
        In this base environment, agent's action won't change the environment during 
        simple brain maturation tasks. As a result, the environment will only change
        according to the instructions predefined by the input task/rule.

        Args:
            action: spaces.Tuple[int, int]
                Action output by the RL agent. The action is the position where the 
                agent is looking to. 

        """
        self.time += 1
        # only update according to the task rather than agent's action
        self.observation = self._update_observation(self.time)
        done = self.time >= self.task.tot_frames - 1
        # reward = self.task.score(action) if done else 0.0
        reward = self.task.score(action, self.time)
        return (
            self.observation,
            reward,
            done,
            {},
        )  # return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset the env to the initial config.
        """
        self.time = 0
        self.observation = self._update_observation(self.time)
        return self.observation

    def _update_observation(self, time: int) -> np.ndarray:
        """Helper function that updates the observation (screen output)
        Args:
            time: int
                The time step to update the observation
        """
        observation = self.zero_background
        instructions = self.task.issue_instruction(time)
        if instructions is not None:
            for instruction in instructions:
                observation[instruction.position[0], instruction.position[1], 0] = 1
        return observation
