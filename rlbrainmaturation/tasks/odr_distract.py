from __future__ import annotations
from typing import Dict, List, Tuple, Union
from rlbrainmaturation.tasks.task import Task, Instruction
import numpy as np
from rlbrainmaturation.utils.general_utils import Coordinates


class ODRDistract(Task):
    """ODRDistract task class.
    ODRRadom task operates in the following 6 steps:
    1. fixation
    2. fixation + cue signal
    3. fixation
    4. fixation + random distraction signal
    5. fixation
    6. no signal and expect to observe saccade toward the cue signal in step 2
    """

    def __init__(
        self,
        target_x: int = 1,
        target_y: int = 5,
        width: int = 42,
        height: int = 42,
        encourage_mode: bool = True,
    ):
        """
        Args:
            target_x: int
                The horizontal coordinate of the target focus point. The unit of the coordinate is pixel.
            target_y: int
                The vertical coordinate of the target focus point. The unit of the coordinate is pixel.
            width: int
                The width of the screen. Default value is 10. The unit of width is pixel.
            height: int
                The height of the screen. Default value is 10. The unit of width is pixel.
            encourage_mode: bool
                Encourage mode or not. Default value is True, which allows to return the score as np.exp(-distance).
                When encourage_mode is false, a sparse reward of 1 is only returned when the distance is smaller than epsilon. 
        """

        instructions = {
            0: [Instruction(time=0, x=5, y=5)],  # fixation
            1: [
                Instruction(time=1, x=5, y=5),  # fixation
                Instruction(time=1, x=target_x, y=target_y),  # cue
            ],
            2: [Instruction(time=0, x=5, y=5)],  # fixation
            3: [
                Instruction(time=0, x=5, y=5),  # fixation
                Instruction(
                    time=2, x=width, y=height, rng=np.random.default_rng()
                ),  # random distractor signal
            ],
            4: [Instruction(time=0, x=5, y=5)],  # fixation
        }

        super().__init__(
            target_x=target_x,
            target_y=target_y,
            instructions=instructions,
            tot_frames=6,
            width=width,
            height=height,
            encourage_mode=encourage_mode,
        )
