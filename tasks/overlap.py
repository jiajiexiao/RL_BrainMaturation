from __future__ import annotations
from typing import Dict, List, Tuple, Union
from tasks.task import Task, Instruction
import numpy as np
from utils.general_utils import Coordinates


class Overlap(Task):
    """Overlap task class.
    Overlap task operates in the following 3 steps:
    1. fixation
    2. fixation + cue signal
    3. no signal and expect to observe saccade toward the opposite direction of cue signal in step 2
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

        instructions: Dict[int, List[Instruction]] = {
            0: [Instruction(time=0, x=5, y=5)],  # fixation
            1: [
                Instruction(time=1, x=5, y=5),  # fixation
                Instruction(time=1, x=width - target_x, y=height - target_y),  # cue
            ],
        }

        super().__init__(
            target_x=target_x,
            target_y=target_y,
            instructions=instructions,
            tot_frames=3,
            width=width,
            height=height,
            encourage_mode=encourage_mode,
        )
