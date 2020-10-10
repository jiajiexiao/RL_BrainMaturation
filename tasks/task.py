from __future__ import annotations
from typing import Dict, List, Union, Tuple
from utils.general_utils import Coordinates
import numpy as np


class Instruction:
    """This class defines where the signal displays on the screen
    """

    def __init__(
        self, time: int, x: int, y: int, rng: Optional[np.random.Generator] = None
    ):
        """
        Args:
            time: int
                The time when the instructure got released
            x: int
                The horizontal coordinate where the signal displays
            y: int
                The vertical coordinate where the signal displays
            rng: Optional[np.random.Generator] = None
                Optional numpy raomdom number generator to add an randomized component to the signal          
        """
        self.time = time
        self.x = x
        self.y = y
        self.rng = rng

    @property
    def position(self) -> Coordinates:
        """Coordinates of the signal
        """
        if self.rng is None:
            signal_pos = Coordinates(x=self.x, y=self.y)
        else:
            # random shrinkage factors
            rand_factor_x, random_factor_y = self.rng.random(size=2)
            signal_pos = Coordinates(
                x=int(rand_factor_x * self.x), y=int(random_factor_y * self.y)
            )
        return signal_pos


class Task:
    """This Task class defines the task to be learned.
    """

    def __init__(
        self,
        target_x: int,
        target_y: int,
        instructions: Dict[int, List[Instruction]],
        tot_frames: int = 5,
        width: int = 42,
        height: int = 42,
        encourage_mode: bool = True,
        epsilon: float = 1.0,
    ):
        """
        Args:
            target_x: int
                The horizontal coordinate of the target focus point. The unit of the coordinate is pixel. 
            target_y: int
                The vertical coordinate of the target focus point. The unit of the coordinate is pixel.     
            instructions: List[Instruction]
                List of instructions 
            width: int
                The width of the screen. Default value is 10. The unit of width is pixel. 
            height: int
                The height of the screen. Default value is 10. The unit of width is pixel.
            tot_frames: int
                Total number of frames in the task. Default value is 5.
            encourage_mode: bool
                Encourage mode or not. Default value is True, which allows to return the score as np.exp(-distance).
                When encourage_mode is false, a sparse reward of 1 is only returned when the distance is smaller than epsilon. 
        """

        assert 0 <= target_x <= width, "Target x has to be in the range of [0, width]"
        assert 0 <= target_y <= height, "Target y has to be in the range of [0, height]"

        self.tot_frames = tot_frames
        self.width = width
        self.height = height

        self.instructions = instructions
        self.target_pos = Coordinates(x=target_x, y=target_y)
        self.encourage_mode = encourage_mode
        self.epsilon = epsilon

    def issue_instruction(self, t: int) -> Union[List[Instruction], None]:
        """This method is to release signals to display on the screen
        """
        instructions_at_t = self.instructions.get(t, None)

        if instructions_at_t is not None:
            # validate if any signals are located outside the screen
            self._validate_signals(instructions_at_t)
        return instructions_at_t

    def score(self, focus_point: Tuple[int, int], time: int) -> float:
        """This method computes reward score. 
        In the base brain maturation tasks, reward is not returned until the last step. 
        At the last step, the score is based on how close the focus point is to the target focus points.
        
        Args:
            focus_point: Tuple[int, int]
                The tuple that describes the coordinate of focus_point, which is the action taken by the agent
            time: int
                Time step to compute the score
        """
        if time < self.tot_frames - 1:
            # don't provide reward until the last step
            return 0
        else:
            target_pos = self.target_pos
            distance_square = (focus_point[0] - target_pos.x) ** 2 + (
                focus_point[1] - target_pos.y
            ) ** 2
            if self.encourage_mode:
                return np.exp(-distance_square)
            else:
                return 1.0 if distance_square < self.epsilon else 0.0

    def _validate_signals(self, instructions: List[Instruction]) -> None:
        """validate if any signals from the list of instructions are outside the screen.
        """
        for instruction in instructions:
            assert (
                0 <= instruction.x <= self.width
            ), f"x in Instruction at {instruction.time} is outside the screen"

            assert (
                0 <= instruction.y <= self.height
            ), f"y in Instruction at {instruction.time} is outside the screen"
