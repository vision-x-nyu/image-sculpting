from dataclasses import dataclass, field
import torch

from threestudio.utils.typing import *
"""
Copy some part from
    https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/prompt_processors/base.py
"""
@dataclass
class DirectionConfig:
    name: str
    prompt: Callable[[str], str]
    negative_prompt: Callable[[str], str]
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]
    modify_negative_prompt: Callable[[Float[Tensor, "..."], str], str] = lambda _, s: s

def shift_azimuth_deg(azimuth: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return (azimuth + 180) % 360 - 180

class PromptProcessor:
    @dataclass
    class Config:
        """
        Threshold for each view-dependent prompt
            Only concern about azimuth
        """
        overhead_threshold: float = 60.0
        front_threshold: float = 45.0
        back_threshold: float = 45.0

    def __init__(self):
        self.cfg = self.Config()
        
        self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"{s}, side view",
                    lambda s: s,
                    lambda azi: True,
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"{s}, front view",
                    lambda s: s,
                    lambda azi: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"{s}, back view",
                    lambda s: s,
                    lambda azi: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                    lambda azi, s: f"" if torch.any(azi) else s
                ),
            ]
        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}


    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        angles: Float[Tensor, "B"],
    ) -> str:
        """
        Append prompt with view-dependent text based on the mesh rotation. 
        """
        direction_idx = torch.zeros_like(angles, dtype=torch.long)
        for d in self.directions:
            condition = d.condition(angles)
            direction_idx[
                condition
            ] = self.direction2idx[d.name]
            if negative_prompt is not None:
                negative_prompt = d.modify_negative_prompt(condition, negative_prompt)
        
        return self.directions[direction_idx[0].item()].prompt(prompt), negative_prompt

        