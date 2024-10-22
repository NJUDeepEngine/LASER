import os
import asyncio

from typing import List, Dict, TypedDict, Literal

from langchain.pydantic_v1 import BaseModel, Field

from laser.llm_agents.llm_agent import LLMAgent
PROMPTS_DIRECTORY = "./laser/llm_agents/prompts"
system_prompt_path = os.path.join(PROMPTS_DIRECTORY, "pedestrian", "SystemPrompt1d.txt")
user_prompt_path = os.path.join(PROMPTS_DIRECTORY, "pedestrian", "UserPrompt1d.txt")

class PedestrianActions(BaseModel):
    current_step_number: int = Field(description="the number of the current step")
    speed: float = Field(description="speed of the pedestrian in $m/s$")

class PedestrianLLMAgent(LLMAgent):
    def __init__(self, driving_lane_num, decision_interpreter, llm, steps, agent) -> None:
        super().__init__(driving_lane_num, decision_interpreter, llm, steps, agent, PedestrianActions, system_prompt_path, user_prompt_path)

    def render_self_info(self):
        if self.driving_lane_num == 1:
            description = "You are walking on a road with only one lane. "
        else:
            description = f"You are walking on a road with {self.driving_lane_num} lanes, and you are currently walking in the {self.get_lane_description(self.lane_id)}. "
        
        description += f"Your current position is `({self.location[0]:.2f}, {self.location[1]:.2f})`,where {self.location[0]:.2f} is the longitudinal position and {self.location[1]:.2f} is the lateral position. The longitudinal position is parallel to the lane, and the lateral position is perpendicular to the lane."
        description += f"Your current speed is {self.speed:.2f} m/s, acceleration is {self.acceleration:.2f} m/s^2, and lane position is {self.location[0]:.2f} m.\n"
        return description

