import asyncio

from typing import List, Dict, TypedDict, Literal

from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

PROMPTS_DIRECTORY = "./laser/llm_agents/prompts"

NAVIGATION_COMMAND = Literal['TURN_LEFT', 'STRAIGHT', 'TURN_RIGHT']
LANE_CHANGE_DIRECTION = Literal['LEFT LANE CHANGE', 'FOLLOW LANE', 'RIGHT LANE CHANGE']

class Steps:
    def __init__(self, steps: List):
        self._steps = steps
        self._steps_str = []
        for i, step in enumerate(self._steps):
            self._steps_str.append(f"Step {i + 1}. action: {self._steps[i]['action']}, termination_condition: {self._steps[i]['termination_condition']}")

    def get(self, idx):
        return self._steps_str[idx]
    
    def __str__(self) -> str:
        ret = ""
        for step_str in self._steps_str:
            ret += f"{step_str}\n"
        return ret
    def __len__(self):
        return self._steps_str.__len__()
    
class LLMAgent:
    def __init__(self, driving_lane_num, decision_interpreter, llm, steps, agent, Actions, system_prompt_path, user_prompt_path) -> None:
        self._llm = llm.bind_tools([Actions])
        self._decision_interpreter = decision_interpreter
        self._steps = Steps(steps)
        self.driving_lane_num = driving_lane_num
        self._laser_agent = agent
        self._last_step_number = 1
        self._system_prompt_path = system_prompt_path
        self._user_prompt_path = user_prompt_path
        self.lane_id, self.location, self.speed, self.acceleration, self.direction_angle = self._laser_agent.get_self_obs_info()
        self.usage_metadata={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    
    async def get_decisions(self, image) -> Dict:
        if self._last_step_number == self._steps.__len__():
            if self._laser_agent.type == "vehicle":
                decisions = {
                    'current_step_number': self._last_step_number,
                    'lane_change_direction': 'FOLLOW LANE',
                    'lane_change_delay': 0,
                    'target_speed': self.speed
                    }
                return decisions
            elif self._laser_agent.type == "pedestrian":
                decisions = {
                    'current_step_number': self._last_step_number,
                    'steer': 0,
                    'speed': self.speed
                    }
                return decisions
        tool_calls = []
        while not tool_calls:
            template = ChatPromptTemplate.from_messages([
                self.render_system_prompt(),
                self.render_user_prompt(), 
            ])

            self._chain = template | self._llm

            output = self._chain.invoke({})
            usage_metadata = output.usage_metadata
            for k, v in usage_metadata.items():
                self.usage_metadata[k] += v
            tool_calls = output.tool_calls

        args = tool_calls[0]['args']

        self._last_step_number = args['current_step_number']
        return args

    def render_system_prompt(self):
        '''
            system prompt contains
                goal: finish task and generate scene
                input output format
        '''

        system_prompt = SystemMessagePromptTemplate(
            prompt = PromptTemplate.from_file(
                self._system_prompt_path,
                input_variables=None
            )
        )
        
        return system_prompt
    
    def render_user_prompt(self):
        self.lane_id, self.location, self.speed, self.acceleration, self.direction_angle = self._laser_agent.get_self_obs_info()

        t_user_prompt = HumanMessagePromptTemplate(
            prompt = PromptTemplate.from_file(
                self._user_prompt_path,
                ["steps", "last_step" "observation"], 
                partial_variables={
                    "steps": str(self._steps), 
                    "last_step": self._steps.get(self._last_step_number - 1),
                    "observation": self.render_self_info() + self.render_other_agent_info()
                }
            )
        )

        return t_user_prompt

    def render_self_info(self):
        num_lanes = self.driving_lane_num
        if num_lanes == 1:
            description = "You are driving on a road with only one lane, you can't change lane. "
        else:
            description = f"You are driving on a road with {num_lanes} lanes, and you are currently driving in the {self.get_lane_description(self.lane_id)}. "
        
        description += f"Your current position is `({self.location[0]:.2f}, {self.location[1]:.2f})`, speed is {self.speed:.2f} m/s, acceleration is {self.acceleration:.2f} m/s^2, and lane position is {self.location[0]:.2f} m.\n"
        return description

    def render_other_agent_info(self):
        vehicles, pedestrians, target_vehicle = self._laser_agent.get_other_actors()
        description = ""
        for vehicle in vehicles + target_vehicle:
            description += self.render_other_vehicle_info(vehicle)
        for pedestrian in pedestrians:
            description += self.render_other_pedestrian_info(pedestrian)

        return description

    def render_other_vehicle_info(self, vehicle):
        lane_id, location, speed, acceleration, direction_angle = vehicle.get_self_obs_info()
        description = ""
        description += f"- `{vehicle.name}` is driving on {self.get_relative_lane_description(lane_id)} and {self.get_relative_state(vehicle)}. "
        description += f"Its current position is `({location[0]:.2f}, {location[1]:.2f})`, where {location[0]:.2f} is the longitudinal position and {location[1]:.2f} is the lateral position. The longitudinal position is parallel to the lane, and the lateral position is perpendicular to the lane."
        description += f"Its current speed is {speed:.2f} m/s, acceleration is {acceleration:.2f} m/s^2, and lane position is {location[0]:.2f} m.\n"
        if direction_angle > 5:
            description += f"It is changing lane to the right.\n"
        elif direction_angle < -5:
            description += f"It is changing lane to the left.\n"
        else:
            description += f"\n"
        return description

    def render_other_pedestrian_info(self, pedestrian):
        lane_id, location, speed, acceleration, direction_angle = pedestrian.get_self_obs_info()
        description = ""
        description += f"- `{pedestrian.name}` is walking {self.get_pedetrian_direction(direction_angle)}. It is on {self.get_relative_lane_description(lane_id)} and {self.get_relative_state(pedestrian)}. "
        description += f"Its current position is `({location[0]:.2f}, {location[1]:.2f})`, where {location[0]:.2f} is the longitudinal position and {location[1]:.2f} is the lateral position. The longitudinal position is parallel to the lane, and the lateral position is perpendicular to the lane."
        description += f"Its current speed is {speed:.2f} m/s, acceleration is {acceleration:.2f} m/s^2, and lane position is {location[0]:.2f} m.\n"
        return description

    def get_relative_lane_description(self, lane_id):
        if self.lane_id == lane_id:
            return "the same lane as you"
        elif self.lane_id == lane_id - 1:
            return "the lane to your right"
        elif self.lane_id == lane_id + 1:
            return "the lane to your left"
        else:
            return f"the {self.get_lane_description(lane_id)}"
    
    def get_lane_description(self, lane_id):
        num_lanes = self.driving_lane_num
        lane_rank_dict = {
            1: 'first',
            2: 'second',
            3: 'third',
            4: 'fourth'
        }
        return f"{lane_rank_dict[lane_id]} lane from the left"
         
    def get_relative_state(self, vehicle):
        lane_id, location, speed, acceleration, direction_angle = vehicle.get_self_obs_info()
        if self.location[0] >= location[0] + 1:
            description = "is behind of you"
        elif self.location[0] <= location[0] - 1:
            description = "is ahead of you"
        else:
            description = "is on the same horizontal line of you"
        return description
    
    def get_pedetrian_direction(self, direction_angle):
        while direction_angle > 180:
            direction_angle -= 360
        while direction_angle < -180:
            direction_angle += 360
        if direction_angle > 0:
            return "from left to right"
        else:
            return "from right to left"
