from typing import TypedDict, Literal

from enum import IntEnum
import carla
import numpy as np
from laser.carla_agents.navigation.basic_agent import RoadOption, BasicAgent
from .base_decision_interpreter import BaseDecisionInterpreter
from laser.llm_agents.llm_agent import NAVIGATION_COMMAND, LANE_CHANGE_DIRECTION


class VehicleDecisionInterpreter(BaseDecisionInterpreter):
    def __init__(self, actor, driving_lane_num, lane_id2lane_num, init_wp, init_speed) -> None:
        super().__init__(actor)
        direction = 1
        self._pnc = BasicAgent(self._actor, direction, target_speed=init_speed, opt_dict={}, map_inst=None)
        self.driving_lane_num = driving_lane_num
        self.lane_id2lane_num = lane_id2lane_num
        self.target_lane_num = self.lane_id2lane_num[init_wp.lane_id]

    def handle_decisions(self, decisions, dt):
        self.lane_change_handler(decisions['lane_change_direction'])
        self.target_speed_handler(decisions['target_speed'])
        
    def on_tick(self, dt):
        control = self._pnc.run_step()
        self._actor.apply_control(control)

    def lane_change_handler(self, lane_change_direction: LANE_CHANGE_DIRECTION): 
        if lane_change_direction == 'FOLLOW LANE' or lane_change_direction == 'FOLLOW_LANE': # LLM may make mistakes
            pass
        elif lane_change_direction == 'LEFT LANE CHANGE':
            if self.target_lane_num - 1 == 0:
                return  
            self.target_lane_num = self.target_lane_num - 1
            self._pnc.lane_change('left', other_lane_time=10)
        elif lane_change_direction == 'RIGHT LANE CHANGE':
            if self.target_lane_num + 1 == self.driving_lane_num + 1:
                return
            self.target_lane_num = self.target_lane_num + 1
            self._pnc.lane_change('right', other_lane_time=10)
        else:
            raise ValueError('Invalid DSL')

    def target_speed_handler(self, target_speed):
        self._pnc.set_target_speed(3.6 * target_speed)
        
    def get_current_lane_num(self):
        return self.lane_id2lane_num[self.get_current_waypoint().lane_id]

    def get_current_waypoint(self):
        return self._pnc._map.get_waypoint(self._actor.get_location())
    
    def get_self_obs_info(self):
        lane_id = self.get_current_waypoint().lane_id
        transform = self._actor.get_transform()
        speed = self._actor.get_velocity()
        acceleration = self._actor.get_acceleration()

        return lane_id, transform, speed, acceleration
