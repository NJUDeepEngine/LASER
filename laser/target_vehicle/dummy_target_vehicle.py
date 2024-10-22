from typing import TypedDict, Literal, Type
import asyncio

import carla
from laser.sensor import CollisionSensor
import numpy as np
from laser.carla_agents.vehicle_decision_interpreter import VehicleDecisionInterpreter
from laser.laser_agents import Agent, spawn_actor_by_script

class DummyTargetVehicle(Agent):
    def __init__(self, world, agent_name, agent_script, lane_wps, driving_lane_num, agent_manager, queue) -> None:
        self.name = agent_name

        self.carla_actor, self.init_wp, self.init_velocity = spawn_actor_by_script(world, agent_script, lane_wps, 'scenario')
        self.type = 'vehicle'
        self._decision_interpreter = VehicleDecisionInterpreter
        self._script = agent_script
        self.lane_wps = lane_wps
        self.driving_lane_num = driving_lane_num
        self.lane_id2lane_num = {}
        for i, lane_wp in enumerate(lane_wps):
            self.lane_id2lane_num[lane_wp.lane_id] = i + 1

        self.agent_manager = agent_manager

        self.collision_sensor = CollisionSensor(queue, self)

        self._llm_agent = None


    def init_after_carla_tick(self):
        self._decision_interpreter_inst = self._decision_interpreter(self.carla_actor, self.driving_lane_num, self.lane_id2lane_num, self.init_wp, self.init_velocity.length() * 3.6)

    async def get_decisions(self, sensor_data):
        decisions = {
                    'current_step_number': 1,
                    'lane_change_direction': 'FOLLOW LANE',
                    'lane_change_delay': 0,
                    'target_speed': self.init_velocity.length()
                    }
        return decisions

    def handle_decisions(self, decisions, dt):
        self._decision_interpreter_inst.handle_decisions(decisions, dt)

    def on_tick(self, dt):
        self._decision_interpreter_inst.on_tick(dt)


        
    