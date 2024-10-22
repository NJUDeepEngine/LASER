from typing import TypedDict, Literal, Type
import asyncio

import carla
from laser.sensor import CollisionSensor
import numpy as np
from laser.carla_agents.vehicle_decision_interpreter import VehicleDecisionInterpreter
from laser.carla_agents.pedestrian_decision_interpreter import PedestrianDecisionInterpreter
from laser.llm_agents.vehicle_llm_agent import VehicleLLMAgent
from laser.llm_agents.pedestrian_llm_agent import PedestrianLLMAgent

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
def handle_list(list_maybe):
    if hasattr(list_maybe, '__getitem__'):
        item = list_maybe[0]
    else:
        item = list_maybe
    return item

def get_bp(world, script):
    if script['model'] == 'car':
        bp_name = 'model3'
    if script['model'] == 'taxi':
        bp_name = 'vehicle.ford.crown' 
    elif script['model'] == 'police car':
        bp_name = 'vehicle.dodge.charger_police_2020'
    elif script['model'] == 'truck':
        bp_name = 'carlacola'
    elif script['model'] == 'bus':
        # bp_name = '*subishi*' # Mitsubishi Fusorosa
        bp_name = 'vehicle.volkswagen.t2_2021' 
    elif script['model'] == 'ambulance':
        bp_name = 'vehicle.ford.ambulance'
    elif script['model'] == 'firetruck':
        bp_name = 'vehicle.carlamotors.firetruck'
    elif script['model'] == 'pedestrian':
        bp_name = 'walker.pedestrian.*'
    return bp_name

def name2bp(world, bp_name):
    blueprint_library = world.get_blueprint_library()
    bps = blueprint_library.filter(bp_name)
    
    return handle_list(bps)

def get_type(script):
    if script['model'] == 'pedestrian':
        agent_type = 'pedestrian'
    else:
        agent_type = 'vehicle'
    return agent_type

def get_init_state(world, lane_wps, agent_script):
    init_state = agent_script['init_state']
    l, x = init_state[0] - 1, abs(init_state[1])
    wp = handle_list(lane_wps[l].previous(x))
    
    transform = wp.transform

    transform.location.z += 0.5
    if get_type(agent_script) == 'pedestrian':
        rotation = transform.rotation
        transform.rotation = carla.Rotation(pitch=rotation.pitch, yaw=rotation.yaw - 90, roll=rotation.roll)

    velocity = transform.transform_vector(carla.Vector3D(init_state[2], 0, 0))
    return transform, wp, velocity

def spawn_actor_by_script(world, agent_script, lane_wps, role_name):
    actor_bp_name = get_bp(world, agent_script)

    transform, init_wp, init_velocity = get_init_state(world, lane_wps, agent_script)

    carla_actor = CarlaDataProvider.request_new_actor(actor_bp_name, transform, rolename=role_name)

    assert carla_actor is not None
    carla_actor.set_target_velocity(init_velocity)
    print(f"location: {agent_script['init_state']}\n speed: {carla_actor.get_velocity()}")

    return carla_actor, init_wp, init_velocity


class Agent:
    def __init__(self, world, agent_name, agent_script, lane_wps, driving_lane_num, llm, agent_manager, queue) -> None:
        self.name = agent_name

        self.carla_actor, self.init_wp, self.init_velocity = spawn_actor_by_script(world, agent_script, lane_wps, role_name='scenario')
        self.type = get_type(agent_script)
        if self.type == 'vehicle':
            self._decision_interpreter = VehicleDecisionInterpreter
            self._llm_agent = VehicleLLMAgent
        elif self.type == 'pedestrian':
            self._decision_interpreter = PedestrianDecisionInterpreter
            self._llm_agent = PedestrianLLMAgent
        else:
            assert 0
        self._script = agent_script
        self.lane_wps = lane_wps
        self.driving_lane_num = driving_lane_num
        self.lane_id2lane_num = {}
        for i, lane_wp in enumerate(lane_wps):
            self.lane_id2lane_num[lane_wp.lane_id] = i + 1

        self._llm = llm
        
        self.agent_manager = agent_manager

        self.minimal_tick = 0.5
        self.stride = 1

        self.collision_sensor = CollisionSensor(queue, self)

    
    def init_after_carla_tick(self):
        self._decision_interpreter_inst = self._decision_interpreter(self.carla_actor, self.driving_lane_num, self.lane_id2lane_num, self.init_wp, self.init_velocity.length() * 3.6)
        self._llm_agent_inst = self._llm_agent(self.driving_lane_num, self._decision_interpreter_inst, self._llm, self._script['steps'], self)

    async def get_decisions(self, sensor_data):
        decisions = await self._llm_agent_inst.get_decisions(sensor_data)
        print(f"{self.name} at {self.get_self_obs_info()}:")
        print(f"LLM decision: {decisions}")
        return decisions

    def handle_decisions(self, decisions, dt):
        self._decision_interpreter_inst.handle_decisions(decisions, dt)

    def on_tick(self, dt):
        self._decision_interpreter_inst.on_tick(dt)

    def destroy(self):
        self.collision_sensor.destroy()
        self.carla_actor.destroy()
        
        if self.name == "bus": 
            import os
            with open(os.path.join(self.agent_manager.target_recorder.directory, 'stop_location.txt'), 'w') as file:
                # file.write(f"{20.5 - (150 - self.stop_location[0])}")
                file.write(f"{44.1 - (150 - self.stop_location[0])}")
                # file.write(f"{18.6 - (150 - self.stop_location[0])}")


    def get_other_actors(self):
        vehicles = [actor for actor in self.agent_manager._vehicles if actor.name != self.name]
        pedestrians = [actor for actor in self.agent_manager._pedestrians if actor.name != self.name]
        return vehicles, pedestrians, self.agent_manager._target_vehicle
    
    def get_self_obs_info(self):
        lane_id, transform, speed, acceleration = self._decision_interpreter_inst.get_self_obs_info()
        lane_num = self.lane_id2lane_num[lane_id]
        
        lane_wp = self.lane_wps[0]
        transform_wp = lane_wp.transform
        transform_self = transform
        m_wp = np.array(transform_wp.get_matrix())
        m_wp_inv = np.array(transform_wp.get_inverse_matrix())
        m_self = np.array(transform_self.get_matrix())
        t_in_wp_view = np.dot(m_wp_inv, (m_self - m_wp)[:, 3]) # x axis points to front, y axis points to right
        location = t_in_wp_view[:2]

        direction_angle = transform_self.rotation.yaw - transform_wp.rotation.yaw
        while direction_angle > 180:
            direction_angle -= 360
        while direction_angle < -180:
            direction_angle += 360


        location = [150 + location[0], location[1]]

        speed = speed.length()
        acceleration = acceleration.length()

        if self.name == "bus": 
            if speed < 0.01 and acceleration < 0.01:
                self.stop_location = location

        return lane_num, location, speed, acceleration, direction_angle