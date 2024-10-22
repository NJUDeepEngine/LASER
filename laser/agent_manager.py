import logging
import os
import json
import asyncio
import time
import carla
from queue import Queue
from laser.laser_agents import Agent

import numpy as np
from laser.target_vehicle.dummy_target_vehicle import DummyTargetVehicle
from laser.target_vehicle.target_vehicle import TargetVehicle

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from laser.target_vehicle.target_vehicle_recorder import TargetVehicleRecorder

time_delta = 0.05 # time per frame
time_query = 0.5 # time per llm query

class AgentManager:
    def __init__(self, client, world, simulation_time, scene_script, lane_wps, driving_lane_num, llm) -> None:
        self._timeout = 10
        self.client = client
        self.world = world
        self.simulation_time = simulation_time
        
        settings = world.get_settings()
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#possible-configurations
        # Synchronous mode + fixed time-step
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = time_delta
        world.apply_settings(settings)

        world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))
        time.sleep(1)
        self._stride = int(time_query / time_delta)
        self._stride_cnt = self._stride - 1

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        self.target_recorder = TargetVehicleRecorder(client)
        
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # Init LASER agents
        self.sensor_queue = Queue()
        self._vehicles = []
        self._pedestrians = []
        self._target_vehicle = []
        for agent_name, agent_script in scene_script.items():
            if agent_script["type"] == 'VUT':
                self._target_vehicle.append(TargetVehicle(world, agent_name, agent_script, lane_wps, self, self.sensor_queue))
            elif agent_script["type"] == 'dummy':
                self._vehicles.append(DummyTargetVehicle(world, agent_name, agent_script, lane_wps, driving_lane_num, self, self.sensor_queue))
            elif agent_script["type"] == 'agent':
                agent = Agent(world, agent_name, agent_script, lane_wps, driving_lane_num, llm, self, self.sensor_queue)
                if agent.type == 'vehicle':
                    self._vehicles.append(agent)
                elif agent.type == 'pedestrian':
                    self._pedestrians.append(agent)

        world.tick() # for actors to spawn, and then their locations are correct
        self.agents = self._target_vehicle + self._vehicles + self._pedestrians
        for agent in self.agents:
            agent.init_after_carla_tick()
        for _ in range(9): 
            world.tick() # for setting velocity
        
        self.spectator_chase()

        self.target_recorder.register_actor(self.agents[0].carla_actor)

        self.logger = logging.getLogger(__name__)

    def parse_sensor_data(self, dt):
        self.logger.debug(self.sensor_queue.qsize())
        self.handle_sensor_data(dt)
        while (self.sensor_queue.qsize() > 0):
            sensor_data, agent = self.sensor_queue.get()
            if type(sensor_data) == carla.CollisionEvent:
                self.target_recorder.parse_CollisionEvent(sensor_data, agent)

    async def get_decisions(self, dt):
        decisions = [agent.get_decisions(None) for agent in self.agents]
        results = await asyncio.gather(*decisions)
        return results
        
    def handle_sensor_data(self, dt):
        results = asyncio.run(self.get_decisions(dt))
        # print(results)
        for agent, decisions in zip(self.agents, results):
            agent.handle_decisions(decisions, dt)

    def track(self, dt):
        for agent in self.agents:
            agent.on_tick(dt)

    def destroy(self, SW_token):
        # token usage
        
        with open(os.path.join(self.target_recorder.directory, 'token.txt'), 'w') as file:
            token_usage = {
                "SW": SW_token,
            }
            for agent in self.agents:
                if agent._llm_agent is not None:
                    token_usage[agent.name] = agent._llm_agent_inst.usage_metadata
            json_object = json.dumps(token_usage, indent=4)
            file.write(json_object)

        with open(os.path.join(self.target_recorder.directory, 'time.txt'), 'w') as file:
            time_usage = {
                "simulation_time": self._timestamp_last_run - self._timestamp_start,
                "real_world_time": time.time() - self.start_system_time,
            }
            json_object = json.dumps(time_usage, indent=4)
            file.write(json_object)

        # destroy
        print(f"Finish time: {self._timestamp_last_run}")
        self.target_recorder.destroy()
        for agent in self.agents:
            agent.destroy()



    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()

        self._running = True

        world = CarlaDataProvider.get_world()
        snapshot = world.get_snapshot()
        self._timestamp_start = snapshot.timestamp.elapsed_seconds
        self._timestamp_last_run = self._timestamp_start
        print(f"Start time: {self._timestamp_start}")

        while self._timestamp_last_run - self._timestamp_start <= self.simulation_time - time_delta: 
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self._stride_cnt += 1
            if self._stride_cnt == self._stride:
                self._stride_cnt = 0
                self.parse_sensor_data(time_delta)
            self.track(time_delta)

            self.spectator_chase()


        if self._running:
            CarlaDataProvider.get_world().tick(self._timeout)

    def spectator_chase(self):
        chased_actor = self.agents[0].carla_actor
        spectator = CarlaDataProvider.get_world().get_spectator()
        trans = chased_actor.get_transform()
        spectator.set_transform(carla.Transform(trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

