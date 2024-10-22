from typing import TypedDict, Literal, Type

import carla
from laser.sensor import CollisionSensor
import numpy as np
from laser.carla_agents.vehicle_decision_interpreter import VehicleDecisionInterpreter
from laser.laser_agents import Agent, spawn_actor_by_script

from team_code.interfuser_agent import InterfuserAgent

from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.envs.sensor_interface import SensorInterface
from leaderboard.utils.route_manipulation import interpolate_trajectory

def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


class TargetVehicle(Agent):
    def __init__(self, world, agent_name, agent_script, lane_wps, agent_manager, queue) -> None:
        self.name = agent_name
        self.carla_actor, self.init_wp, self.init_velocity = spawn_actor_by_script(world, agent_script, lane_wps, role_name='hero')
        CarlaDataProvider.get_world().tick()
        self.type = 'vehicle'
        self._script = agent_script
        self.lane_wps = lane_wps
        self.lane_id2lane_num = {}
        for i, lane_wp in enumerate(lane_wps):
            self.lane_id2lane_num[lane_wp.lane_id] = i + 1

        self.agent_manager = agent_manager

        chased_actor = self.carla_actor
        spectator = CarlaDataProvider.get_world().get_spectator()
        trans = chased_actor.get_transform()
        spectator.set_transform(carla.Transform(trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

        # Set up the user's agent, and the timer to avoid freezing the simulation
        self.agent_instance = InterfuserAgent('leaderboard/team_code/interfuser_config.py')
        # Set target location

        init_state = agent_script['init_state']
        l, x = init_state[0] - 1, abs(init_state[1])

        # if wp.lane_id > 0:
        #     wp = lane_wps[l].previous(x)
        # else:
        #     wp = lane_wps[l].next(x)

        from laser.laser_agents import handle_list
        wp = handle_list(lane_wps[l].previous(x))
        print(wp)
    
        gps_route, self.route = interpolate_trajectory(world, [wp.transform.location, lane_wps[l].next(50)[0].transform.location], hop_resolution=1.0)
        CarlaDataProvider.set_ego_route(convert_transform_to_location(self.route))

        self.agent_instance.set_global_plan(gps_route, self.route)

        CarlaDataProvider.get_world().tick()

        debug_mode = False
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)


        self.agent_instance._init()
        self.agent_instance.sensor_interface = SensorInterface()

        self._agent = AgentWrapper(self.agent_instance)
        self._agent.setup_sensors(self.carla_actor, False)
        # self.camera = sensor(self, sensor_args, self.carla_actor, carla.Transform())
        # self.camera.sensor.listen(lambda data, agent=self: Agent.sensor_callback(data, queue, agent))
    

        # sync state
        # CarlaDataProvider.get_world().tick()


        # Night mode
        # if config.weather.sun_altitude_angle < 0.0:
        #     for vehicle in scenario.ego_vehicles:
        #         vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        self.collision_sensor = CollisionSensor(queue, self)

        self._llm_agent = None

    def init_after_carla_tick(self):
        pass

    def on_tick(self, dt):
        ego_action = self._agent()
        # try:
            # ego_action = self._agent()

        # Special exception inside the agent that isn't caused by the agent
        # except SensorReceivedNoData as e:
            # return

        # except Exception as e:
            # raise AgentError(e)

        print(ego_action)

        self.carla_actor.apply_control(ego_action)

    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0) # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def get_self_obs_info(self):
        transform = self.carla_actor.get_transform()
        wp = CarlaDataProvider.get_map().get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_id = wp.lane_id
        speed = self.carla_actor.get_velocity()
        acceleration = self.carla_actor.get_acceleration()


        # lane_id, transform, speed, acceleration = self._decision_interpreter_inst.get_self_obs_info()
        # print(f"{self.name}: {transform}")
        lane_num = self.lane_id2lane_num[lane_id]
        
        lane_wp = self.lane_wps[0]
        transform_wp = lane_wp.transform
        transform_self = transform
        m_wp = np.array(transform_wp.get_matrix())
        m_wp_inv = np.array(transform_wp.get_inverse_matrix())
        m_self = np.array(transform_self.get_matrix())
        t_in_wp_view = np.dot(m_wp_inv, (m_self - m_wp)[:, 3]) # x axis points to front, y axis points to right
        location = t_in_wp_view[:2]
        # print(lane_wp.transform, transform, t_rel_transformed)

        direction_angle = transform_self.rotation.yaw - transform_wp.rotation.yaw
        while direction_angle > 180:
            direction_angle -= 360
        while direction_angle < -180:
            direction_angle += 360

        location = [150 + location[0], location[1]]

        speed = speed.length()
        acceleration = acceleration.length()
        return lane_num, location, speed, acceleration, direction_angle

    async def get_decisions(self, sensor_data):
        decisions = {
                    'current_step_number': 1,
                    'lane_change_direction': 'FOLLOW LANE',
                    'lane_change_delay': 0,
                    'target_speed': self.init_velocity.length()
                    }
        return decisions

    def handle_decisions(self, decisions, dt):
        pass

