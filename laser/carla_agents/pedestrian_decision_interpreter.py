import carla
import numpy as np
from .base_decision_interpreter import BaseDecisionInterpreter

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class PedestrianDecisionInterpreter(BaseDecisionInterpreter):
    def __init__(self, actor, driving_lane_num, lane_id2lane_num, init_wp, speed) -> None:
        super().__init__(actor)

        self._speed_limit = 1.5
        self.current_wp = init_wp
        self._target_speed = speed
        self._target_yaw = actor.get_transform().rotation.yaw
        self.driving_lane_num = driving_lane_num
        self.lane_id2lane_num = lane_id2lane_num

    def handle_decisions(self, decisions, dt):
        self.speed_handler(decisions['speed'])

        speed = self._target_speed
        yaw = np.deg2rad(self._target_yaw)
        loc_x = speed * np.cos(yaw)
        loc_y = speed * np.sin(yaw)
        direction = carla.Vector3D(loc_x, loc_y, 0.0)
        self._actor.apply_control(carla.WalkerControl(direction, speed))
        
    def on_tick(self, dt):
        pass

    def steer_handler(self, decision): 
        print(decision)
        steer = decision
        self._target_yaw = self._target_yaw + steer

    def speed_handler(self, decision):
        speed = decision
        self._target_speed = speed
        if self._target_speed > self._speed_limit:
            self._target_speed = self._speed_limit
        if self._target_speed < 0:
            self._target_speed = 0

    def get_self_obs_info(self):
        transform = self._actor.get_transform()
        wp = CarlaDataProvider.get_map().get_waypoint(transform.location, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
        lane_id = wp.lane_id
        speed = self._actor.get_velocity()
        acceleration = self._actor.get_acceleration()

        return lane_id, transform, speed, acceleration


