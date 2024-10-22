import os
import cv2
import time
import numpy as np
import carla
from laser.sensor import CameraSensor

class TargetVehicleRecorder:
    def __init__(self, client) -> None:
        self.client = client
        self.current_time = time.localtime()
        self.directory = os.path.join('se_records', time.strftime('%m%d-%H-%M-%S', self.current_time))
        # linux server
        carla_recording_path_on_server = os.path.join(self.directory, f"{time.strftime('%m%d-%H-%M-%S', self.current_time)}recording.log")
        print(f"target_vehicle_recorder: \nrecording_path: {carla_recording_path_on_server}")
        
        os.makedirs(self.directory, exist_ok=True)
        os.environ['SAVE_PATH'] = self.directory # for leaderboard/team_code/interfuser_agent.py line 219

        self.client.start_recorder(carla_recording_path_on_server, True)
        
    def register_actor(self, carla_actor):
        self._carla_actor = carla_actor
        video_fps = 20
        video_size = (1920,1080)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.video_recorder = cv2.VideoWriter()
        video_path = os.path.join(self.directory, time.strftime('%m%d-%H-%M-%S', self.current_time) + '.mp4')
        print(f"target_vehicle_recorder: \nvideo_path: {video_path}")
        self.video_recorder.open(video_path, fourcc, video_fps, video_size, True)
        sensor_args = {
            'sensor_bp_name': 'sensor.camera.rgb', 
            'sensor_type': 'Front Camera RGB', 
            'sensor_bp_args': {
                'image_size_x': str(video_size[0]),
                'image_size_y': str(video_size[1]),
                'fov': '90',
                'exposure_mode': 'auto exposure histogram',
                'sensor_tick': str(1.0 / video_fps),
            }
        }

        location = carla.Location(0, 0, 30.0)
        transform = carla.Transform(location, carla.Rotation(roll=90, yaw=180, pitch=-90))
        self.camera = CameraSensor(self, sensor_args, self._carla_actor, transform)
        self.camera.sensor.listen(lambda data, video_recorder=self.video_recorder: TargetVehicleRecorder.sensor_callback(data, video_recorder))

        self.collision_num = 0
        self._world = self._carla_actor.get_world()
        blueprint = self._world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = self._world.spawn_actor(blueprint, carla.Transform(), attach_to=self._carla_actor)
        self._collision_sensor.listen(lambda event: self.collision_sensor_callback())
 


    @staticmethod
    def sensor_callback(sensor_data, video_recorder):
        """
        WARNING: CONCURRENCY & DATA RACE
        """
        image = np.array(sensor_data.raw_data)
        image = image.reshape((1080,1920,4))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        video_recorder.write(image)

    def collision_sensor_callback(self):
        """
        WARNING: CONCURRENCY & DATA RACE
        """
        self.collision_num += 1

    def destroy(self):
        # video
        self.camera.destroy()
        self.video_recorder.release()
        # recording
        self.client.stop_recorder()
        print('released')


    def parse_CollisionEvent(self, sensor_data, agent):
        set_agent = {12, # Pedestrian
                     13, # Rider
                     14, # Car 	
                     15, # Truck 	
                     16, # Bus
                     17, # Train
                     18, # Motorcycle 	
                     19 # Bicycle
                    }
        if set_agent & set(sensor_data.other_actor.semantic_tags):
            with open(os.path.join(self.directory, 'collisions.txt'), 'w') as file:
                file.write(f"{sensor_data.timestamp}, {sensor_data.actor}, {sensor_data.other_actor}")
