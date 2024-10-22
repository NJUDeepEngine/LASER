import carla

sensor_args = {
    'sensor_bp_name': 'sensor.camera.rgb', 
    'sensor_type': 'Front Camera RGB', 
    'sensor_bp_args': {
        'image_size_x': str(1600),
        'image_size_y': str(900),
        'fov': '70',
        'exposure_mode': 'auto exposure histogram',
        'sensor_tick': str(0.5),
    }
}

class CameraSensor:
    def __init__(self, sensor_name, sensor_args, parent_actor, transform):
        self.sensor_name = sensor_name
        self.sensor_args = sensor_args
        self.parent_actor = parent_actor

        world = self.parent_actor.get_world()
        bp_library = world.get_blueprint_library()

        args = self.sensor_args
        bp = bp_library.find(args['sensor_bp_name'])
        for attr_name, attr_value in args['sensor_bp_args'].items():
            bp.set_attribute(attr_name, attr_value)
        self.sensor = world.spawn_actor(bp, transform, attach_to=self.parent_actor, attachment_type=carla.AttachmentType.Rigid)
    
    def destroy(self):
        self.sensor.destroy()

        
    # @staticmethod
    # def sensor_callback(sensor_data, sensor_queue, sensor_name):
    #     """
    #     WARNING: CONCURRENCY & DATA RACE
    #     """
    #     sensor_queue.put((sensor_data, sensor_name))

class CollisionSensor:
    def __init__(self, queue, parent_agent):
        self.parent_agent = parent_agent
        self.parent_actor = parent_agent.carla_actor
        self.collision_count = 0

        world = self.parent_actor.get_world()
        bp_library = world.get_blueprint_library()

        bp = bp_library.find("sensor.other.collision")

        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.parent_actor, attachment_type=carla.AttachmentType.Rigid)
        self.sensor.listen(lambda data, agent=self.parent_agent: CollisionSensor.sensor_callback(data, agent, queue))

    def destroy(self):
        self.sensor.destroy()

        
    @staticmethod
    def sensor_callback(sensor_data, agent, sensor_queue):
        """
        WARNING: CONCURRENCY & DATA RACE
        """
        sensor_queue.put((sensor_data, agent))

