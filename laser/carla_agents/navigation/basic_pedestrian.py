import carla

class BasicPedestrian:
    def __init__(self, actor) -> None:
        self._actor = actor
        
        self.target_speed = 3.0