import time
from time import perf_counter, sleep
from functools import wraps
from typing import Callable, Any

import carla
import numpy as np

def get_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:

        # Note that timing your code once isn't the most reliable option
        # for timing your code. Look into the timeit module for more accurate
        # timing.
        start_time: float = perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = perf_counter()

        print(f'"{func.__name__}()" took {end_time - start_time:.3f} seconds to execute')
        return result

    return wrapper


def relative_transform(source: carla.Transform, target: carla.Transform):
    """

    https://github.com/carla-simulator/carla/issues/2915
    """
    source_t = np.array(source.get_matrix())[3, :2]
    # target_t = transform2mat(target)
    target_inv = np.array(target.get_inverse_matrix())

    relative_loaction = np.dot(target_inv , source_t)


    return relative_transform_transform

def vector3D2list(vector: carla.Vector3D):
    return [vector.x, vector.y, vector.z]
