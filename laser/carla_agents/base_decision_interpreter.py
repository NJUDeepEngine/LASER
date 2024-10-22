from typing import Dict, TypedDict, Literal
from abc import ABC, abstractmethod

from enum import IntEnum
import carla
import numpy as np
from .navigation.basic_agent import RoadOption, BasicAgent

class BaseDecisionInterpreter(ABC):
    def __init__(self, actor) -> None:
        self._actor = actor

    @abstractmethod
    def handle_decisions(self, decisions: Dict, dt: float) -> None:
        pass

    @abstractmethod 
    def on_tick(self, dt: float) -> None:
        pass
