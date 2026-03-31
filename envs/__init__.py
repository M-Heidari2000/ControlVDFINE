from .torus import Torus
from .circle import Circle
from .swiss_roll import SwissRoll
from .cos import Cos
import numpy as np
from omegaconf.dictconfig import DictConfig


def make(config: DictConfig):
    
    match config.name:
        case "torus":
            env = Torus(
                A=np.array(config.A),
                B=np.array(config.B),
                Ns=np.array(config.Ns),
                No=np.array(config.No),
                horizon=config.horizon,
                render_mode="rgb_array",
                periodic=config.periodic,
                radius1=config.radius1,
                radius2=config.radius2,
                rotated=config.rotated,
            )
        case "circle":
            env = Circle(
                A=np.array(config.A),
                B=np.array(config.B),
                Ns=np.array(config.Ns),
                No=np.array(config.No),
                horizon=config.horizon,
                render_mode="rgb_array",
                periodic=config.periodic,
            )
        case "swiss_roll":
            env = SwissRoll(
                A=np.array(config.A),
                B=np.array(config.B),
                Ns=np.array(config.Ns),
                No=np.array(config.No),
                horizon=config.horizon,
                render_mode="rgb_array",
                periodic=config.periodic, 
            )
        case "cos":
            env = Cos(
                A=np.array(config.A),
                B=np.array(config.B),
                Ns=np.array(config.Ns),
                No=np.array(config.No),
                horizon=config.horizon,
                render_mode="rgb_array",
                periodic=config.periodic, 
            )
        case _:
            raise ValueError(f"env {config.name} not found!")
    return env