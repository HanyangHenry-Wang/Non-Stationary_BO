from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from dataclasses import dataclass
import math
from torch.distributions import Kumaraswamy
import torch
import numpy as np
from botorch.utils.sampling import draw_sobol_samples

# def get_initial_points(bounds,num,seed=0):
    
    
#     np.random.seed(seed)
    
#     bounds = bounds.T
#     dim = bounds.shape[0]
#     train_x = torch.tensor(np.random.uniform(bounds[:, 0], bounds[:, 1],size=(num,dim)))
    
#     return train_x

def get_initial_points(bounds,num,device,dtype,seed=0):
    
        train_x = draw_sobol_samples(
        bounds=bounds, n=num, q=1,seed=seed).reshape(num,-1).to(device, dtype=dtype)
        
        return train_x


######################### TuRBO #########################################
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state






###################### Warping ##############################
def obj(X,k,fun):
    X_warp = k.icdf(X)
    return fun(X_warp)


