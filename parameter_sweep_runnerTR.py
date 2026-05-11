from typing import Callable, Dict, List, Optional
import numpy as np
from determination.ukf import UKFConfig
from simulation_runnerTR import run_simulation
import copy
from vectors_logger import UKFLogger

#UKF parameter in the UKFConfig class is overriden with parameters that are in the parameter set
def override_config(config: UKFConfig, overrides: Dict[str, float]) -> UKFConfig:
    for key, value in overrides.items():#loop through parameters
        if hasattr(config, key):#if parameter key found...
            setattr(config, key, value)#...replace with new value of the key-value pair
    return config

def parameter_sweep_run(
    base_config: UKFConfig, #ukf parameters in the algortihm that are overriden for simulation
    param_overrides: Dict[str, float], #This is the key value pair got from param_ranges, overrides the actual parameter in UKF
    ref_vectors: Optional[List[np.ndarray]] = None, #Vectors obtained from Basilisk environment
    trial_no = None #used for debugging
    ) -> float:
    
    errors: List[float] = [] 
    errors_ang: List[float] = []

    if ref_vectors is None:
        raise ValueError("ref_vectors must be provided to parameter_sweep_run")
 
    if 0:
        #For debugging, initiates debug logger in simulation_runnerTR.py, for Triton simulation this needs to be off!
        #otherwise it will blow up the storage space, csv:s are large
        debug_logger = UKFLogger(trial_id=trial_no, seed=0, params=param_overrides) 
    else:
        debug_logger = None
    config_to_log = override_config(copy.deepcopy(base_config), param_overrides)#replace current parameter values in UKF
    #Initiate parameter sweep simulation, which compares the truth model to UKF with new parameter set
    error, error_ang = run_simulation(config_to_log, logger=debug_logger,                                
                        attitudes_true=ref_vectors[0],      #true attitude vectors
                        sun_vectors = ref_vectors[1],       #sun vectors
                        sun_vectors_ref = ref_vectors[2],   #reference sun vectors
                        mag_vectors = ref_vectors[3],       #magnetic field vectors
                        mag_vectors_global = ref_vectors[4],#magnetic field reference attitude vectors
                        ang_vel_vectors = ref_vectors[5],   #angular velocity vectors
                        torque_vectors = ref_vectors[6])    #torque vectors
    errors.append(error) #attitude error
    errors_ang.append(error_ang) #angular velocity error

    return np.mean(errors), np.mean(errors_ang)    
