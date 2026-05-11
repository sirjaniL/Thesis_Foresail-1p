from run_control_simulation import run_basilisk_sim
import numpy as np
import pandas as pd
from decimal import Decimal
from determination.ukf import UKF
from config_loader import covariance_logic, covariancemode
bias = True 

#Increments start value with step gradually until stop value is reached, this way one parameter set is built
def float_range(start, stop, step):
    while start <= stop:
        yield start
        start *= step

#Digits counter after decimal
def count_decimal(number):
    
    d = Decimal(str(number))
    return abs(d.as_tuple().exponent)

if __name__ == "__main__":

    ukf = UKF
    #Basilisk sim run for aggregating required vectors
    sim     = run_basilisk_sim(config_file="tests/test_ukf_001.toml", use_sil=False, seed=0) 
    cm      = sim.control                           #Making instance "cm" to ControlModule
    a_q     = cm.attitude_sim_q_log                 #True attitude quaternion from basilisk model

    reference_vectors = cm.vlogger.multiple_series  #Reference vectors needed calculating true attitude from UKF referenced to control_module.py
    s_v     = reference_vectors[0]                  #Sun vectors
    s_v_r   = reference_vectors[1]                  #Reference sun vectors
    m_v     = reference_vectors[2]                  #Magnetic field vectors
    m_v_g   = reference_vectors[3]                  #Global magnetic field vectors
    a_v_v   = reference_vectors[4]                  #Angular velocity vectors
    t_v     = reference_vectors[5]                  #Torque vectors

    vec_all = [a_q, s_v, s_v_r, m_v, m_v_g, a_v_v, t_v]

    names   = ["a_q", "s_v", "s_v_r", "m_v", "m_v_g", "a_v_v", "t"]

    #Component labels
    comp_labels = {
        "a_q": ["x","y","z","w"],
        "s_v": ["x","y","z"],
        "s_v_r": ["x","y","z"],
        "m_v": ["x","y","z"],
        "m_v_g": ["x","y","z"],
        "a_v_v": ["x","y","z"],
        "t": ["tx", "ty", "tz"],
    }

    #Reference vectors generation
    frames = []
    for name, data in zip(names, vec_all):
        M = np.asarray(data, dtype=float)
        labels = comp_labels.get(name, None)
        if not labels or len(labels) != M.shape[1]:
            labels = [str(i) for i in range(M.shape[1])]
        cols = [f"{name}_{c}" for c in labels]
        frames.append(pd.DataFrame(M, columns=cols))

    df = pd.concat(frames, axis=1)

    df.to_csv("vectors.csv", index=False)

############## PARAMETER COMBINATIONS ##############

    logic = covariance_logic("tests/test_ukf_001.toml") #Covariance logic mode read from a file
    mode = logic.mode #mode variable from covariance_logic function saved to mode variable
    smallest_param = float("inf")
    if mode == covariancemode.R_DYNAMIC:
        if bias:
            params_ranges = { 
                    
            "Q1_attitude": (1e-8, 1e-5),
            "Q1_angular_rate": (1e-9, 1e-6),
            "Q1_mag_bias": (1e-7, 1e-3),
            "Q1_gyro_bias": (1e-7, 1e-3),
            "R1_sun_vector": (1e-4, 1),
            "R1_gyroscope": (1e-5, 1e-1),
            "R1_magnetometer": (1e-6, 1e-4),
            "R_no_gyroscope": (1e-4,1), #= dynamic gyroscope measurement noise covariance
            "R_no_magnetometer": (1e-5,1e-4),  #=dynamic magnetometer measurement noise covariance
            }
            print("############# with bias! #############")
        else:
            params_ranges = { 
            
            
            "Q1_attitude": (1e-8, 1e-5),
            "Q1_angular_rate": (1e-8, 1e-5),
            "R1_sun_vector": (1e-10, 1e-6),
            "R1_gyroscope": (1e-3, 1),
            "R1_magnetometer": (1e-5, 1e-2),
            "R_no_gyroscope": (1e-4,1), #= dynamic gyroscope measurement noise covariance
            "R_no_magnetometer": (1e-8,1e-4), #=dynamic magnetometer measurement noise covariance
            
            }
            print("############# without bias! #############")
        print("----in ref_vectors: R_DYNAMIC----")

    elif mode == covariancemode.Q_R_DYNAMIC:
        if bias:
            params_ranges = {

            "Q1_attitude": (1e-6, 1e-4),
            "Q1_angular_rate": (1e-6, 1e-4),
            "Q1_mag_bias": (1e-8, 1e-4),
            "Q1_gyro_bias": (1e-8, 1e-4),
            "R1_sun_vector": (1e-8, 1e-6),
            "R1_gyroscope": (1e-3, 1e-1),
            "R1_magnetometer": (1e-6, 1e-4),
            "R_no_gyroscope": (1e-1,10), #= dynamic gyroscope measurement noise covariance
            "R_no_magnetometer": (1e-6,1e-4), #=dynamic magnetometer measurement noise covariance
            "Q_attitude_dyn": (1e-7, 1e-5),
            "Q_angular_rate_dyn": (1e-7, 1e-5),
            }  
            print("############# with bias! #############")
        else:
            params_ranges = {
                
            "Q1_attitude": (1e-8, 1e-5),
            "Q1_angular_rate": (1e-8, 1e-5),
            "R1_sun_vector": (1e-10, 1e-7),
            "R1_gyroscope": (1e-3, 1),
            "R1_magnetometer": (1e-5, 1e-3),
            "R_no_gyroscope": (1e-3,1), #= dynamic gyroscope measurement noise covariance
            "R_no_magnetometer": (1e-8,1e-5), #=dynamic magnetometer measurement noise covariance
            "Q_attitude_dyn": (1e-9, 1e-5),
            "Q_angular_rate_dyn": (1e-9, 1e-5),
                  
            }
            print("############# without bias! #############")
        print("----in ref_vectors: Q_R_DYNAMIC----")

    else:
        if bias:
            params_ranges = {

            "Q1_attitude": (1e-6, 1e-1),
            "Q1_angular_rate": (1e-7, 1e-2),
            "Q1_mag_bias": (1e-3, 1e-2),
            "Q1_gyro_bias": (1e-3, 1e-2),
            "R1_sun_vector": (1e-9, 1e-2),
            "R1_gyroscope": (1e-7, 1e-2),
            "R1_magnetometer": (1e-7, 1e-2),

            }
            print("############# with bias! #############")
        else:
            params_ranges = {

            "Q1_attitude": (1e-6, 1e-1),
            "Q1_angular_rate": (1e-7, 1e-2),
            "R1_sun_vector": (1e-9, 1e-2),
            "R1_gyroscope": (1e-7, 1e-2),
            "R1_magnetometer": (1e-7, 1e-2),

            }
            print("############# without bias! #############")
        print("----in ref_vectors: DEFAULT----")
    #arrays are formed from parameter sets
    arrays = np.array([
        (v[0], v[1]) for _, v in params_ranges.items()])
    #find the smallest value
    for _, v in params_ranges.items():
        if v[0] < smallest_param:
            smallest_param = v[0]
    #solves the number of digits after decimal
    rounder = count_decimal(smallest_param) # smallest value needs to be found to keep the intervals intact later in rounding


    #set is created below which has parameters boundaries including their intervals for example
    #"Q1_attitude" = [1e-6, 1e-5 ,1e-4, 1e-3]
    #"Q1_angular_rate" = [1e-5, 1e-4 ,1e-3, 1e-2]
    #               .
    #               .
    #               .
    #               etc.
    sets = []  
    for j in range(len(arrays)):
        new = []
        for i in float_range(arrays[j][0], arrays[j][1], 10): #iterates over range of floats
            new.append(round(i, rounder)) # without rounding very small float-point errors are appearing
        sets.append(new)
    
    #Grid of combination sets generated, all possible parameter combinations of the parameter set is created
    if mode == covariancemode.R_DYNAMIC:
        if bias:
            combinations = [(q1_att, q1_ang, q1_mag, q1_gyro, r1_sun, r1_gyro, r1_mag, r1_no_gyro, r1_no_mag) for q1_att in sets[0] for q1_ang in sets[1] 
                        for q1_mag in sets[2] for q1_gyro in sets[3] for r1_sun in sets[4] 
                        for r1_gyro in sets[5] for r1_mag in sets[6] for r1_no_gyro in sets[7] for r1_no_mag in sets[8]]
                        
        else:
            combinations = [(q1_att, q1_ang, r1_sun, r1_gyro, r1_mag, r1_no_gyro, r1_no_mag) for q1_att in sets[0] for q1_ang in sets[1] 
                        for r1_sun in sets[2] for r1_gyro in sets[3] for r1_mag in sets[4] for r1_no_gyro in sets[5] for r1_no_mag in sets[6]]

    elif mode == covariancemode.Q_R_DYNAMIC:
        if bias:
            combinations = [(q1_att, q1_ang, q1_mag, q1_gyro, r1_sun, r1_gyro, r1_mag, r1_no_gyro, r1_no_mag, q1_att_dyn, q1_ang_dyn) for q1_att in sets[0] for q1_ang in sets[1] 
                        for q1_mag in sets[2] for q1_gyro in sets[3] for r1_sun in sets[4] 
                        for r1_gyro in sets[5] for r1_mag in sets[6] for r1_no_gyro in sets[7] for r1_no_mag in sets[8] for q1_att_dyn in sets[9] for q1_ang_dyn in sets[10]]
        
        else:
            combinations = [(q1_att, q1_ang, r1_sun, r1_gyro, r1_mag, r1_no_gyro, r1_no_mag, q1_att_dyn, q1_ang_dyn) for q1_att in sets[0] for q1_ang in sets[1] 
                        for r1_sun in sets[2] for r1_gyro in sets[3] for r1_mag in sets[4] for r1_no_gyro in sets[5] for r1_no_mag in sets[6] for q1_att_dyn in sets[7] for q1_ang_dyn in sets[8]]   
        
    else:
        if bias:    
            combinations = [(q1_att, q1_ang, q1_mag, q1_gyro, r1_sun, r1_gyro, r1_mag) for q1_att in sets[0] for q1_ang in sets[1] 
                        for q1_mag in sets[2] for q1_gyro in sets[3] for r1_sun in sets[4] 
                        for r1_gyro in sets[5] for r1_mag in sets[6]]
        
        else:
            combinations = [(q1_att, q1_ang, r1_sun, r1_gyro, r1_mag) for q1_att in sets[0] for q1_ang in sets[1]
                        for r1_sun in sets[2] for r1_gyro in sets[3] for r1_mag in sets[4]]
        
        print("count of different combinations. ", len(combinations))
    
    #Covariance sets are written to a file line by line
    with open("combinations.txt", "w") as f:
        #Running index is added before every line created for identifying every combination
        for idx, line in enumerate(combinations, start=1):
            f.write(f"{idx} " + ' '.join(map(str, line)) + '\n')
    