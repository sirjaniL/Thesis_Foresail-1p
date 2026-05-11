import numpy as np
from typing import List
from determination.ukf import UKF, UKFConfig
from determination_module import Determination
from utils.quaternion import Quaternion, Vector
from scipy.spatial.transform import Rotation
import csv


#Attitude error calculation see eq. 54 in thesis
def attitude_error_angle(q_true: Quaternion, q_est: Quaternion) -> float:
    if q_true is None:
        print("Missing quaternion, check that correct vectors.csv is provided.")
    else:
        q_rel = q_true * q_est.inverse() #quaternion error
        angle_rad = 2 * np.arccos(np.clip(abs(q_rel.q[3]), -1.0, 1.0))  #transformed to principal euler angle in radians from scalar part of the quaternion
    return np.degrees(angle_rad)

#Angular velocity error calculation see eq. 170 in thesis
def ang_vel_error(omega_true: Vector, omega_ukf: Vector) -> float:
    if omega_true is None:
        print("Missing angular velocity, check that correct vectors.csv is provied.")
    else:
        ang_vel_sub_sqr = 0.0
        for i in range(len(omega_true)):
            ang_vel_sub_sqr += (omega_true[i]-omega_ukf[i])**2          
    return np.sqrt(ang_vel_sub_sqr) #Return normed angular velocity error

def get_reference_vectors(file_name): #Reference vectors from a file

    true_quat = []  #True attitude vectors
    sun_v = []      #Sun vectors
    sun_v_ref = []  #Reference sun vectors
    mag = []        #Magnetic field vectors
    mag_ref = []    #Magnetic field reference attitude vectors
    gyro_rate = []  #Angular velocity vectors
    torque = []     #Torque vectors

    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
        #Convert strings to integers
            quats = np.array(list(map(float, row[:4])))
            sun_vector = np.array(list(map(float, row[4:7])))
            sun_vector_r = np.array(list(map(float, row[7:10])))
            mag_field = np.array(list(map(float, row[10:13])))
            mag_field_r = np.array(list(map(float, row[13:16])))
            gyro = np.array(list(map(float, row[16:19])))
            torq = np.array(list(map(float, row[19:22])))
            true_quat.append(quats)
            sun_v.append(sun_vector)
            sun_v_ref.append(sun_vector_r)
            mag.append(mag_field)
            mag_ref.append(mag_field_r)
            gyro_rate.append(gyro)
            torque.append(torq)

    sweep_vectors = [true_quat,                                         
                    sun_v, 
                    sun_v_ref,
                    mag,
                    mag_ref,
                    gyro_rate, 
                    torque] 
    
    return sweep_vectors
      
#Actual simulation run, which intakes saved vectors from Basilisk and runs them through
#Determination class after that executes the comparison between the true and estimated values
def run_simulation(config: UKFConfig, logger = None, 
                            attitudes_true = None, 
                            sun_vectors = None,
                            sun_vectors_ref = None,
                            mag_vectors = None,
                            mag_vectors_global = None,
                            ang_vel_vectors = None,
                            torque_vectors = None) -> float: 
     
    if attitudes_true is None:
        raise ValueError("[run_simulation] Error: 'attitudes_true' is None. Make sure it's passed from parameter_sweep_runner.")

    N = len(attitudes_true)

    #Initialize UKF and Determination class from determination_module.py
    ukf = UKF(config)
    #There is no need to run the whole simulation, we can just deliver required vectors to UKF
    det = Determination(timeStep=0.5, inertia_tensor=ukf.Iv, q_s_c=ukf.q_s_c, ukf=ukf) 
    errors: List[float] = []
    errors_ang: List[float] = []
    
    for t in range(N - 1):
                
        ukf_output = None
        # Required vectors provided to ukf
        try:
            ukf_output = det.execute(
                torque=torque_vectors[t],
                sun_vector=sun_vectors[t],
                sun_vector_ref=sun_vectors_ref[t],
                mag_field=mag_vectors[t],
                mag_field_ref=mag_vectors_global[t],
                gyro_rate=ang_vel_vectors[t]
            )
        except Exception as e:
            print(f"[t={t}] UKF execution failed: {e}", flush=True)
            raise  
        #Rotate quaternion as in control_module.py
        ukf_output[0:4] = Rotation.from_quat(ukf_output[0:4]).as_quat(canonical=True, scalar_first=False)

        #Comparison of estimated values to true values to acquire errors                    
        q_est = Quaternion(q=ukf_output[:4])
        q_true = Quaternion(q=attitudes_true[t])  

        omega_ukf = ukf_output[4:7]
        omega_true = ang_vel_vectors[t]

        error_ang = ang_vel_error(omega_true, omega_ukf)
        errors_ang.append(error_ang)

        error_deg = attitude_error_angle(q_true, q_est)
        errors.append(error_deg)
        errors_interm = np.mean(errors)#for intermediate logging of the mean error, used only in debugging
        
        #Separated log file option if needed if debug logger set else than None in parameter_sweep_runnerTR.py
        #Practically ukf_output always not None
        if logger and ukf_output is not None:
            logger.log_step(t, errors_interm,
                            ukf_output, 
                            attitudes_true[t],
                            sun_vectors[t],
                            sun_vectors_ref[t],
                            mag_vectors[t],
                            mag_vectors_global[t],
                            ang_vel_vectors[t],
                            torque_vectors[t])
            
    #Only write file after simulation finishes
    if logger:
        logger.write_csv()
    
    return np.mean(errors), np.mean(errors_ang)

