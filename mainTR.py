from determination.ukf import UKFConfig
from simulation_runnerTR import get_reference_vectors
from parameter_sweep_runnerTR import parameter_sweep_run
import sys
from config_loader import covariance_logic, covariancemode

bias = True

if __name__ == "__main__":
    
    base_config = UKFConfig()
    
    combination_v = list(map(float, sys.argv[1:])) #reads command line argumets and convert the to floats and saves them to variable
    file_name = "vectors.csv" #This is obtained from ref_vectors.py
    trial = int(combination_v[0]) #first element from combinations array has to be int, since it is used as id for combinations file   
    
    logic = covariance_logic("tests/test_ukf_001.toml")#covariance logic mode read from a file
    mode = logic.mode    

    if mode == covariancemode.R_DYNAMIC:
        if bias:
            param_ranges = { 
                "Q_attitude": combination_v[1],
                "Q_angular_rate": combination_v[2],
                "Q_mag_bias": combination_v[3],
                "Q_gyro_bias": combination_v[4],
                "R_sun_vector": combination_v[5],
                "R_gyroscope": combination_v[6],
                "R_magnetometer": combination_v[7],
                "R_no_gyroscope": combination_v[8],
                "R_no_magnetometer": combination_v[9], 
                "P0_attitude": 0.01722, 
            }
        else:
            param_ranges = { 
                "Q_attitude": combination_v[1],
                "Q_angular_rate": combination_v[2],
                "R_sun_vector": combination_v[3],
                "R_gyroscope": combination_v[4],
                "R_magnetometer": combination_v[5],
                "R_no_gyroscope": combination_v[6],
                "R_no_magnetometer": combination_v[7], 
                "P0_attitude": 0.01722,
            }
    elif mode == covariancemode.Q_R_DYNAMIC: 
           
        if bias:
            param_ranges = { 
            
                "Q_attitude": combination_v[1],
                "Q_angular_rate": combination_v[2],
                "Q_mag_bias": combination_v[3],
                "Q_gyro_bias": combination_v[4],
                "R_sun_vector": combination_v[5],
                "R_gyroscope": combination_v[6],
                "R_magnetometer": combination_v[7],                    
                "R_no_gyroscope": combination_v[8],
                "R_no_magnetometer": combination_v[9],
                "Q_attitude_dyn": combination_v[10],
                "Q_angular_rate_dyn": combination_v[11],
                "P0_attitude": 0.01722,
            }
        else:    
            param_ranges = { 

                "Q_attitude": combination_v[1],
                "Q_angular_rate": combination_v[2],
                "R_sun_vector": combination_v[3],
                "R_gyroscope": combination_v[4],
                "R_magnetometer": combination_v[5],
                "R_no_gyroscope": combination_v[6],
                "R_no_magnetometer": combination_v[7],
                "Q_attitude_dyn": combination_v[8],
                "Q_angular_rate_dyn": combination_v[9],
                "P0_attitude": 0.01722,
            }
    else:
        if bias:

            param_ranges = { 
                "Q_attitude": combination_v[1],
                "Q_angular_rate": combination_v[2],
                "Q_mag_bias": combination_v[3],
                "Q_gyro_bias": combination_v[4],
                "R_sun_vector": combination_v[5],
                "R_gyroscope": combination_v[6],
                "R_magnetometer": combination_v[7],
                "P0_attitude": 0.01722,
            }
        else:
        
            param_ranges = { 
                "Q_attitude": combination_v[1],
                "Q_angular_rate": combination_v[2],
                "R_sun_vector": combination_v[3],
                "R_gyroscope": combination_v[4],
                "R_magnetometer": combination_v[5],
                "P0_attitude": 0.01722,
            }
    ref_vectors = get_reference_vectors(file_name) #Reference vectors obtained from a file saved to another variable   
    #initiates the parameter sweep simulation
    try:
        mean_error, mean_error_ang = parameter_sweep_run(base_config, param_ranges, ref_vectors = ref_vectors, trial_no=trial) 

    except Exception as e:
        print(f"Run failed with exception: {e}")
        sys.exit(1)
    best_params = param_ranges
    best_error = mean_error
    best_error_ang = mean_error_ang
    
    print("\n==== Final Summary ====")        
    print("Best configuration:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.10g}")
    print(f"Mean error: {best_error:.14f}")
    print(f"Mean angular error : {best_error_ang:.14f}")
    