
import numpy as np
from typing import *
import os
import tomllib

from .flae import FLAE
from .triad import TRIAD
from utils.quaternion import *
from config_loader import covariance_logic, covariancemode

class UKFConfig:

    estimate_bias: bool

    rk_steps: int # RK4 integration time [sec]
    rk_substeps: int  # RK4 sub-steps

    #  Sigma point spreading factor 
    alpha: float # default value: 1e-3
    beta: float # Optimally 2 for Gaussian distribution
    kapa: float # default value = 0

    # Principal moments of the inertia
    Iv: Vector # [kgm^2]

    # Quaternion from spacecraft frame to control frame
    q_s_c: Quaternion

    # State noise covariance
    Q_attitude: float     # For quaternion which magnitude is 1
    Q_angular_rate: float # rad/s
    Q_mag_bias: float     # nT
    Q_gyro_bias: float    # [(rad/s)^2]
    Q_attitude_dyn: float
    Q_angular_rate_dyn: float

    # Measurement noise covariances
    R_sun_vector: float   # unit vector
    R_gyroscope: float    # [(rad/s)^2]
    R_magnetometer: float
    R_no_sun_vector: float
    R_no_magnetometer: float
    R_no_gyroscope: float

    # Initial state covariances
    P0_attitude: float
    P0_angular_rate: float
    P0_mag_bias: float    # [T^2]
    P0_gyro_bias: float   # [(rad/s)^2]
    P0_solved_attitude: float
    P0_solved_angular_rate: float

    initial_mag_bias: Vector
    initial_gyro_bias: Vector

    def __init__(self,
            estimate_bias=True,  
            rk_steps=1,
            rk_substeps=4,
            alpha=1e-3, 
            beta=2.0,
            kapa=0.0,
            Iv=np.array([0.046994, 0.047877, 0.010571]), 
            q_s_c=Quaternion.identity(),
            
            Q_attitude = 1e-05, 
            Q_angular_rate = 1e-05, 
            Q_mag_bias = 1e-7, 
            Q_gyro_bias = 1e-06, 
            R_sun_vector = 0.00001, 
            R_gyroscope = 7.7106e-06, 
            R_magnetometer = 168100, 
            R_no_sun_vector = 0.5,
            R_no_gyroscope = 7.7106e-07, #Used as dynamic covariance
            R_no_magnetometer = 1681,  #Used as dynamic covariance         
            Q_attitude_dyn = 0.5e-05,
            Q_angular_rate_dyn = 0.5e-05,
            P0_attitude = 0.5,
            P0_angular_rate = 0.01,
            P0_mag_bias = 0.1,
            P0_gyro_bias = 0.01,
            P0_solved_attitude = 0.00190265,
            P0_solved_angular_rate = 0.0001,

            initial_mag_bias=np.array([5, 1, -3])*1e2,
            initial_gyro_bias = np.array([0.001745329, 0.001745329, 0.001745329])
        ):
        self.estimate_bias = estimate_bias
        self.rk_steps = rk_steps
        self.rk_substeps = rk_substeps
        self.R_sun_vector = R_sun_vector
        self.R_gyroscope = R_gyroscope
        self.R_magnetometer = R_magnetometer
        self.R_no_sun_vector = R_no_sun_vector
        self.R_no_magnetometer = R_no_magnetometer
        self.R_no_gyroscope = R_no_gyroscope
        self.alpha = alpha
        self.beta = beta
        
        self.kapa = kapa
        self.Iv = Iv
        self.q_s_c = q_s_c
        self.Q_attitude = Q_attitude
        self.Q_angular_rate = Q_angular_rate
        self.Q_attitude_dyn = Q_attitude_dyn
        self.Q_angular_rate_dyn = Q_angular_rate_dyn
        self.Q_mag_bias = Q_mag_bias
        self.Q_gyro_bias = Q_gyro_bias
        self.P0_attitude = P0_attitude
        self.P0_angular_rate = P0_angular_rate
        self.P0_mag_bias = P0_mag_bias
        self.P0_gyro_bias = P0_gyro_bias
        self.P0_solved_attitude = P0_solved_attitude
        self.P0_solved_angular_rate = P0_solved_angular_rate
        self.initial_mag_bias = initial_mag_bias
        self.initial_gyro_bias = initial_gyro_bias
      
    
    @classmethod
    def from_toml(cls, filename: str, section: str = "ukf"):
        print(f"Loading UKF config from: {os.path.basename(filename)}")

        with open(filename, "rb") as f:
            data = tomllib.load(f)

        if section not in data:
            raise ValueError(f"Missing section [{section}] in {filename}")

        cfg_data = data[section]

        #Handle vector fields 
        vector_fields = {
            "Iv", "initial_mag_bias", "initial_gyro_bias"
        }
        quaternion_fields = {"q_s_c"}

        kwargs = {}
        for key, value in cfg_data.items():
            if key in vector_fields:
                kwargs[key] = np.array(value)
            elif key in quaternion_fields:
                kwargs[key] = Quaternion.identity()  
            else:
                kwargs[key] = value

        return cls(**kwargs)    

def calculate_error_quaternion(q1: Quaternion, q2: Quaternion) -> np.ndarray:
    """ Calculate an error quaternion between two quaternions. """
    return (Quaternion(q1) * Quaternion(q2).inverse()).vector
    
def expand_error_quaternion(error: Vector, q: Quaternion) -> Quaternion:
    """ Expand an error quaternion to full one. """
    assert error.size == 3
    dot = np.dot(error, error)
    if dot <= 1:
        return Quaternion(v=error, s=np.sqrt(1.0 - dot)) * q
    else:
        raise ValueError("Error quaternion is too large")
        print("problem")
        norm = np.sqrt(1.0 + dot)
        return Quaternion(v=error / norm , s=1.0 / norm) * q

class UKF:
    """
    Unscented Kalman Filter
    """

    # Dimensions
    N_STATES: int # Number of states (Quaternion, angular velocities, mag bias and gyro bias)
    N_STATES_ERR: int # Number of states in error representation (error form quaternion, angular velocities, mag bias and gyro bias)
    N_SIGMA_POINTS: int # Number of sigma points in unscented transform
    N_MEAS: int # Size of measurement vector (3xSun, 3xmag, 3xgyro)


    initialized: bool

    cfg: UKFConfig

    # State estimate (posteori) vector [N_STATES]
    x_posteriori: Vector

    # State covariance (posteori) matrix [N_STATES_ERR, N_STATE_ERR]
    P_posteriori: Vector

    # Measurement vector
    meas: Vector
    sun_available: bool
    mag_available: bool
    gyro_available: bool

    # Applied torque vector
    torque: Vector

    # Reference data calculated from ephemeris
    reference_available: bool
    sun_reference: Vector
    mag_reference: Vector

    q_s_c: Quaternion # Quaternion from spacecraft frame to control frame

    Iv: Vector # Principal moments of the inertia [kgm^2]

    # Cached sigma point weigths
    lambdaa: float
    w_m_0: float
    w_c_0: float
    w_i: float

    # General temporary vectors and matrices for intermediate results
    v_temp1: Vector
    v_temp2: Vector
    m_temp1: np.ndarray
    m_temp2: np.ndarray

    # A-priori state vector
    x_priori: Vector

    # Sigma points full state (intermediate)
    # array of arrays
    sigma_states: npt.NDArray[np.float64] 

    # Sigma points error state (intermediate)
    # array of arrays
    sigma_states_err: npt.NDArray[np.float64] 

    # Sigma points measurement estimate (intermediate)
    # array of arrays
    sigma_meas_est: npt.NDArray[np.float64] 

    # A-priori covariance matrix (intermediate)
    P_priori: np.ndarray
    
    # A-priori measurement covariance matrix (intermediate)
    P_zz: np.ndarray

    # Kalman gain matrix (intermediate)
    K: np.ndarray
    
    # temporary vectors for intermediate results in Runge-Kutta integration
    rk4_xn: Vector
    rk4_k: Vector
    rk4_sum: Vector

    # Covariance checker
    cov_checker_eigen: Vector

   
    def __init__(self, cfg: Optional[UKFConfig] = None, use_optimized: bool = False):
        """
        Initialize UKF object

        """

        if cfg is None:
            if use_optimized:
                base_dir = os.path.dirname(__file__)
                config_path = os.path.join(base_dir, "ukf_param_opti.toml")
                cfg = UKFConfig.from_toml(config_path)
                print("Using optimized UKF parameters from ukf_param_opti.toml")
            else:
                cfg = UKFConfig()  #Use default parameters defined in class
                print("Using default UKFConfig parameters (in ukf.py)")

 
        self.cfg = cfg

        self.N_MEAS = 3 + 3 + 3

        self.N_STATES = 4 + 3 + 3 + 3
        self.N_STATES_ERR = 3 + 3 + 3 + 3
    
        self.N_SIGMA_POINTS = 2 * self.N_STATES_ERR + 1
        
        self.v_temp1 = np.zeros(self.N_STATES)
        self.v_temp2 = np.zeros(self.N_STATES)

        self.m_temp1 = np.zeros((self.N_STATES_ERR, self.N_STATES_ERR))
        self.m_temp2 = np.zeros((self.N_STATES_ERR, self.N_STATES_ERR))

        self.x_priori = np.zeros(self.N_STATES)

        self.P_priori = np.zeros((self.N_STATES_ERR, self.N_STATES_ERR))
        self.P_zz = np.zeros((self.N_MEAS, self.N_MEAS))
        self.K = np.zeros((self.N_STATES_ERR, self.N_MEAS))

        self.rk4_xn = np.zeros(4+3)
        self.rk4_k = np.zeros(4+3)
        self.rk4_sum = np.zeros(4+3)

        self.cov_checker_eigen = np.zeros(self.N_STATES)

        self.initialized = False

        self.meas = np.zeros(self.N_MEAS)
        self.x_posteriori = np.zeros(self.N_STATES)
        self.P_posteriori = np.zeros((self.N_STATES_ERR, self.N_STATES_ERR))

        self.sigma_states = [np.zeros(self.N_STATES) for _ in range(self.N_SIGMA_POINTS)]
        self.sigma_states_err = [np.zeros(self.N_STATES_ERR) for _ in range(self.N_SIGMA_POINTS)]
        self.sigma_meas_est = [np.zeros(self.N_MEAS) for _ in range(self.N_SIGMA_POINTS)]

        self.q_s_c = cfg.q_s_c
        self.Iv = cfg.Iv

        self.reset()

    def reset(self) -> None:
        """
        Reset estimator state
        """

        self.initialized = False

        self.lambdaa = self.cfg.alpha*self.cfg.alpha *(self.N_STATES_ERR + self.cfg.kapa) - self.N_STATES_ERR

        # Precalculate constants used by the unscented transformation
        self.w_m_0 = self.lambdaa / (self.N_STATES_ERR + self.lambdaa)
        self.w_c_0 = self.w_m_0 + (1.0 - self.cfg.alpha*self.cfg.alpha + self.cfg.beta)
        self.w_i = 0.5 / (self.N_STATES_ERR + self.lambdaa)

        self.x_posteriori = np.array([
            *Quaternion.identity(), # 4 Attitude quaternion
            0, 0, 0, # 3 Angular velocities
            *self.cfg.initial_mag_bias, # 3 Magnetometer bias
            *self.cfg.initial_gyro_bias, # 3 Gyroscope bias
        ], dtype=np.float64)
        self.P_posteriori = np.diag([
            self.cfg.P0_attitude, self.cfg.P0_attitude, self.cfg.P0_attitude, # 3 x3 
            self.cfg.P0_angular_rate, self.cfg.P0_angular_rate, self.cfg.P0_angular_rate, # 3 x3 
            self.cfg.P0_mag_bias, self.cfg.P0_mag_bias, self.cfg.P0_mag_bias, # 3 x3
            self.cfg.P0_gyro_bias, self.cfg.P0_gyro_bias, self.cfg.P0_gyro_bias, # 3 x3
        ]) 

        self.meas = np.zeros(self.N_MEAS)
        self.torque = np.zeros(3)

        self.sun_available = False
        self.mag_available = False
        self.gyro_available = False
        self.reference_available = False

        self.sun_reference = np.zeros(3)
        self.mag_reference = np.zeros(3)


    def system_function_internal(self, X, torque: np.ndarray) -> np.ndarray:
        # inputs
        q = Quaternion(q=X[0:4]) # Attitude quaternion
        w =  np.array([X[4], X[5], X[6]], dtype=np.float64) # Angular velocity (omega)
        # outputs
        qd = Quaternion(q) # Quaternion derivate
        wd = np.array([0, 0, 0], dtype=np.float64) # Angular acceleration

        if 0:
            
            I = self.cfg.Iv 

            omega_hat = Quaternion(v=w, s=0)
            qd = 0.5 * (omega_hat * q).q
            
            output = np.concatenate([
                qd.q,
                np.linalg.inv(I) @ (-np.cross(w, I @ w) + torque)
            ])
            return output

        else:
            # Calculation of the attitude rotation using a skew-symmetric matrix from the angular velocities 

            #ω in body, right–multiplication
            qd.x = 0.5*(               w[2]*q.y - w[1]*q.z + w[0]*q.w)
            qd.y = 0.5*( -w[2]*q.x                + w[0]*q.z + w[1]*q.w)
            qd.z = 0.5*(w[1]*q.x - w[0]*q.y                + w[2]*q.w)
            qd.w = 0.5*(-w[0]*q.x - w[1]*q.y - w[2]*q.z               )
            
            # Calculate angular acceleration
            wd[0] = (w[1] * w[2] * (self.cfg.Iv[1] - self.cfg.Iv[2]) + torque[0]) / self.cfg.Iv[0]
            wd[1] = (w[0] * w[2] * (self.cfg.Iv[2] - self.cfg.Iv[0]) + torque[1]) / self.cfg.Iv[1]
            wd[2] = (w[0] * w[1] * (self.cfg.Iv[0] - self.cfg.Iv[1]) + torque[2]) / self.cfg.Iv[2]

            
            return np.array([qd.x, qd.y, qd.z, qd.w, wd[0], wd[1], wd[2]], dtype=np.float64)
    
    def rk4_internal(self, X: np.ndarray, torque: np.ndarray, rk_steps: int, rk_substeps: int) -> np.ndarray:
        """
        Runge-Kutta 4th order integration
        Args:
            X: State vector
            torque: Torque vector
            step_size: Step size
            rk_steps: Number of RK4 steps
        Returns:
            New state vector
        """

        h = rk_steps / rk_substeps

        for i in range(rk_substeps):
            # 1st step: k1 = f(ukf.rk4_xn)
            self.rk4_k = self.system_function_internal(X, torque)
            self.rk4_sum = self.rk4_k

            # 2nd step: k2 = f(ukf.rk4_xn + 0.5 * h * k1)
            self.rk4_xn = X + 0.5 * h * self.rk4_k
            self.rk4_k = self.system_function_internal(self.rk4_xn, torque)
            self.rk4_sum += 2.0 * self.rk4_k

            # 3rd step: k3 = f(ukf.rk4_xn + 0.5 * h * k2)
            self.rk4_xn = X + 0.5 * h * self.rk4_k
            self.rk4_k = self.system_function_internal(self.rk4_xn, torque)
            self.rk4_sum += 2.0 * self.rk4_k

            # 4th step: k4 = f(ukf.rk4_xn + h * k3)
            self.rk4_xn = X + h * self.rk4_k
            self.rk4_k = self.system_function_internal(self.rk4_xn, torque)
            self.rk4_sum += self.rk4_k

            X += (h / 6.0) * self.rk4_sum

            # Normalize the attitude quaternion
            X[0:4] = Quaternion(q=X[0:4]).normalized().q

        return X
    
    def check_covariance_matrix(self, matrix_in: np.ndarray) -> bool:
        """
        Check if covariance matrix is positive definite
        Args:
            matrix_in: Covariance matrix to check
        Returns:
            True if covariance matrix is positive definite, False otherwise
        """
        self.cov_checker_eigen = np.zeros_like(matrix_in)

        # caculate eigenvalues
        self.cov_checker_eigen = np.linalg.eigvals(matrix_in)

        # check if all eigenvalues are positive
        if np.all(self.cov_checker_eigen > 0):
            return True
        else:
            return False

    def set_sun_vector_measurement(self, sun_vector: np.ndarray) -> None:
        """
        Set magnetometer field measurement
        Args:
            field: Magnetic field measurement in spacecraft frame
        """

        self.meas[0:3] = self.cfg.q_s_c.rotate(sun_vector)
        self.sun_available = True


    def set_magnetometer_measurement(self, field: np.ndarray) -> None:
        """
        Set magnetometer field measurement
        Args:
            field: Magnetic field measurement in spacecraft frame in nanoTeslas
        """
        
        self.meas[3:6] = self.cfg.q_s_c.rotate(field) 
        self.mag_available = True


    def set_gyroscope_measurement(self, angular_velocities: np.ndarray) -> None:
        """
        Set gyroscope measurement
        Args:
            angular_velocities: Angular rate measurement in spacecraft frame
        """
        self.meas[6:9] = self.cfg.q_s_c.rotate(angular_velocities)
        self.gyro_available = True


    def set_reference(self, sun_vector: np.ndarray, magnetic_field: np.ndarray) -> None:
        """
        Set reference data
        Args:
            sun_vector: Reference sun vector in ECI inertial frame
            magnetic_field: Reference magnetic field vector in ECI inertial frame
        """
        self.sun_reference = sun_vector.copy() 
        self.mag_reference = magnetic_field.copy() 
        self.reference_available = True


    def set_torque(self, torque: np.ndarray) -> None:
        """
        Set applied torque
        Args:
            torque: Applied torque in spacecraft body frame.
        """
        self.torque = self.cfg.q_s_c.rotate(torque)

    
    def calculate_error_quaternion_internal(self, q: Quaternion, q_ref: Quaternion) -> np.ndarray:
        """
        Calculate error quaternion between two quaternions.
        Args:
            q: Quaternion to calculate error from
            q_ref: Reference quaternion
        Returns:
            Error quaternion
        """       
        x = -q.w * q_ref.x + q.x * q_ref.w - q.y * q_ref.z + q.z * q_ref.y
        y = -q.w * q_ref.y + q.x * q_ref.z + q.y * q_ref.w - q.z * q_ref.x
        z = -q.w * q_ref.z - q.x * q_ref.y + q.y * q_ref.x + q.z * q_ref.w
        
        error = np.array([x, y, z], dtype=np.float64)

        return error
    

    def expand_error_quaternion_internal(self, error: Vector, q: Quaternion) -> Quaternion:
        """
        Expand an error quaternion to full one.
        Args:
            error: Error quaternion
            q: Reference quaternion
        Returns:
            Expanded quaternion
        """
        assert error.size == 3
        dot = np.dot(error, error)
        temp = Quaternion.identity()
        if dot <= 1:
            temp.x = error[0]
            temp.y = error[1]
            temp.z = error[2]
            temp.w = np.sqrt(1.0 - dot)
        else:
            raise ValueError("Error quaternion is too large")

        return temp * q 
        
    def execute(self, attitude_sc: Quaternion, angular_velocity_sc: Vector, step_size: int, covariance_check=False) -> np.ndarray:
        """
        Execute the UKF algorithm
        Args:
            attitude_s: Attitude quaternion in spacecraft frame
            angular_velocity_s: Angular velocity in spacecraft frame
            step_size: Step size
        Returns:
            attitude_s: Attitude quaternion in spacecraft frame
            angular_velocity_s: Angular velocity in spacecraft frame
        """

        if self.reference_available == False:
            raise RuntimeError("Cannot run without reference data")

        measurements =  self.sun_available + self.mag_available + self.gyro_available
        if measurements < 2:
            raise RuntimeError("Not enough measurements")

        # If the UKF is not initialized execute the first-run logic.
        if self.initialized == False:

            attitude_c = Quaternion(q=self.x_posteriori[0:4])
            angular_rate_c = self.x_posteriori[4:7]
                        
            # Initialize posteriori attitude estimate using initialization algorithm
            # if sun vector magnetic field measurement are available for rough
            # attitude estimate.
            if self.sun_available and self.mag_available:
                sun_measurement = self.meas[0:3] # In control frame
                mag_measurement = self.meas[3:6].copy() # Since self.meas is numpy array, this declaration creates a view, not a copy.
                                
                mag_measurement[0] -= self.cfg.initial_mag_bias[0]
                mag_measurement[1] -= self.cfg.initial_mag_bias[1]
                mag_measurement[2] -= self.cfg.initial_mag_bias[2]
                
                if 1:
                    
                    attitude_c = TRIAD(self.sun_reference, self.mag_reference, sun_measurement, mag_measurement)
                else:
                    
                    if 1:
                        print("ENTERING FLAE")
                        normed_mag_measurement = mag_measurement / np.linalg.norm(mag_measurement)
                        normed_sun_measurement = sun_measurement / np.linalg.norm(sun_measurement)
                        normed_sun_reference = self.sun_reference / np.linalg.norm(self.sun_reference)
                        normed_mag_reference = self.mag_reference / np.linalg.norm(self.mag_reference)
                        attitude_c = FLAE(normed_sun_reference, normed_mag_reference, normed_sun_measurement, normed_mag_measurement)
                    else:    
                        attitude_c = FLAE(self.sun_reference, self.mag_reference, sun_measurement, mag_measurement)
                self.x_posteriori[0:4] = attitude_c.q

                # Set P_posteori
                self.P_posteriori[0,0] = self.cfg.P0_solved_attitude
                self.P_posteriori[1,1] = self.cfg.P0_solved_attitude
                self.P_posteriori[2,2] = self.cfg.P0_solved_attitude

            # Set current angular rate to gyro measurement
            if self.gyro_available:
                gyro_measurement = self.meas[6:9]
                angular_rate_c[0] = gyro_measurement[0]
                angular_rate_c[1] = gyro_measurement[1]
                angular_rate_c[2] = gyro_measurement[2]

                angular_rate_c[0] -= self.cfg.initial_gyro_bias[0] 
                angular_rate_c[1] -= self.cfg.initial_gyro_bias[1] 
                angular_rate_c[2] -= self.cfg.initial_gyro_bias[2] 

                self.x_posteriori[4:7] = angular_rate_c 

                self.P_posteriori[3,3] = self.cfg.P0_solved_angular_rate
                self.P_posteriori[4,4] = self.cfg.P0_solved_angular_rate
                self.P_posteriori[5,5] = self.cfg.P0_solved_angular_rate

            # Calculate attitude in spacecraft frame and return it
            q_c_s = self.cfg.q_s_c.inverse()
            attitude_s = q_c_s * Quaternion(q=attitude_c.q)
            Quaternion(q=self.x_posteriori[0:4])
            
            angular_rate_s = q_c_s.rotate(angular_rate_c) 

            self.torque = np.zeros(3)
            self.meas = np.zeros(self.N_MEAS)

            self.sun_available = False
            self.mag_available = False
            self.gyro_available = False
            self.reference_available = False

            self.initialized = True
            # Return full state vector
            estimate_state = np.concatenate([attitude_s.q, angular_rate_s, self.x_posteriori[7:], np.array([0, 0, 0, 0 ])])
            return estimate_state

        ############################################
        # Predict steps
        ############################################

        # 1.1: error sigma points
        cholesky = self.m_temp2
        # resize matrix
        np.resize(cholesky, (self.N_STATES_ERR, self.N_STATES_ERR))
        cholesky = (self.N_STATES_ERR + self.lambdaa) * self.P_posteriori

        cholesky = np.linalg.cholesky(cholesky)

        for i in range(self.N_STATES_ERR):
            for j in range(i, self.N_STATES_ERR):
                if i == j:
                    pass
                else:
                    cholesky[i,j] = 0.0

        # 1.2: full sigma points
        error_state = self.v_temp1
        #resize vector
        np.resize(error_state, (self.N_STATES_ERR, 1))
        self.x_priori = np.zeros(self.N_STATES)

        for s in range(self.N_SIGMA_POINTS):

            full_state = self.sigma_states[s]

            if s == 0:
                error_state = np.zeros(self.N_STATES_ERR)
            elif s < self.N_STATES_ERR +1:
                error_state = -cholesky[s - 1,:]
            else:
                error_state = cholesky[s - 1 - self.N_STATES_ERR,:]
            
            # Copy error quaternion, expand it 4D and translate to control frame
            full_state[0:4] = self.expand_error_quaternion_internal(error=error_state[0:3], q=Quaternion(q=self.x_posteriori[0:4])).q

            # Add sigma point error to other states normally
            for j in range(4, self.N_STATES):
                full_state[j] = self.x_posteriori[j] + error_state[j-1]
                
            # 1.4: Numerically propagate each sigma point (state priori)
            full_state[0:7] = self.rk4_internal(full_state[0:7], self.torque, step_size, self.cfg.rk_substeps)
            self.sigma_states[s] = full_state
            
            # 1.5 A priori state estimate
            w_m = self.w_m_0 if s == 0 else self.w_i
            self.x_priori += w_m * full_state

        x_err_priori = self.v_temp1
        np.resize(x_err_priori, (self.N_STATES_ERR, 1))
        x_err_priori = np.zeros(self.N_STATES_ERR)
        
        for s in range(self.N_SIGMA_POINTS):

            full_state = self.sigma_states[s]
            error_state = self.sigma_states_err[s]

            # 1.6: Full to error state
            # Rotate state vector back to make it error quaternion
            error_state[0:3] = self.calculate_error_quaternion_internal(Quaternion(q=full_state[0:4]), Quaternion(q=self.x_priori[0:4]))
            for i in range(3, self.N_STATES_ERR): 
                error_state[i] = full_state[i+1] - self.x_priori[i+1]

            self.sigma_states_err[s] = error_state

            # 1.7: Mean error state
            w_m = self.w_m_0 if s == 0 else self.w_i
            x_err_priori += w_m * error_state

        # 1.8: A priori covariance
        error_residual = self.v_temp2
        np.resize(error_residual, (self.N_STATES_ERR, 1))

        self.P_priori = np.zeros((self.N_STATES_ERR, self.N_STATES_ERR))

        for s in range(self.N_SIGMA_POINTS):
            
            error_residual = (self.sigma_states_err[s] - x_err_priori)
    
            w_c = self.w_c_0 if s == 0 else self.w_i

            # P_priori += w_c * (error_residual * error_residual.T)
            for i in range(self.N_STATES_ERR):
                for j in range(i, self.N_STATES_ERR):
                    tmp = w_c * error_residual[i] * error_residual[j]
                    self.P_priori[i,j] += tmp
                    if i != j:
                        self.P_priori[j,i] += tmp

        #Covariance logic fetched for dynamic covariances
        logic = covariance_logic("tests/test_ukf_001.toml")
        mode = logic.mode

        if mode == covariancemode.Q_R_DYNAMIC:
            
            qc = self.cfg.Q_attitude if self.sun_available else self.cfg.Q_attitude_dyn
            self.P_priori[0,0] += qc
            self.P_priori[1,1] += qc
            self.P_priori[2,2] += qc

            qc = self.cfg.Q_angular_rate if self.sun_available else self.cfg.Q_angular_rate_dyn

            self.P_priori[3,3] += qc
            self.P_priori[4,4] += qc
            self.P_priori[5,5] += qc

            if self.cfg.estimate_bias:

                self.P_priori[6,6] += self.cfg.Q_mag_bias
                self.P_priori[7,7] += self.cfg.Q_mag_bias
                self.P_priori[8,8] += self.cfg.Q_mag_bias

                self.P_priori[9,9] += self.cfg.Q_gyro_bias
                self.P_priori[10,10] += self.cfg.Q_gyro_bias
                self.P_priori[11,11] += self.cfg.Q_gyro_bias
             
        else:

            self.P_priori[0,0] += self.cfg.Q_attitude
            self.P_priori[1,1] += self.cfg.Q_attitude
            self.P_priori[2,2] += self.cfg.Q_attitude

            self.P_priori[3,3] += self.cfg.Q_angular_rate
            self.P_priori[4,4] += self.cfg.Q_angular_rate
            self.P_priori[5,5] += self.cfg.Q_angular_rate


            self.P_priori[6,6] += self.cfg.Q_mag_bias
            self.P_priori[7,7] += self.cfg.Q_mag_bias
            self.P_priori[8,8] += self.cfg.Q_mag_bias

            self.P_priori[9,9] += self.cfg.Q_gyro_bias
            self.P_priori[10,10] += self.cfg.Q_gyro_bias
            self.P_priori[11,11] += self.cfg.Q_gyro_bias


        if covariance_check:
            if not self.check_covariance_matrix(self.P_priori):
                raise RuntimeError("Covariance matrix is not positive definite")
            

        # 2.3: Estimate measurement for every sigma point
        meas_priori = self.v_temp2
        np.resize(meas_priori, (self.N_MEAS, 1))
        meas_priori = np.zeros(self.N_MEAS)
        
        for s in range(self.N_SIGMA_POINTS):
            sigma_state = self.sigma_states[s]
            sigma_meas = self.sigma_meas_est[s]

            sigma_attitude = Quaternion(q=sigma_state[0:4])

            # Predict sun vector measurement if real measurement is available
            sun_measurement = sigma_meas[0:3]
            if (self.sun_available):
                
                sun_measurement[0] = self.sun_reference[0]
                sun_measurement[1] = self.sun_reference[1]
                sun_measurement[2] = self.sun_reference[2]
                # Rotate sun reference vector from inertial frame to control frame using sigma points estimate
                sun_measurement = Quaternion(q=sigma_attitude.q).rotate(sun_measurement)
                sun_measurement /= np.linalg.norm(sun_measurement) 

            else:
                sun_measurement[0] = 0.0
                sun_measurement[1] = 0.0
                sun_measurement[2] = 0.0

            sigma_meas[0:3] = sun_measurement
            # Predict magnetometer measurement if actual actual measurement available
            mag_measurement = sigma_meas[3:6]
            if self.mag_available:
                
                mag_measurement[0] = self.mag_reference[0]
                mag_measurement[1] = self.mag_reference[1]
                mag_measurement[2] = self.mag_reference[2]
                
                # Rotate magnetic field reference vector from inertial frame to control frame using sigma points estimate
                mag_measurement = Quaternion(q=sigma_attitude.q).rotate(mag_measurement)
                if self.cfg.estimate_bias:

                    mag_measurement[0] += sigma_state[7] 
                    mag_measurement[1] += sigma_state[8]
                    mag_measurement[2] += sigma_state[9]


            else:
                mag_measurement[0] = 0.0
                mag_measurement[1] = 0.0
                mag_measurement[2] = 0.0

            sigma_meas[3:6] = mag_measurement

            # Predict gyroscope measurement if actual actual measurement available
            gyro_measurement = sigma_meas[6:9]
            if self.gyro_available:
                gyro_measurement[0] = sigma_state[4]
                gyro_measurement[1] = sigma_state[5]
                gyro_measurement[2] = sigma_state[6]
               
                if self.cfg.estimate_bias:
                    gyro_measurement[0] += sigma_state[10]
                    gyro_measurement[1] += sigma_state[11]
                    gyro_measurement[2] += sigma_state[12]

            else:
                gyro_measurement[0] = 0.0
                gyro_measurement[1] = 0.0
                gyro_measurement[2] = 0.0

            sigma_meas[6:9] = gyro_measurement

            
            self.sigma_meas_est[s] = sigma_meas
            w_m = self.w_m_0 if s == 0 else self.w_i
            meas_priori += w_m * self.sigma_meas_est[s]

        # 2.6: Calculate covariances 
        measurement_error = self.v_temp1
        np.resize(measurement_error, (self.N_MEAS, 1))

        P_xz = self.m_temp1
        np.resize(P_xz, (self.N_STATES_ERR, self.N_MEAS))
        P_xz = np.zeros((self.N_STATES_ERR, self.N_MEAS))
        self.P_zz = np.zeros((self.N_MEAS, self.N_MEAS))

        for s in range(self.N_SIGMA_POINTS):

            state_err_residual = self.sigma_states_err[s] - x_err_priori 
            measurement_error = self.sigma_meas_est[s] - meas_priori

            w_c = self.w_c_0 if s == 0 else self.w_i

            # P_zz += w_c * measurement_error * measurement_error.T
            for i in range(self.N_MEAS):
                for j in range(i, self.N_MEAS):
                    tmp = w_c * measurement_error[i] * measurement_error[j]
                    self.P_zz[i,j] += tmp
                    if i != j:
                        self.P_zz[j,i] += tmp

            # P_xz += w_c * state_err_residual * measurement_residual.T
            for i in range(self.N_STATES_ERR):
                for j in range(self.N_MEAS):
                    P_xz[i,j] += w_c * state_err_residual[i] * measurement_error[j]


        # Add measurement noise covariances 
        
        if mode == covariancemode.R_DYNAMIC or mode == covariancemode.Q_R_DYNAMIC:
            
            r = self.cfg.R_sun_vector if self.sun_available else self.cfg.R_sun_vector 
            self.P_zz[0,0] += r
            self.P_zz[1,1] += r
            self.P_zz[2,2] += r
                        
            r = self.cfg.R_magnetometer if self.sun_available else self.cfg.R_no_magnetometer
            self.P_zz[3,3] += r
            self.P_zz[4,4] += r
            self.P_zz[5,5] += r

            r = self.cfg.R_gyroscope if self.sun_available else self.cfg.R_no_gyroscope
            self.P_zz[6,6] += r
            self.P_zz[7,7] += r
            self.P_zz[8,8] += r

        else:

            r = self.cfg.R_sun_vector if self.sun_available else self.cfg.R_no_sun_vector 
            self.P_zz[0,0] += r
            self.P_zz[1,1] += r
            self.P_zz[2,2] += r

            r = self.cfg.R_magnetometer if self.mag_available else self.cfg.R_no_magnetometer
            self.P_zz[3,3] += r
            self.P_zz[4,4] += r
            self.P_zz[5,5] += r

            r = self.cfg.R_gyroscope if self.gyro_available else self.cfg.R_no_gyroscope
            self.P_zz[6,6] += r
            self.P_zz[7,7] += r
            self.P_zz[8,8] += r
            
        if covariance_check:
            if not self.check_covariance_matrix(self.P_zz):
                raise RuntimeError("Covariance matrix is not positive definite")

        # 2.7: Calculate Kalman gain
        iP_zz = self.m_temp2
        np.resize(iP_zz, (self.N_MEAS, self.N_MEAS))
        iP_zz = np.linalg.inv(self.P_zz)

        # K = P_xz * P_zz^-1
        self.K = P_xz @ iP_zz

        # 2.8: Calculate error state
        measurement_residual = self.v_temp1
        np.resize(measurement_residual, (self.N_MEAS, 1))
        measurement_residual = self.meas - meas_priori 
        x_posteriori_err = self.v_temp2
        x_posteriori_err = self.K @ measurement_residual

        # Update the biases only when both sun and magnetic field vector measurements are available.
        if self.cfg.estimate_bias and not(self.sun_available and self.mag_available and self.gyro_available):
            for i in range(6, self.N_STATES_ERR):
                x_posteriori_err[i] = 0.0

        # 2.9: Expand quaternion
        self.x_posteriori[0:4] = self.expand_error_quaternion_internal(error=x_posteriori_err[0:3], q=Quaternion(q=self.x_priori[0:4])).q
        
        # 2.10: Calculate full state
        for i in range(4, self.N_STATES):
            self.x_posteriori[i] = self.x_priori[i] + x_posteriori_err[i-1]

        # 2.11: Update a posteriori covariance
        inter = self.m_temp1
        np.resize(inter, (self.N_STATES_ERR, self.N_MEAS))
        asd = self.m_temp2  
        np.resize(asd, (self.N_STATES_ERR, self.N_MEAS))

        # P_posteriori = P_priori - K * P_zz * K.T
        asd = self.K @ self.P_zz
        # matrix transpose multiply
        inter = asd @ self.K.T
        self.P_posteriori = self.P_priori - inter 
        
        if covariance_check:
            if not self.check_covariance_matrix(self.P_posteriori):
                raise RuntimeError("Covariance matrix is not positive definite")

        # 2.12 Rotate and output
        attitude_c = Quaternion(q=self.x_posteriori[0:4])
        angular_rate_c = self.x_posteriori[4:7]
        
        q_c_s = self.cfg.q_s_c.inverse()
        attitude_s = q_c_s * Quaternion(q=attitude_c.q) 
        angular_velocity_s = q_c_s.rotate(angular_rate_c)

        # Reset measurements
        self.meas = np.zeros(self.N_MEAS)
        # Reset applied torque vector
        self.torque = np.zeros(3)

        self.sun_available = False
        self.mag_available = False
        self.gyro_available = False
        self.reference_available = False

        # Calculate variances for attitude, angular velocity, magnetic field bias and gyro bias
        # Variance is calculated as the average of the diagonal elements of the covariance matrix
        quaternion_variance = (self.P_posteriori[0,0] + self.P_posteriori[1,1] + self.P_posteriori[2,2]) / 3.0
        angular_velocity_variance = (self.P_posteriori[3,3] + self.P_posteriori[4,4] + self.P_posteriori[5,5]) / 3.0
        magnetic_field_bias_variance = (self.P_posteriori[6,6] + self.P_posteriori[7,7] + self.P_posteriori[8,8]) / 3.0
        gyro_bias_variance = (self.P_posteriori[9,9] + self.P_posteriori[10,10] + self.P_posteriori[11,11]) / 3.0

        variances = np.array([
            quaternion_variance, # Attitude quaternion variance
            angular_velocity_variance, # Angular velocity variance
            magnetic_field_bias_variance, # Magnetic field bias variance
            gyro_bias_variance, # Gyro bias variance
        ])

        # Return full state vector
        estimate_state = np.concatenate([attitude_s.q, angular_velocity_s, self.x_posteriori[7:], variances])
        return estimate_state
