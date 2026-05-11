import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

from utils.quaternion import Quaternion, Vector

def plot(filename):
    with open(filename, "r") as f:
        data = f.read().split('\n')
        
        values = [[] for x in data[0].split(';')]
        for line in data:
            if(line == ""):
                continue
            vals = line.split(';')
            for i,val in enumerate(vals):
                values[i].append(float(val))

        values = [np.array(x[1:]) for x in values]
        print(f"Loaded {len(values)} values from {filename}")

        """
        print(
            *estimated_ukf_attitude, # estimated attitude from true ukf, 4
            *estimated_ukf_angular_velocity, # estimated angular velocity from true ukf, 3
            *estimated_ukf_mag_bias, # magnetometer bias, 3
            *estimated_ukf_gyro_bias, # gyroscope bias, 3
            quaternion_variance, # attitude variance, 1
            angular_velocity_variance, # angular velocity variance, 1
            magnetic_field_bias_variance, # magnetic field bias variance, 1
            gyro_bias_variance, # gyroscope bias variance, 1

            *self.attitude_sim_q, # sil attitude quaternion in inertial frame, 4
            *attitude_q, # true attitude quaternion in inertial frame, 4

            *self.angular_velocity_est_sim, # sil angular velocity in inertial frame, 3
            *angular_velocity, # true angular velocity in inertial frame, 3

            *self.mag_bias_sim, # sil magnetometer bias in inertial frame, 3
            *self.mag_bias, # magnetometer bias in inertial frame, 3
            *self.gyro_bias_sim, # sil gyroscope bias in inertial frame, 3
            *self.gyro_bias, # gyroscope bias in inertial frame, 3

            self.attitude_variance_sim, # attitude variance
            self.angular_velocity_variance_sim, # angular velocity variance
            self.mag_bias_variance_sim, # magnetometer bias variance
            self.gyro_bias_variance_sim, # gyroscope bias variance

            self.attitude_timestamp, # sil attitude timestamp, 1

            *self.angular_velocity_sim, # sil angular velocity in inertial frame, 3
            self.angular_velocity_sim_timestamp, # sil angular velocity timestamp, 1

            *self.sun_vector_sim, # sil sun vector in inertial frame, 3
            self.sun_vector_sim_timestamp, # sil sun vector timestamp, 1

            *sun_vector, # sun vector in inertial frame, 3
            *self.sun_vector_ref_sim, # sil sun vector reference in inertial frame, 3
            self.sun_vector_ref_sim_timestamp, # sil sun vector reference timestamp, 1
            *sun_vector_ref, # sun vector reference in inertial frame, 3

            *self.magnetic_field_sim, # sil magnetic field in inertial frame, 3
            self.magnetic_field_sim_timestamp, # sil magnetic field timestamp, 1

            *local_magnetic_field, # local magnetic field in inertial frame, 3
            *self.global_magnetic_field_sim, # sil global magnetic field in inertial frame, 3
            self.global_magnetic_field_sim_timestamp, # sil global magnetic field timestamp, 1
            *global_magnetic_field, # global magnetic field in inertial frame, 3

            *self.satellite_position_sim, # sil spacecraft position in inertial frame, 3
            *satellite_position, # true spacecraft position in inertial frame, 3
            *self.satellite_velocity_sim, # sil spacecraft velocity in inertial frame, 3
            *satellite_velocity, # true spacecraft velocity in inertial frame, 3
            self.satellite_pos_vel_sim_timestamp, # sil spacecraft position timestamp, 1

            *self.control_sim, # sil control in inertial frame, 3
            self.control_sim_timestamp, # sil control timestamp, 1
            *self.control, # true control in inertial frame, 3

            *self.torque_sim, # sil torque in inertial frame, 3
            self.torque_sim_timestamp, # sil torque timestamp, 1
            *self.torque, # true control torque in inertial frame, 3

            *self.psd_sun_vectors_sim.flatten(), # sil sun sensor results (flattened array of 6 sensors), 6*4 = 24
            self.psd_sun_vectors_sim_timestamp, # sil sun sensor timestamp, 1
            *sun_sensor_result, # sun sensor results (flattened array of 6 sensors), 6*4 = 24

            currentSimDateTime.timestamp(), # current simulation time in seconds
            sep=';', 
            file=self.data
        )
        """


        seconds = values[-1]
        #print(seconds)
        start_time = seconds[0]
        #print(f"Start time: {start_time} s")
        seconds = seconds - start_time  # Normalize time to start at 0

        # true kalman filter
        true_ukf_attitude_x = values[0]  # True attitude quaternion from UKF, from inertial frame to body frame
        true_ukf_attitude_y = values[1]
        true_ukf_attitude_z = values[2]
        true_ukf_attitude_q = values[3]

        true_ukf_angular_velocity_x = values[4] * (180/np.pi)  # Convert from rad/s to deg/s
        true_ukf_angular_velocity_y = values[5] * (180/np.pi)
        true_ukf_angular_velocity_z = values[6] * (180/np.pi)
        
        true_ukf_mag_bias_x = values[7]#*1e9  # Magnetometer bias from UKF
        true_ukf_mag_bias_y = values[8]#*1e9
        true_ukf_mag_bias_z = values[9]#*1e9
        mean_ukf_mag_bias_x = np.mean(true_ukf_mag_bias_x)
        mean_ukf_mag_bias_y = np.mean(true_ukf_mag_bias_y)
        mean_ukf_mag_bias_z = np.mean(true_ukf_mag_bias_z)

        true_ukf_gyro_bias_x = values[10] * (180/np.pi)  # Gyroscope bias from UKF
        true_ukf_gyro_bias_y = values[11] * (180/np.pi)
        true_ukf_gyro_bias_z = values[12] * (180/np.pi)

        quaternion_variance_true = values[13]
        angular_velocity_variance = values[14]
        magnetic_field_bias_variance = values[15]
        gyro_bias_variance = values[16]


        # Attitude quaternions
        attitude_x_sim = values[17] # SIL attitude quaternion from inertial frame to body frame
        attitude_y_sim = values[18] # For reason or another these values are not plotted for the first values, to verify this: enable attitude_q print control_module.py(where declared first time)
        attitude_z_sim = values[19]
        attitude_q_sim = values[20]
        
        attitude_x_true = values[21] # Basilisk attitude quaternion from inertial frame to body frame
        attitude_y_true = values[22]
        attitude_z_true = values[23]
        attitude_q_true = values[24]

        attitude_est = []
        attitude_true_ukf = []
        attitude_true = []

        for i in range(len(values[0])):
            vector = np.array([attitude_x_sim[i], attitude_y_sim[i], attitude_z_sim[i]])
            attitude_est.append(Quaternion(s=attitude_q_sim[i], v=vector))

            vector = np.array([true_ukf_attitude_x[i], true_ukf_attitude_y[i], true_ukf_attitude_z[i]])
            attitude_true_ukf.append(Quaternion(s=true_ukf_attitude_q[i], v=vector))

            vector = np.array([attitude_x_true[i], attitude_y_true[i], attitude_z_true[i]])
            attitude_true.append(Quaternion(s=attitude_q_true[i], v=vector))

        # angular velocity
        angular_velocity_x_est_sim = np.degrees(values[25]) # Convert from rad/s to deg/s
        angular_velocity_y_est_sim = np.degrees(values[26]) # in body frame
        angular_velocity_z_est_sim = np.degrees(values[27])

        angular_velocity_x_true = np.degrees(values[28]) # Convert from rad/s to deg/s #See why we need the minuses
        angular_velocity_y_true = np.degrees(values[29]) # in body frame
        angular_velocity_z_true = np.degrees(values[30])

        # magnetic field bias
        mag_bias_x_est_sim = values[31]
        mag_bias_y_est_sim = values[32]
        mag_bias_z_est_sim = values[33]

        mag_bias_x_true = values[34]#*1e9
        mag_bias_y_true = values[35]#*1e9
        mag_bias_z_true = values[36]#*1e9

        # gyro bias
        gyro_bias_x_est_sim = np.degrees(values[37])  # Convert from rad/s to deg/s
        gyro_bias_y_est_sim = np.degrees(values[38])
        gyro_bias_z_est_sim = np.degrees(values[39])

        gyro_bias_x_true = np.degrees(values[40])
        gyro_bias_y_true = np.degrees(values[41])
        gyro_bias_z_true = np.degrees(values[42])

        mag_bias_rate_error_true = []
        for i in range(len(mag_bias_x_true)):
            mag_bias_rate_error_true.append(
                np.sqrt(
                    (mag_bias_x_true[i] - true_ukf_mag_bias_x[i])**2 +
                    (mag_bias_y_true[i] - true_ukf_mag_bias_y[i])**2 +
                    (mag_bias_z_true[i] - true_ukf_mag_bias_z[i])**2
                )
            )        

        gyro_bias_rate_error_true = []
        for i in range(len(mag_bias_x_true)):
            gyro_bias_rate_error_true.append(
                np.sqrt(
                    (gyro_bias_x_true[i] - true_ukf_gyro_bias_x[i])**2 +
                    (gyro_bias_y_true[i] - true_ukf_gyro_bias_y[i])**2 +
                    (gyro_bias_z_true[i] - true_ukf_gyro_bias_z[i])**2
                )
            )        

        # variances
        quaternion_variance_sim = values[43]
        angular_velocity_variance_sim = values[44]
        magnetic_field_bias_variance_sim = values[45]
        gyro_bias_variance_sim = values[46]


        attitude_sim_timestamp = values[47] # Assuming this is the timestamp for attitude
        start_time = attitude_sim_timestamp[0]  # Normalize time to start at 0
        attitude_sim_timestamp = attitude_sim_timestamp - start_time  # Normalize time to start at 0

        # gyro sensor
        gyro_sensor_x_sim = values[48]* (180/np.pi)  # Assuming this is the gyro sensor x in body frame
        gyro_sensor_y_sim = values[49]* (180/np.pi)  # Assuming this is the gyro sensor y in body frame
        gyro_sensor_z_sim = values[50]* (180/np.pi)  # Assuming this is the gyro sensor z in body frame

        gyro_sim_timestamp = values[51] # Assuming this is the timestamp for gyro sensor
        start_time = gyro_sim_timestamp[0]  # Normalize time to start at 0
        gyro_sim_timestamp = gyro_sim_timestamp - start_time  # Normalize time to start at 0

        # sun vector
        sun_vector_x_sim = values[52] # in body frame
        sun_vector_y_sim = values[53]
        sun_vector_z_sim = values[54]

        sun_vector_sim_timestamp = values[55]  # Assuming this is the timestamp for sun vector
        start_time = sun_vector_sim_timestamp[0]
        sun_vector_sim_timestamp = sun_vector_sim_timestamp - start_time

        sun_vector_x_true = values[56] # in body frame
        sun_vector_y_true = values[57]
        sun_vector_z_true = values[58]

               
        sun_vector_ref_sim_timestamp = values[62]  # Assuming this is the timestamp for sun vector reference
        start_time = sun_vector_ref_sim_timestamp[0]
        sun_vector_ref_sim_timestamp = sun_vector_ref_sim_timestamp - start_time

        magnetometer_sim_timestamp = values[69]  # Assuming this is the timestamp for magnetometer
        start_time = magnetometer_sim_timestamp[0]  # Normalize time to start at 0
        magnetometer_sim_timestamp = magnetometer_sim_timestamp - start_time  # Normalize time to start at 0

        # global magnetic field reference
        global_magnetic_field_x_sim = values[73] # in inertial frame
        global_magnetic_field_y_sim = values[74]
        global_magnetic_field_z_sim = values[75]

        global_magnetic_field_sim_timestamp = values[76] # Assuming this is the timestamp for global magnetic field
        start_time = global_magnetic_field_sim_timestamp[0]  # Normalize time to start at 0
        global_magnetic_field_sim_timestamp = global_magnetic_field_sim_timestamp - start_time  # Normalize time to start at 0


        satellite_pos_vel_timestamp = values[92] # Assuming this is the timestamp for satellite position and velocity
        start_time = satellite_pos_vel_timestamp[0]  # Normalize time to start at 0
        satellite_pos_vel_timestamp = satellite_pos_vel_timestamp - start_time  # Normalize time to start at 0

        # control signals
        control_sim_timestamp = values[96]  # Assuming this is the timestamp for control signals
        start_time = control_sim_timestamp[0]  # Normalize time to start at 0
        control_sim_timestamp = control_sim_timestamp - start_time  # Normalize time to start at 0

        torque_sim_timestamp = values[103]  # Assuming this is the timestamp for torques
        start_time = torque_sim_timestamp[0]  # Normalize time to start at 0
        torque_sim_timestamp = torque_sim_timestamp - start_time  # Normalize time to start at 0

        # sun sensors
        # x, y, z, intensity
        psd_sun_vector_sim = np.empty([6, 4, len(seconds)]) # in body frame
        psd_sun_vector_true = np.empty([6, 4, len(seconds)]) # in sensor frame
        for i in range(6):
            psd_sun_vector_sim[i] = [
                values[107+i*4],
                values[107+i*4+1],
                values[107+i*4+2],
                values[107+i*4+3][:len(seconds)]
            ]

        psd_sun_vector_sim_timestamp = values[131] # Assuming this is the timestamp for sun sensors
        start_time = psd_sun_vector_sim_timestamp[0]  # Normalize time to start at 0
        psd_sun_vector_sim_timestamp = psd_sun_vector_sim_timestamp - start_time  # Normalize time to start at 0

        for i in range(6):
            psd_sun_vector_true[i] = [
                values[132+i*4],
                values[132+i*4+1],
                values[132+i*4+2],
                values[132+i*4+3][:len(seconds)]
            ]

        sun_sensor_rotation_matrices = np.array([
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # +X
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # -X
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],  # +Y
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],  # -Y
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # +Z
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]  # -Z
        ])

        psd_sun_vector_sim_inertial = np.empty([6, 3, len(seconds)])
        psd_sun_vector_sim_body = np.empty([6, 3, len(seconds)])
        psd_sun_vector_true_inertial = np.empty([6, 3, len(seconds)])

        for i in range(6):
            # rotate it to inertial frame
            psd_sun_vector_sim_inertial[i] = psd_sun_vector_sim[i][:3]
            psd_sun_vector_sim_body[i] = sun_sensor_rotation_matrices[i].T @ psd_sun_vector_sim_inertial[i]
            psd_sun_vector_true_inertial[i] = sun_sensor_rotation_matrices[i] @ psd_sun_vector_true[i][:3]
            # then normalize each sun vector at each time step to unit length
            for t in range(psd_sun_vector_sim_inertial.shape[2]):
                norm_sim = np.linalg.norm(psd_sun_vector_sim_inertial[i][:, t])
                if norm_sim != 0:
                    psd_sun_vector_sim_inertial[i][:, t] /= norm_sim
                norm_true = np.linalg.norm(psd_sun_vector_true_inertial[i][:, t])
                if norm_true != 0:
                    psd_sun_vector_true_inertial[i][:, t] /= norm_true

        # get brightest sun sensor and use it as the sun vector in inertial frame
        # sun psd_sun_vector contains x, y, z, intensity for each sensor
        psd_sun_vector_sim_inertial_brightest = np.empty([3, len(seconds)])
        psd_sun_vector_true_inertial_brightest = np.empty([3, len(seconds)])
        # For each time step, find the brightest sensor and use its vector
        for t in range(len(seconds)):
            # Find brightest simulated sensor at time t
            brightest_sim_index = np.argmax([psd_sun_vector_sim[i][3][t] for i in range(6)])
            psd_sun_vector_sim_inertial_brightest[:, t] = psd_sun_vector_sim_inertial[brightest_sim_index][:, t]
            
            # Find brightest true sensor at time t
            brightest_true_index = np.argmax([psd_sun_vector_true[i][3][t] for i in range(6)])
            psd_sun_vector_true_inertial_brightest[:, t] = psd_sun_vector_true_inertial[brightest_true_index][:, t]

        """ PLOTTING """

        fig1, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.grid()
        ax1.set_title("Attitude")
        ax1.set_ylabel(r"Unit quaternion [$\mathbf{q}_{\mathcal{u}}$]")

        ax2.grid()
        ax2.set_title("Angular velocity")
        ax2.set_ylabel(r'$\mathbf{\omega} \ [^{\circ{}}/s$]')

        #ax1.plot(attitude_sim_timestamp, attitude_x_sim, label=r'$q_{1_t}$', ls=':', color='blue')
        #ax1.plot(attitude_sim_timestamp, attitude_y_sim, label=r'$q_{2_t}$', ls=':', color='orange')
        #ax1.plot(attitude_sim_timestamp, attitude_z_sim, label=r'$q_{3_t}$', ls=':', color='green')
        #ax1.plot(attitude_sim_timestamp, attitude_q_sim, label=r'$q_{4_t}$', ls=':', color='red')

        ax1.plot(seconds, true_ukf_attitude_x, label=r'$q_{1_e}$', color='blue')
        ax1.plot(seconds, true_ukf_attitude_y, label=r'$q_{2_e}$', color='orange')
        ax1.plot(seconds, true_ukf_attitude_z, label=r'$q_{3_e}$', color='green')
        ax1.plot(seconds, true_ukf_attitude_q, label=r'$q_{4_e}$', color='red')

        ax1.plot(seconds, attitude_x_true, label=r'$q_{1_t}$', ls=':', color='blue')
        ax1.plot(seconds, attitude_y_true, label=r'$q_{1_t}$', ls=':', color='orange')
        ax1.plot(seconds, attitude_z_true, label=r'$q_{1_t}$', ls=':', color='green')
        ax1.plot(seconds, attitude_q_true, label=r'$q_{1_t}$', ls=':', color='red')
        

        # norm of angular velocity ukf
        angular_velocity_negated_true_ukf = np.sqrt(
            (angular_velocity_x_true-true_ukf_angular_velocity_x)**2 +
            (angular_velocity_y_true-true_ukf_angular_velocity_y)**2 +
            (angular_velocity_z_true-true_ukf_angular_velocity_z)**2
        )    
        ax2.plot(seconds, angular_velocity_negated_true_ukf, label=r'$\delta\mathbf{\omega}$', color='magenta')
        limit = 1 #deg/s
        ax2.fill_between(seconds, -limit, limit, color='black', alpha=0.3, label = rf'$\pm\,{limit}^\circ/s$')
        ax2.set_xlabel("Time (s)")
        #ax2.plot(attitude_sim_timestamp, angular_velocity_x_est_sim, label=r'$x_t$', ls=':', color='blue')
        #ax2.plot(attitude_sim_timestamp, angular_velocity_y_est_sim, label=r'$y_t$', ls=':', color='orange')
        #ax2.plot(attitude_sim_timestamp, angular_velocity_z_est_sim, label=r'$z_t$', ls=':', color='green')

        ax2.plot(seconds, true_ukf_angular_velocity_x, label=r'$\omega_{x_e}$', color='blue') #estimated
        ax2.plot(seconds, true_ukf_angular_velocity_y, label=r'$\omega_{y_e}$', color='orange') #estimated
        ax2.plot(seconds, true_ukf_angular_velocity_z, label=r'$\omega_{z_e}$', color='green') #estimated

        #ax2.plot(gyro_sim_timestamp, gyro_sensor_x_sim, label='x (sensor)', color='blue', ls='-.')
        #ax2.plot(gyro_sim_timestamp, gyro_sensor_y_sim, label='y (sensor)', color='orange', ls='-.')
        #ax2.plot(gyro_sim_timestamp, gyro_sensor_z_sim, label='z (sensor)', color='green', ls='-.')

        ax2.plot(seconds, angular_velocity_x_true, label=r'$\omega_{x_t}$', ls=':', color='blue') #true
        ax2.plot(seconds, angular_velocity_y_true, label=r'$\omega_{y_t}$', ls=':', color='orange') #true
        ax2.plot(seconds, angular_velocity_z_true, label=r'$\omega_{z_t}$', ls=':', color='green') #true

        # norm of angular velocity
        #angular_velocity_norm_true = np.sqrt(
        #    angular_velocity_x_true**2 +
        #    angular_velocity_y_true**2 +
        #    angular_velocity_z_true**2
        #)
        #ax2.plot(seconds, angular_velocity_norm_true, label='total (true)', ls=':', color='black')

        #Settling time
        index = [0,0,0]
        for i in range(len(angular_velocity_x_true)-1,-1,-1):
            if(index[0] == 0 and abs(angular_velocity_x_true[i]) > limit):
                index[0] = i

            if(index[1] == 0 and abs(angular_velocity_y_true[i]) > limit):
                index[1] = i

            if(index[2] == 0 and abs(angular_velocity_z_true[i]) > limit):
                index[2] = i

            if(min(index) > 0):
                break

        settle_seconds = seconds[max(index)]
        settle_minutes = int(settle_seconds/60)
        settle_hours = int(settle_minutes/60)
        
        settle_seconds %= 60
        settle_minutes %= 60

        settle_string = f'{settle_hours} h {settle_minutes} min {"{:.2f}".format(settle_seconds)} s'

        #ax2.axvline(x=seconds[max(index)], color='green', ls='--', label=f'Settle time')
        #ax2.text(seconds[max(index)], 0.5, f' t: {settle_string} min', transform=ax2.get_xaxis_text1_transform(0)[0])

        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        fig1.tight_layout()
        fig1.savefig("figures/att_ang.pdf", bbox_inches="tight")

        # compute attitude error in euler angles in degrees
        attitude_error = []
        attitude_error_true = []
        for i in range(len(attitude_est)):
            attitude_error.append(Quaternion.angle_between(attitude_true[i], attitude_est[i]))
            attitude_error_true.append(Quaternion.angle_between(attitude_true_ukf[i], attitude_est[i]))

            # remove ambiguity of 360 degrees
            if attitude_error[-1] > 180:
                attitude_error[-1] = 360 - attitude_error[-1]

            if attitude_error_true[-1] > 180:
                attitude_error_true[-1] = 360 - attitude_error_true[-1]
        mean_err = np.mean(attitude_error_true)
        
        #Uncomment these to write in a file for comparing different covariance logic sets
        #with open("figures/Comparable_errors/attitude_error_true_Q_R_dynamic.txt", "w") as f:
         #   for value in attitude_error_true:
          #      f.write(f"{value}\n")

        
        # Comment if required
        #attitude_error_true_default = []

        #with open("figures/Comparable_errors/attitude_error_true_default.txt", "r") as f:
        #    for line in f:
        #        attitude_error_true_default.append(float(line.strip()))

        #print(attitude_error_true_default[0], attitude_error_true_default[1])


        attitude_error_true_R_dynamic= []
        with open("figures/Comparable_errors/attitude_error_true_R_dynamic.txt", "r") as f:
            for line in f:  
                attitude_error_true_R_dynamic.append(float(line.strip()))

        print(attitude_error_true_R_dynamic[0], attitude_error_true_R_dynamic[1])

        attitude_error_true_Q_R_dynamic= []
        with open("figures/Comparable_errors/attitude_error_true_Q_R_dynamic.txt", "r") as f:
            for line in f:  
                attitude_error_true_Q_R_dynamic.append(float(line.strip()))

        print(attitude_error_true_Q_R_dynamic[0], attitude_error_true_Q_R_dynamic[1])

        print("attitude error mean: ", mean_err)
        #mean_err_default = np.mean(attitude_error_true_default)
        mean_err_R_dynamic = np.mean(attitude_error_true_R_dynamic)
        mean_err_Q_R_dynamic = np.mean(attitude_error_true_Q_R_dynamic)
        angular_rate_error = []
        angular_rate_error_true = []
        for i in range(len(angular_velocity_x_true)):
            angular_rate_error.append(
                np.sqrt(
                    (angular_velocity_x_true[i] - angular_velocity_x_est_sim[i])**2 +
                    (angular_velocity_y_true[i] - angular_velocity_y_est_sim[i])**2 +
                    (angular_velocity_z_true[i] - angular_velocity_z_est_sim[i])**2
                )
            )
            angular_rate_error_true.append(
                np.sqrt(
                    (angular_velocity_x_true[i] - true_ukf_angular_velocity_x[i])**2 +
                    (angular_velocity_y_true[i] - true_ukf_angular_velocity_y[i])**2 +
                    (angular_velocity_z_true[i] - true_ukf_angular_velocity_z[i])**2
                )
            )
        fig2, (ax5, ax6, ax7, ax75) = plt.subplots(4, sharex=True)
        # errors
        ax5.grid()
        ax5.set_title("Errors")
        ax5.set_ylabel("Error (deg)")
        #ax5.plot(seconds, attitude_error, label='Attitude error (est)', color='blue')
        ax5.plot(seconds, attitude_error_true, label='Attitude error', color='blue') #label='Attitude error (true ukf)', ls=':', color='blue')
        
        #ax5.plot(seconds, angular_rate_error, label='Angular rate error (est)', color='orange')
        ax5.plot(seconds, angular_rate_error_true, label='Angular rate error (true ukf)', ls=':', color='magenta')

        # add 1 deg line as reference
        ax5.axhline(y=1, color='green', ls=':', label='1 deg reference')
        
        # always start the plot at 0
        ax5.set_ylim(bottom=0)

        # show more grid lines
        ax5.yaxis.set_major_locator(plt.MaxNLocator(5))

        # biases
        ax6.grid()
        ax6.set_title("Magnetometer biases")
        ax6.set_ylabel("Bias (nT)")
        #ax6.plot(attitude_sim_timestamp, mag_bias_x_est_sim, label='Magnetometer x (est)', color='blue')
        #ax6.plot(attitude_sim_timestamp, mag_bias_y_est_sim, label='Magnetometer y (est)', color='orange')
        #ax6.plot(attitude_sim_timestamp, mag_bias_z_est_sim, label='Magnetometer z (est)', color='green')

        ax6.plot(seconds, true_ukf_mag_bias_x, label='Magnetometer x (true ukf)', ls=':', color='blue')
        ax6.plot(seconds, true_ukf_mag_bias_y, label='Magnetometer y (true ukf)', ls=':', color='orange')
        ax6.plot(seconds, true_ukf_mag_bias_z, label='Magnetometer z (true ukf)', ls=':', color='green')

        ax6.plot(seconds, mag_bias_x_true, label='Magnetometer x (true)', ls='--', color='blue')
        ax6.plot(seconds, mag_bias_y_true, label='Magnetometer y (true)', ls='--', color='orange')
        ax6.plot(seconds, mag_bias_z_true, label='Magnetometer z (true)', ls='--', color='green')

        ax7.grid()
        ax7.set_title("Gyroscope biases")
        ax7.set_ylabel("Bias (deg/s)")
        ax7.set_xlabel("Time (s)")
        #ax7.plot(attitude_sim_timestamp, gyro_bias_x_est_sim, label='Gyroscope x (est)', color='blue')
        #ax7.plot(attitude_sim_timestamp, gyro_bias_y_est_sim, label='Gyroscope y (est)', color='orange')
        #ax7.plot(attitude_sim_timestamp, gyro_bias_z_est_sim, label='Gyroscope z (est)', color='green')

        ax7.plot(seconds, true_ukf_gyro_bias_x, label='Gyroscope x (true ukf)', ls=':', color='blue')
        ax7.plot(seconds, true_ukf_gyro_bias_y, label='Gyroscope y (true ukf)', ls=':', color='orange')
        ax7.plot(seconds, true_ukf_gyro_bias_z, label='Gyroscope z (true ukf)', ls=':', color='green')

        ax7.plot(seconds, gyro_bias_x_true, label='Gyroscope x (true)', ls='--', color='blue')
        ax7.plot(seconds, gyro_bias_y_true, label='Gyroscope y (true)', ls='--', color='orange')
        ax7.plot(seconds, gyro_bias_z_true, label='Gyroscope z (true)', ls='--', color='green')

        ax75.grid()
        ax75.set_title("Variances")
        ax75.set_ylabel("Variance")
        ax75.set_xlabel("Time (s)")
        #ax75.plot(attitude_sim_timestamp, quaternion_variance_sim, label='Quaternion variance (est)', color='blue')
        ax75.plot(seconds, quaternion_variance_true, label='Quaternion variance', color='blue')
        #ax75.plot(attitude_sim_timestamp, angular_velocity_variance_sim, label='Angular velocity variance (est)', color='orange')
        ax75.plot(seconds, angular_velocity_variance, label='Angular velocity variance', color='orange')
        #ax75.plot(attitude_sim_timestamp, magnetic_field_bias_variance_sim, label='Magnetic field bias variance (est)', color='green')
        ax75.plot(seconds, magnetic_field_bias_variance, label='Magnetic field bias variance', color='green')
        #ax75.plot(attitude_sim_timestamp, gyro_bias_variance_sim, label='Gyroscope bias variance (est)', color='red')
        ax75.plot(seconds, gyro_bias_variance, label='Gyroscope bias variance', color='red')

        ax5.legend(loc="upper right")
        ax6.legend(loc="upper right")
        ax7.legend(loc="upper right")
        ax75.legend(loc="upper right")


        # sun vectors
        fig3, ax8 = plt.subplots(1, figsize=(12, 10), sharex=True)
        ax8.grid()
        ax8.set_title("Sun vector")
        ax8.set_ylabel("Unit vector")

        
        #ax8.plot(sun_vector_sim_timestamp, sun_vector_x_sim, label=r'$x$', color='blue')
        #ax8.plot(sun_vector_sim_timestamp, sun_vector_y_sim, label=r'$y$', color='orange')
        #ax8.plot(sun_vector_sim_timestamp, sun_vector_z_sim, label=r'$z$', color='green')

        # plot brightest sun vector in inertial frame
        for i in range(3):
            ax8.plot(sun_vector_sim_timestamp, psd_sun_vector_sim_inertial_brightest[i], label=f'{[r'$x$', r'$y$', r'$z$'][i]} (brightest sim)', ls='-.', color=['blue', 'orange', 'green'][i])
            #ax8.plot(seconds, psd_sun_vector_true_inertial_brightest[i], label=f'{["x", "y", "z"][i]} (brightest true)', ls=':', color=['blue', 'orange', 'green'][i])
        
        ax8.plot(seconds, sun_vector_x_true, label=r'$x$', ls='--', color='blue')
        ax8.plot(seconds, sun_vector_y_true, label=r'$x$', ls='--', color='orange')
        ax8.plot(seconds, sun_vector_z_true, label=r'$x$', ls='--', color='green')
        

        
        ax8.legend(loc="upper right")
        

        fig3.tight_layout()

        # sun sensors
        fig4, axs = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
        for i in range(6):
            axs[i].grid()
            axs[i].set_title(f"Sun sensor {i+1}")
            axs[i].set_ylabel("Unit vector")
            axs[i].plot(psd_sun_vector_sim_timestamp, psd_sun_vector_sim_body[i][0], label='x sim') # this is already in body frame
            axs[i].plot(psd_sun_vector_sim_timestamp, psd_sun_vector_sim_body[i][1], label='y sim')
            axs[i].plot(psd_sun_vector_sim_timestamp, psd_sun_vector_sim_body[i][2], label='z sim')
            axs[i].plot(seconds, psd_sun_vector_true[i][0], label='x (true)', ls='--', color='blue')
            axs[i].plot(seconds, psd_sun_vector_true[i][1], label='y (true)', ls='--', color='orange')
            axs[i].plot(seconds, psd_sun_vector_true[i][2], label='z (true)', ls='--', color='green')
            #axs[i].plot(psd_sun_vector_sim_timestamp, psd_sun_vector_sim[i][3], label='intensity', color='black')
            #axs[i].plot(seconds, psd_sun_vector_true[i][3], label='intensity (true)', ls='--', color='black')
            axs[i].legend(loc="upper right")
        axs[-1].set_xlabel("Time (s)")
        fig4.tight_layout()
        

        # convert global_magnetic_field_sim to local magnetic field
        local_magnetic_field_x_from_global_sim = []
        local_magnetic_field_y_from_global_sim = []
        local_magnetic_field_z_from_global_sim = []
        for i in range(len(global_magnetic_field_x_sim)):
            # convert global magnetic field to local magnetic field using attitude quaternion
            q = attitude_true[i]
            global_field_vector = np.array([global_magnetic_field_x_sim[i], global_magnetic_field_y_sim[i], global_magnetic_field_z_sim[i]])
            local_field_vector = q.rotate(global_field_vector)
            local_magnetic_field_x_from_global_sim.append(local_field_vector[0])
            local_magnetic_field_y_from_global_sim.append(local_field_vector[1])
            local_magnetic_field_z_from_global_sim.append(local_field_vector[2])
        
        fig6, ax70 = plt.subplots(1, sharex=True)
        ax70.grid()
        ax70.set_title("Variances")
        ax70.set_ylabel(r"$\sigma ^2$")
        ax70.set_xlabel("Time (s)")
        ax70.plot(attitude_sim_timestamp, quaternion_variance_sim, label='Quaternion variance', color='blue')
        #ax70.plot(seconds, quaternion_variance_true, label='Quaternion variance (true ukf)', ls=':', color='blue')
        ax70.plot(attitude_sim_timestamp, angular_velocity_variance_sim, label='Angular velocity variance', color='orange')
        #ax75.plot(seconds, angular_velocity_variance, label='Angular velocity variance (true ukf)', ls=':', color='orange')
        #ax75.plot(attitude_sim_timestamp, magnetic_field_bias_variance_sim, label='Magnetic field bias variance (est)', color='green')
        #ax75.plot(seconds, magnetic_field_bias_variance, label='Magnetic field bias variance (true ukf)', ls=':', color='green')
        #ax75.plot(attitude_sim_timestamp, gyro_bias_variance_sim, label='Gyroscope bias variance (est)', color='red')
        #ax75.plot(seconds, gyro_bias_variance, label='Gyroscope bias variance (true ukf)', ls=':', color='red')

        #ax5.legend(loc="upper right")
        #ax6.legend(loc="upper right")
        #ax7.legend(loc="upper right")
        
        #For bigger legend and bigger everything in there
        #leg = ax70.legend(fontsize=14, loc="upper right")

        #for line in leg.get_lines():
            #line.set_linewidth(3)

        #for handle in leg.legendHandles:
        #    handle.set_markersize(10)
        #ax70.set_ylim(0, 0.0014)
        ax70.legend(loc="upper right")

        fig6.tight_layout()
        fig6.savefig("figures/variances_att_ang.pdf", bbox_inches="tight")

        fig5, (ax69, ax68) = plt.subplots(2, sharex=True)        

                # --- Eclipse shading (vertical band(s) across the full y-range) ---
        sun_norm = np.sqrt(sun_vector_x_true**2 + sun_vector_y_true**2 + sun_vector_z_true**2)
        eps = 1e-9
        eclipse = sun_norm <= eps

        if np.any(eclipse):
            # Shade on ax69 (Errors)
            ax69.fill_between(
                seconds, 0, 1,
                where=eclipse,
                transform=ax69.get_xaxis_transform(),  # y in [0,1] axis fraction
                color='black', alpha=0.15,
                label='Eclipse'
            )

            # Shade on ax68 too (same x-axis, different subplot)
            ax68.fill_between(
                seconds, 0, 1,
                where=eclipse,
                transform=ax68.get_xaxis_transform(),
                color='black', alpha=0.15
                # no label here to avoid duplicate legend entries
            )



        
        # errors
        ax69.grid()
        ax69.set_title("Errors")
        ax69.set_ylabel(r"$\delta \ [\degree]$") # delta and degree sign
        #ax5.plot(seconds, attitude_error, label='Attitude error (est)', color='blue')
        ax69.plot(seconds, attitude_error_true, label = fr'$\delta \mathbf{{q}} \ (\overline{{\|\delta\mathbf{{q}}\|}} = {mean_err:.4f}^\circ)$', color='blue') #label='Attitude error (true ukf)', ls=':', color='blue')
         

        #ax5.plot(seconds, angular_rate_error, label='Angular rate error (est)', color='orange')
        ax69.plot(seconds, angular_rate_error_true, label=r'$\delta\mathbf{\omega}$', ls=':', color='magenta')

        # add 1 deg line as reference
        ax69.axhline(y=1, color='green', ls=':', label=r'$1^{\circ{}}$ reference')
        
        # always start the plot at 0
        ax69.set_ylim(bottom=0)

        
                # --- Mean error shown in legend only (no line drawn on axes) ---
        #mean_proxy = Line2D([], [], linestyle='None')  # invisible / no line
        #ax69.legend(
        #    handles=[*ax69.get_legend_handles_labels()[0], mean_proxy],
        #    labels=[*ax69.get_legend_handles_labels()[1], fr"Mean = {mean_err:.3f}$^\circ$"],
        #    loc="best"
        #)
        
        # show more grid lines
        ax69.yaxis.set_major_locator(plt.MaxNLocator(5))

        ax68.grid()
        ax68.set_title("Sun vector")
        ax68.set_ylabel(r"Unit vector [$\mathbf{v}_{\mathcal{u}}$]")
        ax68.set_xlabel("Time (s)")
        
        #ax68.plot(sun_vector_sim_timestamp, sun_vector_x_sim, label='x (est)', color='blue')
        #ax68.plot(sun_vector_sim_timestamp, sun_vector_y_sim, label='y (est)', color='orange')
        #ax68.plot(sun_vector_sim_timestamp, sun_vector_z_sim, label='z (est)', color='green')

        # plot brightest sun vector in inertial frame
        #for i in range(3):
            #ax68.plot(sun_vector_sim_timestamp, psd_sun_vector_sim_inertial_brightest[i], label=f'{["x", "y", "z"][i]} (brightest sim)', ls='-.', color=['blue', 'orange', 'green'][i]) #When no sil, can be commented out
            #ax68.plot(seconds, psd_sun_vector_true_inertial_brightest[i], label=f'{["x", "y", "z"][i]} (brightest true)', ls=':', color=['blue', 'orange', 'green'][i])
        #sun_vector_x_sim: x (est): 
        #Brightest sim: 

        ax68.plot(seconds, sun_vector_x_true, label=r'$v_{x_t}$', color='blue')
        ax68.plot(seconds, sun_vector_y_true, label=r'$v_{y_t}$', color='orange')
        ax68.plot(seconds, sun_vector_z_true, label=r'$v_{z_t}$', color='green')

        #ax68.plot(seconds, sun_vector_x_true, label='x (true)', ls='--', color='blue')
        #ax68.plot(seconds, sun_vector_y_true, label='y (true)', ls='--', color='orange')
        #ax68.plot(seconds, sun_vector_z_true, label='z (true)', ls='--', color='green')


        ax69.legend(loc="upper right") #comment this if you want to use "mean_proxy..." in legend as separate
        ax68.legend(loc="upper right")


        fig7, (ax89, ax98, ax90) = plt.subplots(3, sharex=True)


        # biases
        ax89.grid()
        ax89.set_title("Magnetometer biases")
        ax89.set_ylabel(r"$\mathbf{b}_{mag} \ [nT]$")
        #ax6.plot(attitude_sim_timestamp, mag_bias_x_est_sim, label='Magnetometer x (est)', color='blue')
        #ax6.plot(attitude_sim_timestamp, mag_bias_y_est_sim, label='Magnetometer y (est)', color='orange')
        #ax6.plot(attitude_sim_timestamp, mag_bias_z_est_sim, label='Magnetometer z (est)', color='green')

        ax89.plot(seconds, true_ukf_mag_bias_x, label=r'$b_{x_{mag}}$', color='blue')
        ax89.plot(seconds, true_ukf_mag_bias_y, label=r'$b_{y_{mag}}$', color='orange')
        ax89.plot(seconds, true_ukf_mag_bias_z, label=r'$b_{z_{mag}}$', color='green')

        #ax6.plot(seconds, mag_bias_x_true, label='Magnetometer x (true)', ls='--', color='blue')
        #ax6.plot(seconds, mag_bias_y_true, label='Magnetometer y (true)', ls='--', color='orange')
        #ax6.plot(seconds, mag_bias_z_true, label='Magnetometer z (true)', ls='--', color='green')
        mean_line_mag_x = ax89.axhline(
            mean_ukf_mag_bias_x,
            color='tab:blue',
            linestyle='--',
            linewidth=1.5,
            zorder=2
        )
        mean_line_mag_x.set_label(fr"Mean x = {mean_ukf_mag_bias_x:.4f} [nT]")
        mean_line_mag_y = ax89.axhline(
            mean_ukf_mag_bias_y,
            color='tab:orange',
            linestyle='--',
            linewidth=1.5,
            zorder=2
        )
        mean_line_mag_y.set_label(fr"Mean y = {mean_ukf_mag_bias_y:.4f} [nT]")
        mean_line_mag_z = ax89.axhline(
            mean_ukf_mag_bias_z,
            color='tab:green',
            linestyle='--',
            linewidth=1.5,
            zorder=2
        )
        mean_line_mag_z.set_label(fr"Mean z = {mean_ukf_mag_bias_z:.4f} [nT]")

        # biases
        ax98.grid()
        ax98.set_title("Magnetometer measurements biases")
        ax98.set_ylabel(r"$\mathbf{b}_{mag} \ [nT]$")
        #ax6.plot(attitude_sim_timestamp, mag_bias_x_est_sim, label='Magnetometer x (est)', color='blue')
        #ax6.plot(attitude_sim_timestamp, mag_bias_y_est_sim, label='Magnetometer y (est)', color='orange')
        #ax6.plot(attitude_sim_timestamp, mag_bias_z_est_sim, label='Magnetometer z (est)', color='green')

        #ax98.plot(seconds, true_ukf_mag_bias_x, label=r'$b_{x_{mag}}$', color='blue')
        #ax98.plot(seconds, true_ukf_mag_bias_y, label=r'$b_{y_{mag}}$', color='orange')
        #ax98.plot(seconds, true_ukf_mag_bias_z, label=r'$b_{z_{mag}}$', color='green')

        ax98.plot(seconds, mag_bias_x_true, label='Magnetometer x (true)', ls='--', color='blue')
        ax98.plot(seconds, mag_bias_y_true, label='Magnetometer y (true)', ls='--', color='orange')
        ax98.plot(seconds, mag_bias_z_true, label='Magnetometer z (true)', ls='--', color='green')

        ax90.grid()
        ax90.set_title("Gyroscope biases")
        ax90.set_ylabel(r"$\mathbf{b}_{gyro} \ [^{\circ{}}/s]$")
        ax90.set_xlabel("Time (s)")
        #ax7.plot(attitude_sim_timestamp, gyro_bias_x_est_sim, label='Gyroscope x (est)', color='blue')
        #ax7.plot(attitude_sim_timestamp, gyro_bias_y_est_sim, label='Gyroscope y (est)', color='orange')
        #ax7.plot(attitude_sim_timestamp, gyro_bias_z_est_sim, label='Gyroscope z (est)', color='green')

        ax90.plot(seconds, true_ukf_gyro_bias_x, label=r'$b_{x_{gyro}}$', color='blue')
        ax90.plot(seconds, true_ukf_gyro_bias_y, label=r'$b_{y_{gyro}}$', color='orange')
        ax90.plot(seconds, true_ukf_gyro_bias_z, label=r'$b_{z_{gyro}}$', color='green')
        
        #ax7.plot(seconds, gyro_bias_x_true, label='Gyroscope x (true)', ls='--', color='blue')
        #ax7.plot(seconds, gyro_bias_y_true, label='Gyroscope y (true)', ls='--', color='orange')
        #ax7.plot(seconds, gyro_bias_z_true, label='Gyroscope z (true)', ls='--', color='green')
        ax89.legend(loc="upper right")
        ax98.legend(loc="upper right")
        ax90.legend(loc="upper right")

        fig7.tight_layout()
        fig7.savefig("figures/biases.pdf", bbox_inches="tight")

        fig8, ax91 = plt.subplots(1, sharex=True)

        # --- Eclipse shading (full height, no autoscale interference) ---
        if np.any(eclipse):
            shade = ax91.fill_between(
                seconds,
                0, 1,                      # full axis height (0–1 in axis coords)
                where=eclipse,
                transform=ax91.get_xaxis_transform(),  # y is axis fraction
                color='black',
                alpha=0.15,
                label='Eclipse'
            )
            shade.set_clip_on(False)      # prevents clipping to data limits
            shade.set_zorder(-1)          # keeps shading behind the data

        # --- Plot attitude error ---
        ax91.grid()
        ax91.set_title("Errors")
        ax91.set_ylabel(r"$\delta \ [\degree]$")

        ax91.plot(
            seconds,
            attitude_error_true,
            label=fr'$\delta \mathbf{{q}}$  (Max = {np.max(attitude_error_true):.4f}$^\circ$)',
            color='blue'
        )
        # --- Mean error reference line ---
        mean_line = ax91.axhline(
            mean_err,
            color='tab:red',
            linestyle='--',
            linewidth=1.5,
            zorder=2
        )

        # If you want it in the legend (keeps it self-documenting):
        mean_line.set_label(fr"Mean = {mean_err:.4f}$^\circ$")



        ax91.set_xlabel("Time (s)")
        # --- Autoscale to data ---
        ax91.relim()
        ax91.autoscale_view()

        # Optional: nicer y‑axis spacing
        ax91.yaxis.set_major_locator(plt.MaxNLocator(5))

        ax91.legend()
        fig8.tight_layout()
        fig8.savefig("figures/attitude_error.pdf", bbox_inches="tight")




        fig9, ax75 = plt.subplots(1, sharex=True)


        ax75.grid()
        ax75.set_title("Variances")
        ax75.set_ylabel(r"$\sigma ^2$")
        ax75.set_xlabel("Time (s)")
        #ax75.plot(attitude_sim_timestamp, quaternion_variance_sim, label='Quaternion variance (est)', color='blue')
        ax75.plot(seconds, quaternion_variance_true, label='Quaternion variance', color='blue')
        #ax75.plot(attitude_sim_timestamp, angular_velocity_variance_sim, label='Angular velocity variance (est)', color='orange')
        ax75.plot(seconds, angular_velocity_variance, label='Angular velocity variance', color='orange')
        #ax75.plot(attitude_sim_timestamp, magnetic_field_bias_variance_sim, label='Magnetic field bias variance (est)', color='green')
        ax75.plot(seconds, magnetic_field_bias_variance, label='Magnetic field bias variance', color='green')
        #ax75.plot(attitude_sim_timestamp, gyro_bias_variance_sim, label='Gyroscope bias variance (est)', color='red')
        ax75.plot(seconds, gyro_bias_variance, label='Gyroscope bias variance', color='red')
        #ax75.set_ylim(0, 0.012)   


        #For bigger legend and bigger everything in there
        #leg = ax75.legend(fontsize=14, loc="upper right")

        #for line in leg.get_lines():
        #    line.set_linewidth(3)

        #for handle in leg.legendHandles:
        #    handle.set_markersize(10)
        ax75.set_ylim(0, 0.00002)
        ax75.legend(loc="upper right")

        fig9.tight_layout()
        fig9.savefig("figures/Variances_all.pdf", bbox_inches="tight")


        #plt.tight_layout()
        #plt.show()


        fig5.tight_layout()

        fig10, ax92 = plt.subplots(1, sharex=True)

        ax92.grid()
        ax92.set_title("Gyroscope biases")
        ax92.set_ylabel(r"$\mathbf{b}_{gyro} \ [^{\circ{}}/s]$")
        ax92.set_xlabel("Time (s)")
        #ax7.plot(attitude_sim_timestamp, gyro_bias_x_est_sim, label='Gyroscope x (est)', color='blue')
        #ax7.plot(attitude_sim_timestamp, gyro_bias_y_est_sim, label='Gyroscope y (est)', color='orange')
        #ax7.plot(attitude_sim_timestamp, gyro_bias_z_est_sim, label='Gyroscope z (est)', color='green')

        ax92.plot(seconds, true_ukf_gyro_bias_x, label=r'$b_{x_{gyro}} \ (e)$', color='darkblue')#, linewidth=2.5)
        ax92.plot(seconds, true_ukf_gyro_bias_y, label=r'$b_{y_{gyro}} \ (e)$', color='darkorange')#,linewidth=2.5)
        ax92.plot(seconds, true_ukf_gyro_bias_z, label=r'$b_{z_{gyro}} \ (e)$', color='darkgreen')#,linewidth=2.5)
        
        ax92.plot(seconds, gyro_bias_x_true, label=r'$b_{x_{gyro}} \ (t)$', ls='--', color='blue',linewidth=1.2, alpha=0.4)
        ax92.plot(seconds, gyro_bias_y_true, label=r'$b_{y_{gyro}} \ (t)$', ls='--', color='orange',linewidth=1.2, alpha=0.4)
        ax92.plot(seconds, gyro_bias_z_true, label=r'$b_{z_{gyro}} \ (t)$', ls='--', color='green',linewidth=1.2, alpha=0.4)

        ax92.legend(loc="upper right")

        fig10.tight_layout()

        fig12, ax93 = plt.subplots(1, sharex=True)
        # biases
        ax93.grid()
        ax93.set_title("Magnetometer biases")
        ax93.set_ylabel(r"$\mathbf{b}_{mag} \ [nT]$")
        #ax6.plot(attitude_sim_timestamp, mag_bias_x_est_sim, label='Magnetometer x (est)', color='blue')
        #ax6.plot(attitude_sim_timestamp, mag_bias_y_est_sim, label='Magnetometer y (est)', color='orange')
        #ax6.plot(attitude_sim_timestamp, mag_bias_z_est_sim, label='Magnetometer z (est)', color='green')

        ax93.plot(seconds, true_ukf_mag_bias_x, label=r'$b_{x_{mag}} \ (e)$', color='darkblue')
        ax93.plot(seconds, true_ukf_mag_bias_y, label=r'$b_{y_{mag}} \ (e)$', color='darkorange')
        ax93.plot(seconds, true_ukf_mag_bias_z, label=r'$b_{z_{mag}} \ (e)$', color='darkgreen')

        ax93.plot(seconds, mag_bias_x_true, label=r'$b_{x_{mag}} \ (t)$', ls='--', color='blue',linewidth=1.2, alpha=0.4)
        ax93.plot(seconds, mag_bias_y_true, label=r'$b_{y_{mag}} \ (t)$', ls='--', color='orange',linewidth=1.2, alpha=0.4)
        ax93.plot(seconds, mag_bias_z_true, label=r'$b_{z_{mag}} \ (t)$', ls='--', color='green',linewidth=1.2, alpha=0.4)
        
        ax93.legend(loc="upper right")
        fig12.tight_layout()

        fig13, (ax94, ax95) = plt.subplots(2, sharex=True)

        ax94.grid()
        ax94.set_title("Gyroscope biases")
        ax94.set_ylabel(r"$\mathbf{b}_{gyro} \ [^{\circ{}}/s]$")
        
        #ax7.plot(attitude_sim_timestamp, gyro_bias_x_est_sim, label='Gyroscope x (est)', color='blue')
        #ax7.plot(attitude_sim_timestamp, gyro_bias_y_est_sim, label='Gyroscope y (est)', color='orange')
        #ax7.plot(attitude_sim_timestamp, gyro_bias_z_est_sim, label='Gyroscope z (est)', color='green')

        ax94.plot(seconds, true_ukf_gyro_bias_x, label=r'$b_{x_{gyro}} \ (e)$', color='darkblue')#, linewidth=2.5)
        ax94.plot(seconds, true_ukf_gyro_bias_y, label=r'$b_{y_{gyro}} \ (e)$', color='darkorange')#,linewidth=2.5)
        ax94.plot(seconds, true_ukf_gyro_bias_z, label=r'$b_{z_{gyro}} \ (e)$', color='darkgreen')#,linewidth=2.5)
        
        ax94.plot(seconds, gyro_bias_x_true, label=r'$b_{x_{gyro}} \ (t)$', ls='--', color='blue',linewidth=1.2, alpha=0.4)
        ax94.plot(seconds, gyro_bias_y_true, label=r'$b_{y_{gyro}} \ (t)$', ls='--', color='orange',linewidth=1.2, alpha=0.4)
        ax94.plot(seconds, gyro_bias_z_true, label=r'$b_{z_{gyro}} \ (t)$', ls='--', color='green',linewidth=1.2, alpha=0.4)

        #ax94.plot(seconds, gyro_bias_rate_error_true, label=r'$\delta\mathbf{\omega}$', ls=':', color='magenta')

        ax94.legend(loc="upper right")

                
        # biases
        ax95.grid()
        ax95.set_title("Magnetometer biases")
        ax95.set_ylabel(r"$\mathbf{b}_{mag} \ [nT]$")
        ax95.set_xlabel("Time (s)")
        #ax6.plot(attitude_sim_timestamp, mag_bias_x_est_sim, label='Magnetometer x (est)', color='blue')
        #ax6.plot(attitude_sim_timestamp, mag_bias_y_est_sim, label='Magnetometer y (est)', color='orange')
        #ax6.plot(attitude_sim_timestamp, mag_bias_z_est_sim, label='Magnetometer z (est)', color='green')

        ax95.plot(seconds, true_ukf_mag_bias_x, label=r'$b_{x_{mag}} \ (e)$', color='darkblue')
        ax95.plot(seconds, true_ukf_mag_bias_y, label=r'$b_{y_{mag}} \ (e)$', color='darkorange')
        ax95.plot(seconds, true_ukf_mag_bias_z, label=r'$b_{z_{mag}} \ (e)$', color='darkgreen')

        ax95.plot(seconds, mag_bias_x_true, label=r'$b_{x_{mag}} \ (t)$', ls='--', color='blue',linewidth=1.2, alpha=0.4)
        ax95.plot(seconds, mag_bias_y_true, label=r'$b_{y_{mag}} \ (t)$', ls='--', color='orange',linewidth=1.2, alpha=0.4)
        ax95.plot(seconds, mag_bias_z_true, label=r'$b_{z_{mag}} \ (t)$', ls='--', color='green',linewidth=1.2, alpha=0.4)
        
        ax95.legend(loc="upper right")
        fig13.tight_layout()
        fig13.savefig("figures/biases_true_est.pdf", bbox_inches="tight")


        
        # ------------------------ COMPARISON GRAPH BETWEEN DEFAULT, R_DYNAMIC AND Q_R_DYNAMIC ------------------------
        fig11, ax99 = plt.subplots(1, sharex=True)

        # --- Eclipse shading (full height, no autoscale interference) ---
        shade = None
        if np.any(eclipse):
            shade = ax99.fill_between(
                seconds,
                0, 1,
                where=eclipse,
                transform=ax99.get_xaxis_transform(),
                color='black',
                alpha=0.15,
                label='Eclipse'
            )
            shade.set_clip_on(False)
            shade.set_zorder(-1)

        # --- Plot attitude errors ---
        ax99.grid(True)
        ax99.set_title("Errors")
        ax99.set_ylabel(r"$\delta \ [\degree]$")

        # Colors: default=green, R_dynamic=blue, Q_R_dynamic=red
        c_def = 'green'
        c_dyn = 'blue'
        c_qrd = 'red'

        # --- Delta-q series first (for consistent handle references) ---
        l_qrd, = ax99.plot(
            seconds,
            attitude_error_true_Q_R_dynamic,
            color=c_qrd,
            linewidth=1.8,
            zorder=3,
            label=fr'$\delta \mathbf{{q}}_{{QR \ dynamic}}$ (Max = {np.nanmax(attitude_error_true_Q_R_dynamic):.4f}$^\circ$)'
        )

        l_dyn, = ax99.plot(
            seconds,
            attitude_error_true_R_dynamic,
            color=c_dyn,
            linewidth=1.8,
            zorder=3,
            label=fr'$\delta \mathbf{{q}}_{{R \ dynamic}}$ (Max = {np.nanmax(attitude_error_true_R_dynamic):.4f}$^\circ$)'
        )

        l_def, = ax99.plot(
            seconds,
            attitude_error_true,  # "base" series, default
            color=c_def,
            linewidth=1.8,
            zorder=3,
            label=fr'$\delta \mathbf{{q}}_{{default}}$ (Max = {np.nanmax(attitude_error_true):.4f}$^\circ$)'
        )

        # --- Mean lines after the series ---
        m_qrd = ax99.axhline(
            mean_err_Q_R_dynamic,
            color=c_qrd,
            linestyle='--',
            linewidth=1.5,
            zorder=2,
            label=fr"Mean (QR dynamic) = {mean_err_Q_R_dynamic:.4f}$^\circ$"
        )

        m_dyn = ax99.axhline(
            mean_err_R_dynamic,
            color=c_dyn,
            linestyle='--',
            linewidth=1.5,
            zorder=2,
            label=fr"Mean (R dynamic) = {mean_err_R_dynamic:.4f}$^\circ$"
        )

        m_def = ax99.axhline(
            mean_err,  # "base" mean, now named default
            color=c_def,
            linestyle='--',
            linewidth=1.5,
            zorder=2,
            label=fr"Mean (default) = {mean_err:.4f}$^\circ$"
        )

        ax99.set_xlabel("Time (s)")

        # --- Autoscale to data ---
        ax99.relim()
        ax99.autoscale_view()
        ax99.yaxis.set_major_locator(plt.MaxNLocator(5))

        # --- Legend ordering: Eclipse first, then delta-q, then means ---
        handles = []
        if shade is not None:
            handles.append(shade)

        handles += [l_def, l_dyn, l_qrd, m_def, m_dyn, m_qrd]
        labels = [h.get_label() for h in handles]

        ax99.legend(handles, labels)
        fig11.tight_layout()
        fig11.savefig("figures/covariance_logic.pdf", bbox_inches="tight")

        plt.show()
        

        plt.tight_layout()

        plt.show()
if __name__ == '__main__':

    filename = "result.txt"
    if(len(sys.argv) > 1):
        filename = sys.argv[1]
    plot(filename)



    """
    # ------------------------ COMPARISON GRAPH BETWEEN DEFAULT, R_DYNAMIC AND Q_R_DYNAMIC ------------------------
    fig11, ax99 = plt.subplots(1, sharex=True)

    # --- Eclipse shading (full height, no autoscale interference) ---
    shade = None
    if np.any(eclipse):
        shade = ax99.fill_between(
            seconds,
            0, 1,
            where=eclipse,
            transform=ax99.get_xaxis_transform(),
            color='black',
            alpha=0.15,
            label='Eclipse'
        )
        shade.set_clip_on(False)
        shade.set_zorder(-1)

    # --- Plot attitude errors ---
    ax99.grid(True)
    ax99.set_title("Errors")
    ax99.set_ylabel(r"$\delta \ [\degree]$")

    # Colors: default=green, R_dynamic=blue, Q_R_dynamic=red
    c_def = 'green'
    c_dyn = 'blue'
    c_qrd = 'red'

    # --- Delta-q series first (for consistent handle references) ---
    l_def, = ax99.plot(
        seconds,
        attitude_error_true_default,
        color=c_def,
        linewidth=1.8,
        zorder=3,
        label=fr'$\delta \mathbf{{q}}_{{default}}$ (Max = {np.nanmax(attitude_error_true_default):.4f}$^\circ$)'
    )

    l_dyn, = ax99.plot(
        seconds,
        attitude_error_true_R_dynamic,
        color=c_dyn,
        linewidth=1.8,
        zorder=3,
        label=fr'$\delta \mathbf{{q}}_{{R \ dynamic}}$ (Max = {np.nanmax(attitude_error_true_R_dynamic):.4f}$^\circ$)'
    )

    l_qrd, = ax99.plot(
        seconds,
        attitude_error_true,  # "base" series, now named Q_R_dynamic
        color=c_qrd,
        linewidth=1.8,
        zorder=3,
        label=fr'$\delta \mathbf{{q}}_{{QR \ dynamic}}$ (Max = {np.nanmax(attitude_error_true):.4f}$^\circ$)'
    )

    # --- Mean lines after the series ---
    m_def = ax99.axhline(
        mean_err_default,
        color=c_def,
        linestyle='--',
        linewidth=1.5,
        zorder=2,
        label=fr"Mean (default) = {mean_err_default:.4f}$^\circ$"
    )

    m_dyn = ax99.axhline(
        mean_err_R_dynamic,
        color=c_dyn,
        linestyle='--',
        linewidth=1.5,
        zorder=2,
        label=fr"Mean (R dynamic) = {mean_err_R_dynamic:.4f}$^\circ$"
    )

    m_qrd = ax99.axhline(
        mean_err,  # "base" mean, now named Q_R_dynamic
        color=c_qrd,
        linestyle='--',
        linewidth=1.5,
        zorder=2,
        label=fr"Mean (QR dynamic) = {mean_err:.4f}$^\circ$"
    )

    ax99.set_xlabel("Time (s)")

    # --- Autoscale to data ---
    ax99.relim()
    ax99.autoscale_view()
    ax99.yaxis.set_major_locator(plt.MaxNLocator(5))

    # --- Legend ordering: Eclipse first, then delta-q, then means ---
    handles = []
    if shade is not None:
        handles.append(shade)

    handles += [l_def, l_dyn, l_qrd, m_def, m_dyn, m_qrd]
    labels = [h.get_label() for h in handles]

    ax99.legend(handles, labels)
    fig11.tight_layout()
    fig11.savefig("figures/covariance_logic.pdf", bbox_inches="tight")

    plt.show()
    """