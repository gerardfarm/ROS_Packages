"""  www.github.com/diegoavillegasg/IMU-GNSS-Lidar-sensor-fusion-using-Extended-Kalman-Filter-for-State-Estimation """

import numpy as np
from numpy import matmul
from localization.rotations import skew_symmetric, Quaternion


# Extended-Kalman Filter Class
class EKF:
    def __init__(self, var_gps = 0.01, var_acc=0.0001, var_gy=0.008):
        # Setting the estimated sensor variances correctly.
        self.var_acc  = var_acc
        self.var_gyro = var_gy
        self.var_gps  = var_gps

        self.g = np.array([0, 0, -9.81])  # gravity

        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)  # measurement model jacobian

        # EKF Initialization.
        self.p_est_prev    = np.zeros(3)
        self.v_est_prev    = np.zeros(3)
        self.q_est_prev    = np.zeros(4)
        self.q_est_prev    = np.array([0.99999, 0.00001, 0.00001, 0.00001]) # [1,0,0,0]

        self.p_cov_prev    = np.zeros([9, 9])

        self.acc_data_t_previous = None

    def measurement_update(self, sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
        # 3.1 Compute Kalman Gain
        # K_k = P_k * H_k.T * inv( H_k * P_k * H_k.T + R_k )
        try:
            temp = matmul(self.h_jac, matmul(p_cov_check, self.h_jac.T)) + sensor_var*np.eye(3)
            inv = np.linalg.inv(temp)
            #print("temp: ", temp.shape, "sensor_var: ", sensor_var)
            #K = np.linalg.inv(matmul(h_jac, matmul(p_cov_check, h_jac.T)) + sensor_var )))
            K = matmul(p_cov_check, matmul(self.h_jac.T, inv))
            
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                raise "A singular matrix "

        # 3.2 Compute error state
        #print("y_k size: ", y_k.shape, "h_jac size: ", self.h_jac.shape, "p_check size: ", p_check.shape, 
        #                                                            "P_CHECK: ", p_check)
        error_state = y_k - p_check #matmul(self.h_jac[:3, :3], p_check)

        # 3.3 Correct predicted state
        p_hat = p_check + matmul(K, error_state)[:3]
        v_hat = v_check + matmul(K, error_state)[3:6]
        #print("error_state ", error_state.shape, "K: ", K.shape, "q_check: ", q_check.shape)
        q_hat = Quaternion(axis_angle = matmul(K, error_state)[6:]).quat_mult_right(q_check)

        # 3.4 Compute corrected covariance
        p_cov_hat = matmul( np.eye(9) - matmul(K, self.h_jac), p_cov_check)

        return p_hat, v_hat, q_hat, p_cov_hat


    def Predict_and_Update(self, acc_data_t_ini,
                                 gps_data_t,
                                 gps_data_current, 
                                 acc_data_t_current,
                                 acc_data_current, 
                                 gyro_data_current,
                                 start):
        """
        Params:
           - gps_data: gps position data
           - acc_data: linear velocities
           - gyro_data: angular velocities
        Return:
            - p_est: Estimated position
            - v_est: Estimated velocities
            - q_est: Estimated quaternion (Euler Angles)
            - p_cov: Covariance Matrix
        """
        
        if start: self.acc_data_t_previous = acc_data_t_ini
        
        delta_t = acc_data_t_current - self.acc_data_t_previous

        # 1. Update state with IMU inputs
        ################ CORRECTION STEP #####################
        Cns = Quaternion(*self.q_est_prev).to_mat()
        p_est_current = self.p_est_prev + delta_t*self.v_est_prev + (delta_t**2) * 0.5*(matmul(Cns, acc_data_current) + self.g)
        v_est_current = self.v_est_prev + delta_t * (matmul(Cns, acc_data_current) + self.g)
        q_est_current = Quaternion(euler=delta_t * gyro_data_current).quat_mult_right(self.q_est_prev)

        # 1.1 Linearize the motion model and compute Jacobians
        F = np.eye(9)
        imu = acc_data_current.reshape((3, 1))
        F[0:3, 3:6] = np.eye(3) * delta_t
        F[3:6, 6:]  = matmul(Cns, -skew_symmetric(imu)) * delta_t

        # 2. Propagate uncertainty
        Q = np.eye(6)
        Q[0:3, 0:3] = self.var_acc * Q[0:3, 0:3]
        Q[3:6, 3:6] = self.var_gyro * Q[3:6, 3:6]
        Q = Q*(delta_t**2)
        p_cov_current = matmul(F, matmul(self.p_cov_prev, F.T)) + matmul(self.l_jac, matmul(Q, self.l_jac.T))
        
        # 3. Check availability of GPS measurements
        ################ PREDICTION STEP #####################
        if abs(gps_data_t - acc_data_t_current) < 0.01:
            # print("hey, a new GPS measurement")
            p_est_current, \
                v_est_current, \
                q_est_current, \
                    p_cov_current = self.measurement_update( self.var_gps, 
                                                                p_cov_current, 
                                                                gps_data_current, 
                                                                p_est_current, 
                                                                v_est_current, 
                                                                q_est_current)

        self.acc_data_t_previous = acc_data_t_current
        self.p_est_prev = p_est_current
        self.v_est_prev = v_est_current
        self.q_est_prev = q_est_current
        self.p_cov_prev = p_cov_current

        return p_est_current, v_est_current, q_est_current, p_cov_current