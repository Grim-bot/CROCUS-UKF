# -*- coding: utf-8 -*-
"""
Author: Felix Grimberg
Version: 2019-10-03
"""
import numpy as np
from scipy.linalg import expm
#from pykalman import KalmanFilter as KF
import config

class EKF:
    
    def __init__(self, point_reactor, tstep):
        self.point_reactor = point_reactor
        self.tstep = tstep
    
    def nvars(self):
        return self.point_reactor.nvars
    
    def filter(self, observations, state_initial, COV_initial):
        self.obs_mat = np.zeros(self.point_reactor.state_dims)
        self.obs_mat[0] = 1
        
        # Start by updating the initial guess with the initial measurement:
        state_upd, COV_upd = self.update(state_initial, COV_initial, observations[0])
        states = [state_upd]
        COVs = [COV_upd]
        
        for obs in observations[1:]:
            # Prediction step:
            state_pred = self.predict(states[-1])
            Jac = self.jacobian_pred(states[-1])
            
            # COV_pred = Jac @ COVs[-1] @ Jac.T + COV_transition
            COV_pred = Jac @ COVs[-1] @ Jac.T
            COV_pred[:self.nvars(), :self.nvars()] = COV_pred[:self.nvars(), :self.nvars()] + np.diag( np.power(
                    config.stdev_transition_dep * state_pred[:self.nvars()]
                    , 2))
            
            # Update step:            
            state_upd, COV_upd = self.update(state_pred, COV_pred, obs)
            states.append(state_upd)
            COVs.append(COV_upd)
        return np.array(states), np.array(COVs)
    
    def update(self, state_pred, COV_pred, obs):
        """Update step (eqs are same as linear KF):"""
        COV_obs = obs # obs = Poiss(obs)
        K = COV_pred @ self.obs_mat.T / (self.obs_mat @ COV_pred @ self.obs_mat.T + COV_obs)
        state_upd = state_pred + np.dot(K, obs - self.obs_mat @ state_pred)
        COV_upd = COV_pred - K.reshape((K.size,1)) @ self.obs_mat.reshape((1, self.obs_mat.size)) @ COV_pred
        return state_upd, COV_upd
        
    def predict(self, state):
        """
        INPUT:
            state at t0 = [n, c_1, ... c_6, rho, beta_1, ..., beta_6, lambda_1, ..., lambda_6, Lambda]
        
        OUTPUT:
            state at t0 + self.tstep
        """
        return self.point_reactor.rod_drop(
                (0, self.tstep),
                initial_state=state,
                store_time_steps=False,
                )
    
    def jacobian_pred(self, state):
        """
        
        Based on state = [n, c_1, ..., c_6, rho, beta_1, ..., beta_6, lambda_1, ..., lambda_6, Lambda]^T
        """
        # We have equations for dx/dt = f(x), for which we can compute the Jacobian. We obtain the Jacobian PHI with x^{(i|i-1)} approx.= PHI @ x^{(i-1|i-1)} by taking the matrix exponential of Jac_of_dx_dt*tstep .
        # The Jacobian will be of dimension state_dims x state_dims
        Jac_of_dx_dt = np.zeros([self.point_reactor.state_dims]*2)
        
        # dd/dt = A @ d where d = x[:nvars] are the dependent variables
        A = self.point_reactor.PRKE_matrix(state=state)
        # So del (dd/dt) / del d = A
        Jac_of_dx_dt[:self.nvars(), :self.nvars()] = A
        
        # Still missing del (dd/dt) / del e, del (de/dt) / del d, and del (de/dt) / del e, where e = x[nvars:] are the independent variables (estimated "parameters")
        
        # e are the independent variables. They are assumed constant. Hence:
        # de/dt = 0 --> del (de/dt) / del x = 0 <-- already 0 in Jac
        
        # dd/dt = A @ d, where A is a (nonlinear) function of the independent variables e. So we derived the expressions for the terms of del (dd/dt) / del e (by hand).
        
        slc_c_l = slice(1, self.nvars())
        
        n = state[0]
        c_l = state[slc_c_l]
        rho = state[self.point_reactor.ind_rho]
        beta_l = state[self.point_reactor.slc_beta_l]
        Lambda = state[self.point_reactor.ind_Lambda]
        
        # dn/dt = (rho - beta) / Lambda * n + lambda_l @­ beta_l
        # => del (dn/dt) / del rho = n / Lambda         ...etc
        Jac_of_dx_dt[0, self.point_reactor.ind_rho] = n / Lambda
        Jac_of_dx_dt[0, self.point_reactor.slc_beta_l] = - n / Lambda
        Jac_of_dx_dt[0, self.point_reactor.slc_lambda_l] = c_l
        Jac_of_dx_dt[0, self.point_reactor.ind_Lambda] = (beta_l.sum() - rho) * state[0] / Lambda**2
        
        # dc_l/dt = beta_l / Lambda * n - lambda_l * c_l
        Jac_of_dx_dt[slc_c_l, self.point_reactor.slc_beta_l] = n / Lambda * np.eye(self.point_reactor.fuel.ngroups())
        Jac_of_dx_dt[slc_c_l, self.point_reactor.slc_lambda_l] = np.diag(- c_l)
        Jac_of_dx_dt[slc_c_l, self.point_reactor.ind_Lambda] = - n / Lambda**2 * beta_l
        
        # PHI = matrix exponential of (Jac_of_dx_dt * tstep)
        return expm(Jac_of_dx_dt * self.tstep)
        

if __name__ == '__main__':
    config.include_reactivity = True
    config.process_noise_on_params = False
    config.noise_before = False
    config.compare_purely_model_based = True
    config.use_EKF = False
    config.use_UKF = True
    config.stdev_initial_factor = 0.5
    config.stdev_transition_dep = 1e-3
    
    import PRK as prk
    
    reactivity_unitless = 0.00112
    params = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed May 15 13:30:01 2019', '__version__': '1.0', '__globals__': [], 'BETA_MEAN': np.array([[7.354466e-03, 2.381485e-04, 1.261007e-03, 1.228657e-03,
        2.838228e-03, 1.263191e-03, 5.252351e-04]]), 'BETA_STD': np.array([[7.245201e-05, 1.211003e-05, 2.883927e-05, 2.757947e-05,
        4.144158e-05, 2.897984e-05, 1.774963e-05]]), 'LAMBDA_MEAN': np.array([[4.986124e-01, 1.335352e-02, 3.261235e-02, 1.210585e-01,
        3.056654e-01, 8.610382e-01, 2.892019e+00]]), 'LAMBDA_STD': np.array([[6.465619e-03, 3.066375e-06, 9.880134e-06, 2.044211e-05,
        1.411160e-04, 6.096978e-04, 2.933295e-03]]), 'LIFETIME_MEAN': np.array([[4.717137e-05, 4.715002e-05, 5.005594e-05]]), 'LIFETIME_STD': np.array([[1.827880e-07, 1.833994e-07, 4.855028e-07]]), 'GEN_TIME_MEAN': np.array([[4.686777e-05, 4.684656e-05, 4.973400e-05]]), 'GEN_TIME_STD': np.array([[9.682227e-08, 9.715106e-08, 4.796570e-07]])}
    crocus = prk.PointReactor(params['LAMBDA_MEAN'].flatten()[1:], params['BETA_MEAN'].flatten()[1:], params['GEN_TIME_MEAN'].flatten()[0])
    crocus.set_include_params(True)
    uncertainties = prk.Fuel(params['LAMBDA_STD'].flatten()[1:], params['BETA_STD'].flatten()[1:], params['GEN_TIME_STD'].flatten()[0])
    
    ekf = EKF(crocus, 1e-1)
    state_initial = crocus.stationary_PRKE_solution(24, reactivity=reactivity_unitless)
    print(state_initial)
    jac_initial = ekf.jacobian_pred(state_initial)
    
    observations = [24, 25, 26, 28]
    COV_initial = np.diag( np.power(
            np.concatenate((
                    state_initial[:crocus.nvars] * config.stdev_initial_factor,
                    uncertainties.param_values(5.883572956799998e-05)
                    )),
            2) )
    
    states, COVs = ekf.filter(observations, state_initial, COV_initial)
    
    
### The following is wrong!
#        fac1 = (rho - beta_l.sum()) * self.tstep / Lambda
#        fac2 = state[0] * self.tstep / Lambda
#        dn_by_dlambda_l = c_l * self.tstep * np.exp(lambda_l * self.tstep)
#        dc_l_by_dbeta_l = fac2 * np.exp(beta_l * self.tstep / Lambda)
#        
#        Jac[0, self.point_reactor.ind_rho] = fac2 * np.exp(fac1)
#        Jac[0, self.point_reactor.slc_beta_l] = -1 * Jac[0, self.point_reactor.ind_rho]
#        Jac[0, self.point_reactor.slc_lambda_l] = dn_by_dlambda_l
#        Jac[0, self.point_reactor.ind_Lambda] = -1 * Jac[0, self.point_reactor.ind_rho] * fac1 / self.tstep
#        
#        Jac[1:self.nvars(), self.point_reactor.slc_beta_l] = np.diag( dc_l_by_dbeta_l )
#        Jac[1:self.nvars(), self.point_reactor.slc_lambda_l] = - dn_by_dlambda_l
#        Jac[1:self.nvars(), self.point_reactor.ind_Lambda] = -1 * beta_l / Lambda * dc_l_by_dbeta_l
#        return Jac