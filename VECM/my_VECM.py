import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from numpy.linalg import inv,multi_dot
from numpy import dot
import scipy.stats as sstats
import scipy
from statsmodels.iolib.summary import Summary

import sys
sys.path.append(r'C:\Users\secret\Desktop\Python\Sublime\Dissertation\statsmodels_vecm')

from sm_coint_tables import c_sja, c_sjt


############
# Statsmodels imports
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import summary_params

############
from np_array_to_bmatrix import bmatrix

############

##################
# Some notation and snippets of code are borrowed from the statsmodels package.
# This was really only the case for the _endog_matrices() function.
# The statsmodels package can be found at https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/vector_ar/vecm.py
##################


def _endog_matrices(endog, exog=None, exog_coint=None, diff_lags=1, deterministic='nc',
                    seasons=0, first_season=0):
    """
    Returns different matrices needed for parameter estimation.

    Compare p. 186 in [1]_. The returned matrices consist of elements of the
    data as well as elements representing deterministic terms. A tuple of
    consisting of these matrices is returned.

    Parameters
    ----------
    endog : ndarray (neqs x nobs_tot)
        The whole sample including the presample.
    exog: ndarray (nobs_tot x neqs) or None
        Deterministic terms outside the cointegration relation.
    exog_coint: ndarray (nobs_tot x neqs) or None
        Deterministic terms inside the cointegration relation.
    diff_lags : int
        Number of lags in the VEC representation.
    deterministic : str {``"nc"``, ``"co"``, ``"ci"``, ``"lo"``, ``"li"``}
        * ``"nc"`` - no deterministic terms
        * ``"co"`` - constant outside the cointegration relation
        * ``"ci"`` - constant within the cointegration relation
        * ``"lo"`` - linear trend outside the cointegration relation
        * ``"li"`` - linear trend within the cointegration relation

        Combinations of these are possible (e.g. ``"cili"`` or ``"colo"`` for
        linear trend with intercept). See the docstring of the
        :class:`VECM`-class for more information.
    seasons : int, default: 0
        Number of periods in a seasonal cycle. 0 (default) means no seasons.
    first_season : int, default: 0
        The season of the first observation. `0` means first season, `1` means
        second season, ..., `seasons-1` means the last season.

    Returns
    -------
    y_1_T : ndarray (neqs x nobs)
        The (transposed) data without the presample.
        `.. math:: (y_1, \\ldots, y_T)
    delta_y_1_T : ndarray (neqs x nobs)
        The first differences of endog.
        `.. math:: (y_1, \\ldots, y_T) - (y_0, \\ldots, y_{T-1})
    y_lag1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        Endog of the previous period (lag 1).
        `.. math:: (y_0, \\ldots, y_{T-1})
    delta_x : ndarray (k_ar_diff*neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        Lagged differenced endog, used as regressor for the short term
        equation.

    References
    ----------
    .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.

    """
    # p. 286:

    ########
    #New code
    ########

    def create_lag_mask(endog, p):
        # This function enables all the differencing and shifting preprocessing operations to
        # be performed on the data before the mask is applied leading to no missing values
        mat = endog.T
        lagged = [mat]
        for i in range(1,p+1):
            tmp = np.roll(mat,i, axis = 0)
            tmp[:i, :]=np.nan 
            lagged.append(tmp)

        mat = np.concatenate(lagged, axis =1)
        mask = np.isnan(mat).any(axis=1)
        return ~mask

    mask = create_lag_mask(endog, diff_lags)
    # Y = endog[[::-1]] # endogenous variables but most recent first
    p = diff_lags+1 # The plus 1 accounts for at least 1 shift of data and hence the loss of
                    #  at least one observation when estimating 
    # y = endog
    y = endog.T[mask].T # K x T
    K = y.shape[0]
    y_1_T = y[:, p-1:] # T for truncated to allow for the fact we lose data when we lag observations
    T = y_1_T.shape[1]
    # delta_y = np.diff(y)
    delta_y = np.diff(endog).T[mask[1:]].T # K x T
    # delta_y_1_T = delta_y[:, p-1:]
    delta_y_1_T = delta_y[:, p-1:]

    y_lag1 = y[:, p-2:-1]

    if "co" in deterministic and "ci" in deterministic:
        raise ValueError("Both 'co' and 'ci' as deterministic terms given. " +
                         "Please choose one of the two.")
    y_lag1_stack = [y_lag1]
    if "ci" in deterministic:  # pp. 257, 299, 306, 307
        y_lag1_stack.append(np.ones(T))
    if "li" in deterministic:  # p. 299
        y_lag1_stack.append(np.arange(endog.shape[1])[mask][p-1:] + p)
    if exog_coint is not None:
        y_lag1_stack.append(exog_coint[-T-1:-1].T)
    y_lag1 = np.row_stack(y_lag1_stack)

    # p. 286:
    delta_x = np.zeros((diff_lags*K, T))
    if diff_lags > 0:
        for j in range(delta_x.shape[1]):
            delta_x[:, j] = (delta_y[:, j+p-2:None if j-1 < 0 else j-1:-1]
                             .T.reshape(K*(p-1)))
    delta_x_stack = [delta_x]
    # p. 299, p. 303:
    if "co" in deterministic:
        delta_x_stack.append(np.ones(T))
    if seasons > 0:
        delta_x_stack.append(seasonal_dummies(seasons, delta_x.shape[1],
                                              first_period=first_season + diff_lags + 1,
                                              centered=True).T)
    if "lo" in deterministic:
        delta_x_stack.append(np.arange(endog.shape[1])[mask][p-1:] + p)
    if exog is not None:
        delta_x_stack.append(exog[-T:].T)
    delta_x = np.row_stack(delta_x_stack)

    return y_1_T, delta_y_1_T, y_lag1, delta_x, mask


def coint_johansen(endog, diff_lags, deterministic='nc', method = 'max_eig', exog = None, exog_coint = None):
    

    y_1_T, delta_y_1_T, y_lag1, delta_x, mask = _endog_matrices(endog = endog, exog = exog, exog_coint=exog_coint, deterministic=deterministic, diff_lags=diff_lags)
        
    T = y_1_T.shape[1]

    M,r0,r1 = _r_matrices(y_1_T, delta_y_1_T, y_lag1, delta_x, T)

    s00, s01, s10, s11, s11neghalf, lambd, u = _sij(r0,r1,T)

    if deterministic =='nc':
        det_order =-1
    elif deterministic =='ci':
        det_order = 0
    else:
        det_order = 1

    if (method =='max_eig') | (method != 'trace'):
        crit_vals = c_sja(endog.shape[0], det_order)

        # From Lutkepohl
        #initialise significance of test to 0 zero so it can be done iteratively until significance above threshold
        significance= 0
        for idx in range(endog.shape[0]):

            coint_rank = idx
            t_stat = - T *np.math.log(1 -lambd[idx]) #in this case lambda_i from lutkepohl is lambd[i-1]
            print("Johansen's cointegration test Statistic for r="+str(coint_rank)+":", t_stat)
            old_significance = significance
            significance = sum(t_stat >= crit_vals)


            if significance >=1:
                continue

            else:
                break

    ### trace method currently unavailable

    return coint_rank, old_significance, t_stat


def _r_matrices(y_1_T, delta_y_1_T, y_lag1, delta_x, T):
    M = np.identity(T) - multi_dot([delta_x.T, inv(dot(delta_x, delta_x.T)), delta_x])
    r0 = dot(delta_y_1_T, M)
    r1 = dot(y_lag1, M)

    return M, r0, r1

def _sij(r0, r1, T):

    s00 = dot(r0, r0.T) / T
    s01 = dot(r0, r1.T) / T
    s10 = s01.T
    s11 = dot(r1, r1.T) / T

    s11neghalf = inv(sqrtm(s11))
    SS = multi_dot([s11neghalf,s10,inv(s00), s01, s11neghalf])
    lambd = np.linalg.eig(SS)[0]
    u = np.linalg.eig(SS)[1]
    lambd_order = np.argsort(lambd)[::-1]
    lambd = lambd[lambd_order]
    u = u[:,lambd_order]

    return s00, s01, s10, s11, s11neghalf, lambd, u


def select_VAR_order(endog, deterministic, max_lags = 10):

    #List to store BICs
    BICs = []
    AICs = []
    HQs = []

    for p in range(1,max_lags+1):
        VAR = My_VAR(endog = endog, p=p, deterministic= deterministic)
        VAR = VAR.ML_fit()

        BICs.append(VAR.BIC)
        AICs.append(VAR.AIC)
        HQs.append(VAR.HQ)

    BICs = np.array(BICs)
    best_p = np.argmin(BICs) + 1

    BAH = [BICs[best_p-1], AICs[best_p-1], HQs[best_p-1]]

    return best_p, BICs, BAH


class My_VECM(object):
    def __init__(self, endog, exog = None, exog_coint= None, diff_lags =1 , deterministic='nc', johansen_method = 'max_eig', names=None):
        if isinstance(endog, type(np.array([]))):
            self.endog = endog.T
            self.dates = None
        elif isinstance(endog, type(pd.DataFrame([]))):
            self.endog = endog.values.T
            self.index = endog.index
        self.exog = exog
        self.exog_coint = exog_coint
        self.diff_lags = diff_lags
        self.deterministic = deterministic
        self.johansen_method = johansen_method
        self.seasons = 0
        self.lag_order_tested = False
        self.granger_tested = False
        self.johansen_tested = False
        if names:
            self.names = names
        else:
            self.names = [str(i) for i in range(self.endog.shape[0])]
    

    def ML_estimate_VECM(self, endog, exog = None, exog_coint= None, diff_lags =1 , deterministic='nc'):

        y_1_T, delta_y_1_T, y_lag1, delta_x, mask = _endog_matrices(endog = endog, exog = exog, exog_coint=exog_coint, deterministic=deterministic, diff_lags=diff_lags)
        self.mask = mask

        T = y_1_T.shape[1]

        M,r0,r1 = _r_matrices(y_1_T, delta_y_1_T, y_lag1, delta_x, T)

        s00, s01, s10, s11, s11neghalf, lambd, u = _sij(r0,r1,T)

        self.coint_rank, self.coint_significance, self.coint_tvalue = coint_johansen(endog = self.endog, diff_lags =self.diff_lags, deterministic=self.deterministic, method = self.johansen_method)
        self.johansen_tested = True

        r = self.coint_rank
        self.r = r

        beta = np.dot(u[:,:self.r].T,s11neghalf)
        beta= dot(beta.T,inv(beta[:,:self.r]))
        self.beta = beta
        alpha = multi_dot([s01, beta, inv(multi_dot([beta.T,s11, beta]))])
        self.alpha = alpha
        gamma = multi_dot([(delta_y_1_T - multi_dot([alpha, beta.T ,y_lag1])),delta_x.T,inv(dot(delta_x,delta_x.T))])
        self.gamma = gamma


        #asymptotics
  
        R11 = r1[:r,:]
        R12 = r1[r:,:]
        tmp = (delta_y_1_T - multi_dot([alpha, beta.T, y_lag1]) - dot(gamma, delta_x))
        sigma_u = dot(tmp,tmp.T)/T

        RRinv = inv(sqrtm(dot(R12, R12.T)))   
        sigma_betaRRhalf = np.kron(np.identity(beta[r:].shape[0]),
                                inv(multi_dot([alpha.T, inv(sigma_u), alpha])))
        beta_dist_var = RRinv.dot(sigma_betaRRhalf).dot(RRinv)
        
        std_err_beta = np.sqrt(beta_dist_var.diagonal())
        self.std_err_beta = std_err_beta

        t_vals_beta = beta/self.std_err_beta

        self.t_vals_beta = t_vals_beta
        z_normal = sstats.norm(0,1)

        p_vals_beta = 2*(1- z_normal.cdf(np.abs(self.t_vals_beta)))
        p_vals_beta[:r] =0
        self.p_vals_beta = p_vals_beta      

        return self

    def EngleGrangerLS_estimate_VECM(self, endog, exog = None, exog_coint= None, diff_lags =1 , deterministic='nc'):
        y_1_T, delta_y_1_T, y_lag1, delta_x, mask = _endog_matrices(endog = endog, exog = exog, exog_coint=exog_coint, deterministic=deterministic, diff_lags=diff_lags)
        self.y_1_T, self.delta_y_1_T, self.y_lag1, self.delta_x, self.mask = y_1_T, delta_y_1_T, y_lag1, delta_x, mask


        T = y_1_T.shape[1]
        self.T = T
        
        K = y_1_T.shape[0]
        self.K = K
        
        p=diff_lags + 1
        self.p = p

        #Calculate the matrix pi and gamma concatenated
        M1 = np.concatenate([dot(delta_y_1_T, y_lag1.T), dot(delta_y_1_T, delta_x.T)],axis=1)

        tmp1 = np.concatenate([dot(y_lag1, y_lag1.T), dot(y_lag1,delta_x.T)],axis=1)
        tmp2 = np.concatenate([dot(delta_x, y_lag1.T), dot(delta_x,delta_x.T)],axis=1)
        M2 = np.concatenate([tmp1,tmp2],axis=0)

        pi_gamma = dot(M1, inv(M2))
        

        #separate out pi and gamma
        if deterministic =='nc':
            pi = pi_gamma[:,:K]
            gamma = pi_gamma[:, K:]
        elif deterministic =='ci':
            pi = pi_gamma[:,:K+1]
            gamma = pi_gamma[:, K+1:]
        elif deterministic =='li':
            pi = pi_gamma[:,:K+1]
            gamma = pi_gamma[:, K+1:]
        elif deterministic =='co':
            pi = pi_gamma[:,:K]
            gamma = pi_gamma[:, K:]
        else:
            raise ValueError('Deterministic has an unrecognised value')

        self.pi = pi
        self.gamma1s = gamma
        #Calculate alpha and beta

        #Normalise

        #Calculate the covariance matrix of the errors
        U = delta_y_1_T - dot(pi, y_lag1) - dot(gamma,delta_x)

        sigma_u = (1/(T- K *p)) * dot(U, U.T )
        self.sigma_u = sigma_u

        # Calculate the covariance matrix of the estimated coefficients
        sigma_co = np.kron(T * inv(M2), sigma_u) # the diagonals are the variances of coefficients
        self.sigma_co = sigma_co

        #calculate Alpha and Beta assuming coint rank, having calculated cointegration rank
        self.coint_rank, self.coint_significance, self.coint_tvalue = coint_johansen(endog = self.endog, diff_lags =self.diff_lags, deterministic=self.deterministic, method = self.johansen_method)
        self.johansen_tested = True
        r = self.coint_rank
        self.r = r
        
        alpha = pi[:, :r].reshape((-1, r)) # first r rows of pi as estimator for alpha
        self.alpha1s = alpha

        # R matrices
        M = np.identity(T) - multi_dot([delta_x.T, inv(dot(delta_x, delta_x.T)), delta_x])
        R0= dot(delta_y_1_T, M)
        R1 = dot(y_lag1,M)

        R11 = R1[:r,:]
        R12 = R1[r:,:]
        self.R11 = R11
        self.R12 = R12

        # Beta
        beta= multi_dot([inv(multi_dot([alpha.T, inv(sigma_u), alpha])),
                         alpha.T,
                         inv(sigma_u),
                         (R0 - dot(alpha,R11)),
                         R12.T,
                         inv(dot(R12, R12.T))
                        ])
        beta = np.concatenate([np.identity(r),beta.T]) # this is the correct shape for beta K x r

        self.beta = beta
        
        #########
        #asymptotics for beta
        #########
        # The Version in Luktepohl
#         beta_dist_val = np.ravel(dot(beta[r:].T , sqrtm(dot(R12, R12.T))))
#         beta_dist_var = np.kron(np.identity(beta[r:].shape[0]),
#                                 inv(multi_dot([alpha.T, inv(sigma_u), alpha])))
#         t_vals = beta_dist_val.T/np.sqrt(beta_dist_var.diagonal())
        
        # The transformed version in statsmodels using multivariate normal affine transformations
        RRinv = inv(sqrtm(dot(R12, R12.T)))  

        sigma_betaRRhalf = np.kron(np.identity(beta[r:].shape[0]),
                                inv(multi_dot([alpha.T, inv(sigma_u), alpha])))
        beta_dist_var = RRinv.dot(sigma_betaRRhalf).dot(RRinv.T)
        self.sigma_beta = beta_dist_var
        
        std_err_beta = np.sqrt(beta_dist_var.diagonal())
        
        self.std_err_beta = np.concatenate([np.zeros((r,r)),std_err_beta.reshape((-1,r))],axis=0)
        
        t_vals_beta = beta/self.std_err_beta

        self.t_vals_beta = t_vals_beta
        z_normal = sstats.norm(0,1)

        p_vals_beta = 2*(1- z_normal.cdf(np.abs(self.t_vals_beta)))
        p_vals_beta[:r] =0
        self.p_vals_beta = p_vals_beta

        #########
        # 2SLS for alpha and gamma
        #########
        
        alpha2s = multi_dot([delta_y_1_T,
                             M,
                             y_lag1.T,
                             beta,
                             inv(multi_dot([beta.T,
                                           y_lag1,
                                           M,
                                           y_lag1.T,
                                           beta]))
                            ])
        
        gamma2s = multi_dot([(delta_y_1_T - alpha2s.dot(beta.T).dot(y_lag1)),
                             delta_x.T,
                             inv(delta_x.dot(delta_x.T))
                            ])
        
        # Parameter distributions and test statistics
        tmp11 = multi_dot([beta.T,y_lag1,y_lag1.T, beta])
        tmp12 = multi_dot([beta.T,y_lag1, delta_x.T])
        tmp1 = np.concatenate([tmp11,tmp12], axis=1)
                              
        tmp21 = tmp12.T
        tmp22 = multi_dot([delta_x,delta_x.T])
        tmp2 = np.concatenate([tmp21,tmp22], axis=1) 
        
        omega = (1/T) * np.concatenate([tmp1,tmp2],axis=0)
        
        
        sigma_alpha_gamma = np.kron(inv(omega), sigma_u)
        self.sigma_alpha_gamma = sigma_alpha_gamma
        
        alpha_gamma2s = np.ravel(np.concatenate([alpha2s, gamma2s], axis=1))
        
        std_err_alpha_gamma = np.sqrt(sigma_alpha_gamma.diagonal()/T)

        num_alpha_params = alpha2s.shape[0] * alpha2s.shape[1]
        
        self.alpha = alpha2s
        self.gamma = gamma2s
        
        self.std_err_alpha = std_err_alpha_gamma[:num_alpha_params].reshape((-1,r))

        if 'co' in deterministic or 'lo' in deterministic:
            col_dim_gamma = K+1
        else:
            col_dim_gamma = K
        self.col_dim_gamma = col_dim_gamma

        self.std_err_gamma = std_err_alpha_gamma[num_alpha_params:].reshape((col_dim_gamma *(p-1), K)).T 

        self.t_vals_alpha = self.alpha/self.std_err_alpha
        self.t_vals_gamma = self.gamma / self.std_err_gamma

        
        self.p_vals_alpha = 2*(1-z_normal.cdf(np.abs(self.t_vals_alpha)))
        self.p_vals_gamma = 2*(1-z_normal.cdf(np.abs(self.t_vals_gamma)))


        U = delta_y_1_T - dot(pi, y_lag1) - dot(gamma2s,delta_x)
        self.U = U
        
        #############################


        
        return self
    
    def fit(self, how = 'EGLS', select_order = True):
        self.how_fit = how

        if select_order==True:
            p, VAR_BICs, BAH = select_VAR_order(endog = self.endog, deterministic = self.deterministic)
            self.lag_order_tested = True
        else:
            p, VAR_BICs, BAH = select_VAR_order(endog = self.endog, deterministic = self.deterministic)
            self.lag_order_tested = True
            p= self.diff_lags+1

        if p<= 1: #On the occasion that the optimal VECM has no lags (VAR(1)) the estimation won't work
            print('Optimal var order was 1 but standard VECM estimation will not work in this case so continuing with VAR(2)')
            p=2

        self.p =p
        self.diff_lags= p-1
        self.VAR_BICs = np.concatenate([np.arange(1,11).reshape((1,-1)),VAR_BICs.reshape((1,-1))],axis=0)
        self.BAH = BAH
        
        if how == 'EGLS':
            self.EngleGrangerLS_estimate_VECM(endog = self.endog, exog = self.exog, exog_coint= self.exog_coint, diff_lags =self.diff_lags , deterministic=self.deterministic)
        elif how =='ML':
            self.ML_estimate_VECM(endog = self.endog, exog = self.exog, exog_coint= self.exog_coint, diff_lags =self.diff_lags , deterministic=self.deterministic)
        else:
            raise ValueError('Estimation method not recognised')
        
        return self




    def granger_causality_test(self, causal_index, verbose=False):
        # Causal index (int, array-like)
        # if causal index is 0 then test 0 granger causing everything else
        p = self.p
        K= self.K
        T=self.T
        
        W = np.zeros((K*p, K*p))
        for i in range(p):
            Ki = K*i # so that we can work in index steps of K instead of 1
            if i ==0:
                W[Ki:K+Ki,Ki:K+Ki] = np.identity(K)
                W[Ki+K: K+Ki+K,Ki:K+Ki] = np.identity(K)
            else:
                W[Ki:K+Ki,Ki:K+Ki] = -1 * np.identity(K)
            if Ki >= K*(p-1): # K(p-1) is the index of the last row of submatrices 
                continue
            else:
                W[Ki+K:K+Ki+K,Ki:K+Ki] = np.identity(K)

        J = np.zeros((K, K*p))
        J[:K,:K] = np.identity(K)

        pi_gamma = np.concatenate([ self.pi[:,:K], self.gamma[:,:K*(p-1)] ],axis=1)

        A = pi_gamma.dot(W) + J
        A_vec = A.flatten('F').reshape((-1,1))

        ### Form the matrix X
        y_1_T = self.y_1_T

        X = np.zeros((p*K, T-1))
        
        for j in range(T-1):
            try:
                X[:, j] = y_1_T[:, j+p-1:None if j-1 < 0 else j-1:-1].T.reshape(K*p) # else j-1 grabs up to index j because indexing non-inclusive
            except:
                break
        X_stack = [X]
        # p. 299, p. 303:
        if "co" in self.deterministic:
            X_stack.append(np.ones(X.shape[1]))

        X = np.row_stack(X_stack)

        U_VAR = y_1_T[:,:T-1] - dot(A,X)

        sigma_co_alpha = np.kron(inv(dot(X, X.T)), dot(U_VAR, U_VAR.T))


        #Form testing matrix C:
        causal_index = [causal_index]

        for c in causal_index:
            C = np.zeros((K, p*K))
            for i in range(p):
                Ki = K*i
                for j in range(K):
                    #letting causal index be c then want A_jc,i = 0 for all i in order for c not to cause j
                    #example: if c = 0, 3 cointegrating series (K=3) and 2 lags in VECM (p=3 as VECM(2)= VAR(3)) we want to check :
                    #       |A_11,1 A_12,1  A_13,1|A_11,2   A_12,2  A_13,2|A_11,3   A_12,3  A_13,3|
                    # A =   |0      A_22,1  A_23,1|0        A_12,2  A_13,2|0        A_12,3  A_13,3|
                    #       |0      A_23,1  A_33,1|0        A_12,2  A_13,2|0        A_12,3  A_13,3|
                    # So C must match up wth A so that C_vec matches up with A_vec
                    #       |0      0       0   |0      0       0   |0      0       0|
                    # C =   |1      0       0   |1      0       0   |1      0       0|
                    #       |1      0       0   |1      0       0   |1      0       0|
                    #
                    # C_vec = [011,000,000,0111,...] = vec(C)
                    #
                    if j!= c:
                        C[j, Ki+c] =1
            C_vec = C.flatten(order = 'F').reshape((1,-1))
        lambd_wald = T * multi_dot([A_vec.T,C_vec.T, (1/multi_dot([C_vec,sigma_co_alpha,C_vec.T])), C_vec, A_vec])

        gc_pval = 1 - sstats.chi2(1).cdf(lambd_wald)

        if verbose:
            print('Test statistic lambda_w:', lambd_wald, '\n', 'p-value for chi-squared {} degrees of freedom:'.format(1), gc_pval)

        self.granger_tested = True

        return self, lambd_wald, gc_pval

    def test_instantaneous_causality(self , sigma_u = None):
        if not sigma_u:
            sigma_u = self.sigma_u
            K= self.K
        else:
            sigma_u = sigma_u
            K = sigma_u.shape[0]
        #Lutkepohl Instantaneous causality - check whether sigma_u is diagonal
        def duplication_matrix(n):
            D= np.zeros(( int(n**2),int(n*(n+1)/2) ))
            coords_dict = {}
            
            new_el_counter = 0
            for j in range(n): # then across columns
                for i in range(n): # move down rows first ^
                    idx_i =i + n*j
                    if j<=i: # lower triangle of matrix we want to be going across
                        coords_dict[str(i)+str(j)] = new_el_counter # column of i,j element where i<=j
                        idx_j = new_el_counter
                        D[idx_i, idx_j]=1
                        new_el_counter +=1 # this puts 1 in the next new column every time we encounter a new element

                    elif j>i:# if j>i i.e. Above the diagonal i.e. already encountered elements
                        #need to find column in duplication matrix of element [i,j] = element [j,i] in old matrix 
                        
                        idx_j = coords_dict[str(j)+str(i)] # finds the column of i,j
                        D[idx_i,idx_j]= 1
                    else:
                        print('something wrong')
            
            return D

        D_plus = np.linalg.pinv(duplication_matrix(K))
        dist_sigma_u = 2* multi_dot([D_plus, np.kron(sigma_u,sigma_u), D_plus.T])/self.T
        std_err_sigma_u = np.sqrt(dist_sigma_u.diagonal())
        t_vals_sigma_u = sigma_u[np.tril_indices(K)]/std_err_sigma_u

        distribution = sstats.norm(0,1)
        p_vals_sigma_u = 1 - distribution.cdf(np.abs(t_vals_sigma_u))

        self.t_vals_sigma_u = t_vals_sigma_u
        self.std_err_sigma_u = std_err_sigma_u
        self.p_vals_sigma_u = p_vals_sigma_u
        self.instantaneous_causality_tested= True

        return self, std_err_sigma_u, t_vals_sigma_u,p_vals_sigma_u


    def test_autocorrelation(self, h = 4, series_index = 0):

        tmp = np.ones((self.K,len(self.mask)))*-50 # length of endog
        autocovs = []
        for i in range(h+1):
            tmp[:,self.mask] = np.hstack([np.ones((self.K,self.p-1))*-50, self.U]) # shape of only values after p-1 are assigned. self.U shape is same as y_1_T (the indexing is taken from the creation of y_1_T from mask)
            tmp_mask = (tmp==float(-50)).any(axis=0)
            tmp_mask = ~tmp_mask.reshape((1,-1))
            tmp_mask_stack = np.concatenate([np.roll(tmp_mask, j,axis=1) for j in range(i+1)],axis=0)
            tmp_mask = tmp_mask_stack.all(axis=0)
            
            U_nonull = tmp[:,tmp_mask]
            
            autocovs.append((1/sum(tmp_mask))*np.dot(U_nonull[:,i:], np.roll(U_nonull, i, axis=1)[:,i:].T))
            
        autocovs = np.concatenate(autocovs,axis=1)
        C0 = autocovs[:,:self.K]
        c_h = autocovs.flatten(order='F')
        C_h = np.sqrt(self.T) * autocovs.flatten(order='F').reshape((-1,1))[4:]

        D = np.diag(np.sqrt(autocovs[:,:2].diagonal()))

        autocorrs = []
        for i in range(h+1):
            autocorrs.append(np.linalg.multi_dot([np.linalg.inv(D),autocovs[:,2*i:2*i+2], np.linalg.inv(D)]))
        autocorrs =np.concatenate(autocorrs, axis=1)
            
            

        sigma_C_h = np.kron(np.kron(np.identity(h), self.sigma_u), self.sigma_u)

        autocov_tvals = C_h.T/np.sqrt(sigma_C_h.diagonal())

        B_test = np.zeros((h*4))
        tmp = [4*i + (self.K+series_index)*series_index for i in range(h) ]
        mask_vals = [el for el in tmp]
        B_test[mask_vals] =1

        chi_val = np.dot(B_test,C_h)*(1/B_test.dot(sigma_C_h).dot(B_test.T))*np.dot(B_test,C_h).T
        p_val = 1- sstats.chi2(1).cdf(chi_val)

        self.chi_val_autocorr = chi_val
        self.p_val_autocorr = p_val

        portmanteau_stat = C_h.T.dot(np.kron(np.kron(np.identity(h), inv(C0)),inv(C0))).dot(C_h)
        self.portmanteau_stat = portmanteau_stat

        portmanteau_p_val = sstats.chi2(h*self.K^2 - self.K^2*(self.p-1) - self.K*self.r).cdf(chi_val)
        self.portmanteau_p_val = portmanteau_p_val

        return self, chi_val, p_val, portmanteau_stat

    def long_run_impulse_response(self):
        beta_svd = np.linalg.svd(self.beta[:self.K])
        beta_orthocomp = beta_svd[0][:,self.beta.shape[1]:]

        alpha_svd = np.linalg.svd(self.alpha)
        alpha_orthocomp = alpha_svd[0][:,self.alpha.shape[1]:]

        sum_gamma = np.zeros((self.K, self.K))
        for i in range(self.p-1):
            sum_gamma+=self.gamma[:,i*self.K:(i+1)*self.K]


        Xi = beta_orthocomp.dot(
            inv(alpha_orthocomp.T.dot(np.identity(self.K) - sum_gamma).dot(beta_orthocomp))
            ).dot(alpha_orthocomp.T)
        print('The long run response of each variable to a shock on each variable is given by the matrix: \n', Xi)
        self.Xi = Xi

        B = np.linalg.cholesky(self.sigma_u)
        self.B = B

        Xi_orthog = Xi.dot(B)
        self.Xi_orthog = Xi_orthog
        print('The long run response of each variable to an orthogonal shock on each variable is given by the matrix: \n', Xi_orthog)



        return self

    def structural_model(self, matrix_representation=True):
        # This is only for the 2x2 case
        # Based off of an Eichenbaum and Green (1993) representation using Lutkepohl's suggestion of a cholesky decomposition.
        
        B = np.linalg.cholesky(self.sigma_u)
        self.B = B
        A_0 = np.linalg.inv(B)
        self.A_0 = A_0

        self.gamma_struct = A_0.dot(self.gamma)
        self.pi_struct = A_0.dot(self.pi)
        self.alpha_struct = A_0.dot(self.alpha)

        #### alpha_gamma_struct variance
        num_alpha_params = self.alpha.shape[0]* self.alpha.shape[1]
        num_gamma_params = self.gamma.shape[0] * self.gamma.shape[1]

        self.sigma_alpha_gamma_struct = np.kron(np.identity(self.alpha.shape[1] + self.gamma.shape[1]),A_0).T.dot(self.sigma_alpha_gamma).dot(np.kron(np.identity(self.alpha.shape[1] + self.gamma.shape[1]),A_0))

        self.std_err_alpha_struct = (self.sigma_alpha_gamma_struct.diagonal()[:num_alpha_params]/self.T).reshape((-1,self.r))
        self.std_err_gamma_struct = (self.sigma_alpha_gamma_struct.diagonal()[num_alpha_params:]/self.T).reshape((self.col_dim_gamma *(self.p-1), self.K)).T 


        self.t_vals_alpha_struct = self.alpha_struct/self.std_err_alpha_struct
        self.t_vals_gamma_struct = self.gamma_struct / self.std_err_gamma_struct

        
        z_normal = sstats.norm(0,1)
        self.p_vals_alpha_struct = 2*(1-z_normal.cdf(np.abs(self.t_vals_alpha_struct)))
        self.p_vals_gamma_struct = 2*(1-z_normal.cdf(np.abs(self.t_vals_gamma_struct)))

        ##############
        # Summary of structural model
        ##############

        self.model = None #place holder that statsmodels summary_params needs
        summary = Summary()

        def make_table(self, params, std_err, t_values, p_values, conf_int,
                        names, title, strip_end=True, alpha=0.05):
            res = (self,
                   params,
                   std_err,
                   t_values,
                   p_values,
                   conf_int
                   )
            param_names = [
                '.'.join(name.split('.')[:-1]) if strip_end else name
                for name in np.array(names).tolist()]
            return summary_params(res, yname=None, xname=param_names,
                                  alpha=alpha, use_t=False, title=title)

        def confint(params, std_err, alpha = 0.05, dist = 'normal', **kwargs):
            if dist == 'normal':
                distribution = sstats.norm(0,1)
            elif dist =='chi2':
                try:    
                    distribution = sstats.chi2(degfree)
                except:
                    raise ValueError('Need to specify the degrees of freedom in paramater degfree')
            else:
                distribution = sstats.norm(0,1)

            upper = (params + std_err * distribution.ppf(1-alpha/2)).flatten(order='F').reshape((-1,1))
            lower = (params - std_err * distribution.ppf(1-alpha/2)).flatten(order='F').reshape((-1,1))
            confints = np.concatenate([lower, upper],axis=1)
            return confints

        ### Structural Gamma
        params = self.gamma_struct.T.flatten(order='F')
        NAs = [0 for i in range(len(params))]
        std_err= self.std_err_gamma_struct.T.flatten(order='F')
        t_values= self.t_vals_gamma_struct.T.flatten(order='F')
        p_values = self.p_vals_gamma_struct.T.flatten(order='F')
        conf_int= confint(self.gamma_struct.T, self.std_err_gamma_struct.T)
        names = [self.names[i2]+' L' + str(j+1) +'.'+self.names[i1]  for i2 in range(self.K) for j in range(self.p-1) for i1 in range(self.K)]
        title = 'Structural Lagged parameter coefficients (gamma)'

        table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
        summary.tables.append(table)
        #Table for structural alpha
        params = self.alpha_struct.T.flatten(order='F')
        NAs = [0 for i in range(len(params))]
        std_err= self.std_err_alpha_struct.T.flatten(order='F')
        t_values= self.t_vals_alpha_struct.T.flatten(order='F')
        p_values = self.p_vals_alpha_struct.T.flatten(order='F')
        conf_int= confint(self.alpha_struct.T, self.std_err_alpha.T)
        names = ['alpha.'+str(i+1)+'.'+str(j+1) for i in range(self.alpha_struct.shape[0]) for j in range(self.alpha_struct.shape[1])]
        title = 'Structural Error correction loading coefficients (alpha)'

        table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
        summary.tables.append(table)

        print(summary)

        ##############
        # Matrix representation of structural model
        ##############


        if matrix_representation:
            # Initial latex set-up
            from matplotlib import rcParams

            rcParams['text.usetex'] = True
            rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
            # plt.figure(1, figsize=(6, 4))
            ax = plt.axes([0,0,self.p + 1,(self.K +1)/2]) #left,bottom,width,height
            ax.set_xticks([])
            ax.set_yticks([])
            plt.axis('off')
            
            #Delta y_t

            texstr = r'$ '
            texstr += bmatrix(self.A_0)
            texstr += r' \Delta y_{t} = '

            # Cointegrating equation alpha, beta, 

            texstr += bmatrix(self.alpha_struct)
            texstr += bmatrix(self.beta.T)

            if 'ci' in self.deterministic:
                texstr += r' y^{+}_{t-1} '
            else:
                texstr += r' y_{t-1} '

            #Gammas
            if 'co' in self.deterministic:
                col_dim_gamma = self.K +1
            else:
                col_dim_gamma = self.K
            for i in range(self.p -1): # -1 because p-1 is number of gamma's 
                texstr += r' + '
                texstr += bmatrix(self.gamma_struct[:,i * col_dim_gamma: (i +1) * col_dim_gamma])
                texstr += r' \Delta y_{t-'+str(i+1)+r'} '


            # End the string with definitions of variables and print all the latex
            texstr += r' + \epsilon_t '
            texstr += r' \\ \\ '
            texstr += r' \\ y_{t} = \begin{bmatrix} '
            texstr = texstr+ ''.join([name+ r'_t \\' for name in self.names])
            texstr += r'\end{bmatrix}'

            if 'ci' in self.deterministic:
                texstr += r' \\ \\ '
                texstr += r' \\ y^{+}_{t} = \begin{bmatrix} '
                texstr = texstr+ ''.join([name+ r'_t \\' for name in self.names])
                texstr += r'1\\'
                texstr += r'\end{bmatrix}'
            

            texstr += ' $'
            texstr.encode('unicode_escape')
            plt.text(0.01,0.1, texstr, fontsize=30)
            ax.set_title('Matrix representation of VECM',fontsize= 30)
            plt.show()

        return self

    def plot_impulse_response(self, steps_ahead=30, structural = True):

        K=self.K
        p=self.p

        #### Calculate transformation from reduced form to structural residuals
        P =np.linalg.cholesky(self.sigma_u)
        D = np.diag(P.diagonal())


        W = np.zeros((K*p, K*p))
        for i in range(p):
            Ki = K*i # so that we can work in index steps of K instead of 1
            if i ==0:
                W[Ki:K+Ki,Ki:K+Ki] = np.identity(K)
                W[Ki+K: K+Ki+K,Ki:K+Ki] = np.identity(K)
            else:
                W[Ki:K+Ki,Ki:K+Ki] = -1 * np.identity(K)
            if Ki >= K*(p-1): # K(p-1) is the index of the last row of submatrices 
                continue
            else:
                W[Ki+K:K+Ki+K,Ki:K+Ki] = np.identity(K)

        J = np.zeros((K, K*p))
        J[:K,:K] = np.identity(K)

        pi_gamma = np.concatenate([ self.pi[:,:K], self.gamma[:,:K*(p-1)] ],axis=1)

        A = pi_gamma.dot(W) + J
        A_system = np.zeros((K*p,K*p))
        A_system[:K,:] = A
        A_system[K:,:-K] =scipy.linalg.block_diag(*[np.identity(K) for i in range(p-1)])

        Phi_N = []
        for i in range(steps_ahead):
            Phi_N.append(J.dot(np.linalg.matrix_power(A_system,i)).dot(J.T))
        Theta_N = [phi_i.dot(P) for phi_i in Phi_N]

        Phi_N = np.hstack(Phi_N)
        Theta_N = np.hstack(Theta_N)

        fig, axs = plt.subplots(K,K,figsize=((K**0.5)*5,(K**0.5)*5))
        if structural:
            fig.suptitle('Impulse response functions of variables to an orthogonalized 1 standard deviation shock')
            for i in range(K):
                for j in range(K):
                    axs[i,j].plot(Theta_N[i,j::K], c='k')
                    if self.names:
                        axs[i,j].set_title('Impulse of '+ self.names[j] + ' on ' + self.names[i])

        else:
            fig.suptitle('Impulse response functions of variables to a 1 standard deviation shock')
            for i in range(K):
                for j in range(K):
                    axs[i,j].plot(Phi_N[i,j::K], c='k')
                    if self.names:
                        axs[i,j].set_title('Impulse of '+ self.names[j] + ' on ' + self.names[i])

        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()


    def summary(self, alpha =0.05, matrix_representation = True):

        granger_chi_vals = []
        granger_p_vals = []
        for i in range(self.K):
            self, chi_val, p_val = self.granger_causality_test(i)
            granger_chi_vals.append(chi_val)
            granger_p_vals.append(p_val)

        self.granger_chi_vals = granger_chi_vals
        self.granger_p_vals = granger_p_vals


        self, std_err_sigma_u, t_vals_sigma_u, p_vals_sigma_u = self.test_instantaneous_causality()

        self.model = None #place holder that statsmodels summary_params needs
        summary = Summary()

        def make_table(self, params, std_err, t_values, p_values, conf_int,
                        names, title, strip_end=True, alpha=0.05):
            res = (self,
                   params,
                   std_err,
                   t_values,
                   p_values,
                   conf_int
                   )
            param_names = [
                '.'.join(name.split('.')[:-1]) if strip_end else name
                for name in np.array(names).tolist()]
            return summary_params(res, yname=None, xname=param_names,
                                  alpha=alpha, use_t=False, title=title)

        def confint(params, std_err, alpha = 0.05, dist = 'normal', **kwargs):
            if dist == 'normal':
                distribution = sstats.norm(0,1)
            elif dist =='chi2':
                try:    
                    distribution = sstats.chi2(degfree)
                except:
                    raise ValueError('Need to specify the degrees of freedom in paramater degfree')
            else:
                distribution = sstats.norm(0,1)

            upper = (params + std_err * distribution.ppf(1-alpha/2)).flatten(order='F').reshape((-1,1))
            lower = (params - std_err * distribution.ppf(1-alpha/2)).flatten(order='F').reshape((-1,1))
            confints = np.concatenate([lower, upper],axis=1)
            return confints

        
        #Table for lag length testing
        if self.lag_order_tested:
            params = np.concatenate([self.BAH,[self.p]])
            NAs = [0 for i in range(len(params))]
            std_err= NAs
            t_values= NAs
            p_values = NAs
            conf_int= confint(np.array([0,0,0,0]),np.array([0,0,0,0]))
            names = ['BIC','AIC','HQIC','Optimal VAR order']
            title = 'Model information criterion'

            table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
            summary.tables.append(table)

        #Table for lagged coeff (gamma)
        params = self.gamma.T.flatten(order='F')
        NAs = [0 for i in range(len(params))]
        std_err= self.std_err_gamma.T.flatten(order='F')
        t_values= self.t_vals_gamma.T.flatten(order='F')
        p_values = self.p_vals_gamma.T.flatten(order='F')
        conf_int= confint(self.gamma.T, self.std_err_gamma.T)
        names = [self.names[i2]+' L' + str(j+1) +'.'+self.names[i1]  for i2 in range(self.K) for j in range(self.p-1) for i1 in range(self.K)]
        title = 'Lagged parameter coefficients (gamma)'

        table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
        summary.tables.append(table)
        #Table for alpha
        params = self.alpha.T.flatten(order='F')
        NAs = [0 for i in range(len(params))]
        std_err= self.std_err_alpha.T.flatten(order='F')
        t_values= self.t_vals_alpha.T.flatten(order='F')
        p_values = self.p_vals_alpha.T.flatten(order='F')
        conf_int= confint(self.alpha.T, self.std_err_alpha.T)
        names = ['alpha.'+str(i+1)+'.'+str(j+1) for i in range(self.alpha.shape[0]) for j in range(self.alpha.shape[1])]
        title = 'Error correction loading coefficients (alpha)'

        table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
        summary.tables.append(table)

        #Table for Johansen testing
        if self.johansen_tested:
            significance = [90,95,99][self.coint_significance-1]
            params = np.array([self.coint_rank, significance] , ndmin=1)
            NAs = [0 for i in range(len(params))]
            std_err= NAs
            t_values= NAs
            p_values = NAs
            conf_int= confint(np.array([0,0]),np.array([0,0]))
            names = ['Coint Rank',r'Significance %']
            title = "Johansen's cointegration rank test results"

            table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
            summary.tables.append(table)

        #Table for Beta
        params = self.beta.T.flatten(order='F')
        NAs = [0 for i in range(len(params))]
        std_err= self.std_err_beta.T.flatten(order='F')
        t_values= self.t_vals_beta.T.flatten(order='F')
        p_values = self.p_vals_beta.T.flatten(order='F')
        conf_int_beta= confint(self.beta.T, self.std_err_beta.T)
        conf_int_beta[0] = 0
        names = ['beta.'+str(i+1)+'.'+str(j+1) for i in range(self.beta.T.shape[1]) for j in range(self.beta.T.shape[0])]
        title = 'Cointegrating Relationship params (beta)'

        table = make_table(self, params, std_err, t_values, p_values, conf_int_beta, names, title,strip_end =False)
        summary.tables.append(table)

        # Table for granger causality
        if self.granger_tested:
            params = np.array(self.granger_p_vals , ndmin=1)<0.05
            NAs = [0 for i in range(len(params))]
            std_err= NAs
            t_values= np.array(self.granger_chi_vals , ndmin=1)
            p_values = np.array(self.granger_p_vals , ndmin=1)
            conf_int= confint(np.array([0,0]),np.array([0,0]))
            names = [self.names[i]+' Granger Causes?' for i in range(self.K)]
            title = "Granger Causality test results"

            table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
            summary.tables.append(table)


        # Table for instantaneous causality

        if self.instantaneous_causality_tested:
            params = np.array(self.sigma_u[np.tril_indices(self.K)] , ndmin=1)
            NAs = [0 for i in range(len(params))]
            std_err= np.array(self.std_err_sigma_u, ndmin=1)
            t_values= np.array(self.t_vals_sigma_u , ndmin=1)
            p_values = np.array(self.p_vals_sigma_u , ndmin=1)
            conf_int= confint(self.sigma_u[np.tril_indices(self.K)],np.array(self.std_err_sigma_u))
            names = ['Cov(U_'+self.names[i] +', U_' + self.names[j] +')' for j in range(self.K) for i in range(self.K) if i<=j]
            title = "Instantaneous Causality results/error covariance matrix"

            table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
            summary.tables.append(table)

        # Table for 4th order autocorrelation
        autocorr_chi_vals =[]
        autocorr_p_vals =[]
        for i in range(self.K):
            self, chi_val, p_val, _ = self.test_autocorrelation(4,i)
            autocorr_chi_vals.append(chi_val[0])
            autocorr_p_vals.append(p_val[0])

        params = np.array(autocorr_chi_vals + [self.portmanteau_stat[0]])
        NAs = [0 for i in range(len(params))]
        std_err= NAs
        t_values= [0 for i in range(len(params))]
        p_values = np.array(autocorr_p_vals + [self.portmanteau_p_val[0]])
        conf_int= confint(np.array([0,0,0]),np.array([0,0,0]))
        names = ['AutoCorr chi squared '+name for name in self.names]
        names +=['Portmanteau chi squared test']
        title = "4th order autocorrelation diagnostics"

        table = make_table(self, params, std_err, t_values, p_values, conf_int, names, title,strip_end =False)
        summary.tables.append(table)

        ######
        # If wanted, create matrix representation of the VECM
        ######
        if matrix_representation:
            # Initial latex set-up
            from matplotlib import rcParams

            rcParams['text.usetex'] = True
            rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
            # plt.figure(1, figsize=(6, 4))
            ax = plt.axes([0,0,self.p + 1,(self.K +1)/2]) #left,bottom,width,height
            ax.set_xticks([])
            ax.set_yticks([])
            plt.axis('off')
            
            #Delta y_t

            texstr = r'$ \Delta y_{t} = '

            # Cointegrating equation alpha, beta, 

            texstr += bmatrix(self.alpha)
            texstr += bmatrix(self.beta.T)

            if 'ci' in self.deterministic:
                texstr += r' y^{+}_{t-1} '
            else:
                texstr += r' y_{t-1} '

            #Gammas
            if 'co' in self.deterministic:
                col_dim_gamma = self.K +1
            else:
                col_dim_gamma = self.K
            for i in range(self.p -1): # -1 because p-1 is number of gamma's 
                texstr += r' + '
                texstr += bmatrix(self.gamma[:,i * col_dim_gamma: (i +1) * col_dim_gamma])
                texstr += r' \Delta y_{t-'+str(i+1)+r'} '


            # End the string with definitions of variables and print all the latex
            texstr += r' + \epsilon_t '
            texstr += r' \\ \\ '
            texstr += r' \\ y_{t} = \begin{bmatrix} '
            texstr = texstr+ ''.join([name+ r'_t \\' for name in self.names])
            texstr += r'\end{bmatrix}'

            if 'ci' in self.deterministic:
                texstr += r' \\ \\ '
                texstr += r' \\ y^{+}_{t} = \begin{bmatrix} '
                texstr = texstr+ ''.join([name+ r'_t \\' for name in self.names])
                texstr += r'1\\'
                texstr += r'\end{bmatrix}'
            

            texstr += ' $'
            texstr.encode('unicode_escape')
            plt.text(0.01,0.1, texstr, fontsize=30)
            ax.set_title('Matrix representation of VECM',fontsize= 30)
            plt.show()

        return summary



###################################################################################################
###################################################################################################
###################################################################################################
#
#       VAR Model
#
####################################################################################################
####################################################################################################
####################################################################################################

class My_VAR(object):

    def __init__(self, endog, p=1, deterministic='c'):
        self.endog = endog
        self.p = p
        self.K = endog.shape[0]
        self.deterministic = deterministic

    def get_var_matrices(self):

        endog = self.endog
        p = self.p
        K= self.K
        deterministic = self.deterministic

        def create_lag_mask(endog, p):
            # This function enables all the differencing and shifting preprocessing operations to
            # be performed on the data before the mask is applied leading to no missing values
            mat = endog.T # get the orientation so it is time on the rows i.e. T x K matrix
            lagged = [mat]
            for i in range(1,p+1):
                tmp = np.roll(mat,i, axis = 0)
                tmp[:i, :]=np.nan 
                lagged.append(tmp)

            mat = np.concatenate(lagged, axis =1)
            mask = np.isnan(mat).any(axis=1)
            return ~mask

        mask = create_lag_mask(endog, p)

        Y_1_T = endog[:,mask][:,p:]
        self.Y_1_T = Y_1_T

        T = Y_1_T.shape[1]
        self.T = T

        if deterministic =='c':
            Z_stack = [np.ones(Y_1_T.shape[1])]
        elif deterministic =='ct':
            Z_stack = [np.arange(endog.shape[1])[mask][p:]]
        else:
            Z_stack = []
        
        for i in range(p):
            Z_stack.append(np.roll(endog, i, axis=1)[:,mask][:,p-1:-1]) # roll across axis 1 because time is on the columns

        Z= np.row_stack(Z_stack)
        self.Z = Z

        y_bar = Y_1_T.mean(axis=1).reshape((-1,1))
        self.y_bar = y_bar
        Y_0 = Y_1_T - y_bar
        self.Y_0 = Y_0

        X_stack = []
        
        for i in range(p):
            X_stack.append(np.roll(endog, i, axis=1)[:,mask][:,p-1:-1] - y_bar) # roll across axis 1 because time is on the columns

        X = np.row_stack(X_stack)
        self.X =X

        mu = np.concatenate([y_bar.T for i in range(p)],axis=1).T
        self.mu = mu
        mu_star = np.concatenate([y_bar.T for i in range(T)],axis=1).T
        self.mu_star = mu_star.reshape((-1,1))

        return self


    def ML_fit(self):

        self = self.get_var_matrices()
        
        Y_1_T = self.Y_1_T
        Y_0 = self.Y_0
        Z= self.Z
        X = self.X
        p= self.p
        deterministic = self.deterministic
        K = self.K
        mu = self.mu
        mu_star = self.mu_star
        T = self.T

        
        y = Y_1_T.flatten(order ='F').reshape((-1,1))
        alpha =dot( np.kron(dot( inv(dot(X,X.T))  , X), np.identity(K)), y-mu_star)

        self.alpha = alpha

        A = alpha.reshape((K*p, K)).T
        self.A = A

        tmp = Y_0 - dot(A,X)
        sigma_u = (1/T) * dot(tmp, tmp.T)
        self.sigma_u = sigma_u

        det_sigma_u = np.linalg.det(sigma_u)
        
        num_params = p* K**2
        if deterministic != 'nc':
            num_params += 1
        
        self.AIC = np.log(det_sigma_u) + 2 * num_params/T
        self.BIC  = np.log(det_sigma_u) +np.log(T) * num_params/T
        self.HQ = np.log(det_sigma_u) + 2* np.log(np.log(T)) * num_params/T


        return self


    def LS_fit(self):

        self = self.get_var_matrices()
        
        Y_1_T = self.Y_1_T
        Y_0 = self.Y_0
        Z= self.Z
        X = self.X
        p= self.p
        deterministic = self.deterministic
        K = self.K
        mu = self.mu
        mu_star = self.mu_star
        T = self.T

        B = multi_dot([Y_1_T, Z.T, inv(dot(Z,Z.T))])
        self.B = B

        self.U_hat = Y_1_T - dot(B,Z)

        self.sigma_u = (1/T) * dot(self.U_hat, self.U_hat.T)

        # sigma_2 = 
        # BIC = 
        # HQ
        # AIC

        return self




    


