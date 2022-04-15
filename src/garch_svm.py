import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse

from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
import pickle
from copy import copy

# preprocessing
def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean(0)
    iqr = dta.quantile([0.25, 0.75], axis=0).diff().T.iloc[:, 1]
    
    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan

    return treated

def preprocess(X, scale=100):
    return remove_outliers(X.diff()).dropna() * scale

def svr_input(dta, smooth=5):
    """
    dta (df): data frame that contains preprocessed log price
    dt (int): difference in X and y 
    returns list of feature array and response array
    """
    # compute volatility
    pr_vol = dta.rolling(smooth).std()
    # compute square of dta_pr
    returns_svm = dta ** 2
    # form list of feature ndarray
    X_feature = [pd.concat(
                    [pr_vol.iloc[(smooth-1):,i], returns_svm.iloc[(smooth-1):,i]], 
                        axis=1, ignore_index=True).values
                            for i in range(10)]
    y_response = [pr_vol.iloc[(smooth-1):, i].values.squeeze() for i in range(10)]

    return X_feature, y_response


def GARCH_SVR():
    def __init__(self, kernel, dim, dt=1, scale=100, smooth=5):
        # SVR model
        self.kernel = SVR(kernel=kernel)
        self.para_grid = {'gamma': sp_rand(),
                            'C': sp_rand(),
                            'epsilon': sp_rand()}
        # data specs
        self.dim = dim
        self.dt = dt # time difference in y and X for regression 
        self.scale = scale # scaling in preprocecing of the data
        self.smooth = smooth # number of time points to compute volatility

        # output
        self.model = [[] for _ in range(self.dim)] 
        self.pred = [[] for _ in range(self.dim)]
    

    def train(self, X_pr, X_vol=None):
        """
        X_pr, X_volu (series data frame): #col is dim - statioanry time series
        
        Generates
        # X is a list of feature matrix for SVR
        # y is the response (volatility)
        """
        # assert len(X) == self.dim
        # assert len(y) == self.dim
        
        dta_pr = preprocess(X_pr, self.scale)
        X, y = svr_input(dta_pr, self.smooth)

        clf = RandomizedSearchCV(self.svr, self.para_grid)
        for i, Xi, yi in enumerate(zip(X, y)):
            clf = clf.fit(Xi[:-self.dt,:], yi[self.dt:])
            pickle.dump(clf, open('model/svr_{}.sav'.format(i), 'wb'))


    @staticmethod
    def _forward1(mod, Xi):
        """
        Forward predict 1 unit of volatility, given data Xi
        
        Updates Xi with new sqaured return and volatility 
        Return new process realisation a_t
        """
        st = mod.predict(Xi)    
        at = np.random.normal(0,1,1) * st[-1]
        x = [st[-1], at**2]
        Xi_new = np.concatenate((Xi, [x]), axis=0)
        return at, Xi_new

    # the get_r_hat function
    def forward(self, X_pr, X_volu=None, step=30):
        """
        Forward predict 'step' units of volatility
        X_pr, X_volu (series data frame): #col is dim - statioanry time series
        
        creats feature matrix inside
        updates the forward prediction of GARCH process
        returns the cumulative returns in 'step' minutes
        """
        # assert len(X_pr) == self.dim
        n_col = X_pr.shape[1]
        
        dta_pr = preprocess(X_pr, self.scale)
        X, _ = svr_input(dta_pr, self.dt, self.smooth)

        # TODO:
        # Figure out a multivariate procedure rather than independant ones

        for i in range(n_col):
            Xi = X[i]
            self.model[i] = pickle.load(open('model/svr_{}.sav'.format(i), 'rb'))
            # depending on the dt, different number of steps is required
            # e.g. if dt = 30, then forward predict 1 is enough
            for _ in range(step//self.dt):
                at, Xi_new = self._forward1(self.model[i], Xi)
                # update prediction list and feature list
                self.pred[i].append(at)
                Xi = Xi_new

        # return the cumulative a_t (in 30 steps)
        return np.cumsum(np.array(self.pred), axis=1)[:,-1].squeeze()


    def test(self, X_pr, X_volu=None, step=30):
        """
        X_pr, X_volu (series data frame): test data indexed with time 
        provided with 1 day of data, forward predict 'steps' mins log return
        """
        t0 = time.time()
        dt = datetime.timedelta(days=1)
        r_hat = pd.DataFrame(index=X_pr.index[30::10], columns=np.arange(10), dtype=np.float64)

        for t in X_pr.index[30::10]: # compute the predictions every 10 minutes
            r_hat.loc[t, :] = self.forward(X_pr.loc[(t - dt):t], X_volu.loc[(t - dt):t])

        t_used = time.time() - t0
        print(t_used)

        
    def rmse(self, y):
        """
        evaluate prediction with true process
        """
        return [np.sqrt(mse(y[i], self.pred[i])) for i in range(self.dim)]


    def _plot(self, y):
        df_y = pd.DataFrame(y.T).rename(columns={f"diff_log_pr_{i}": i for i in range(self.dim)})
        df_y_pred = pd.DataFrame(
                    np.array(self.pred).T).rename(columns={f"diff_log_pr_{i}": i for i in range(self.dim)})


        plt.figure(figsize=(16, 6))
        plt.plot(df_y, label='Realized Volatility')
        plt.plot(df_y_pred, label='Volatility Prediction-SVR-GARCH')
        plt.title('Volatility Prediction with SVR-GARCH (Linear)', fontsize=12)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    log_pr = pd.read_pickle("./data/log_price.df")
    volu = pd.read_pickle("./data/volume_usd.df")

    # split train and test
    t_split = 1440 * 30 # last month as test data
    log_pr_train = log_pr.iloc[:-t_split]
    log_pr_test = log_pr.iloc[-t_split:]

    # fit GARCH-SVR
    kernel = 'rbf'
    dt = 1
    dim = 10

    mod = GARCH_SVR(kernel, dt, dim)
    mod.train(log_pr_train)