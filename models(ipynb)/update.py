import np as numpy
import pandas as pd
import statsmodels.api as sm


def rsi(close_delta, periods=20, ema=True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = close_delta.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi


def wide_format(df):
    df_= df.reset_index(level=['stock']).sort_index()
    df_ = df_.pivot(columns ='stock')
    df_.columns = df_.columns.get_level_values(0) + '_' +  [str(x) for x in df_.columns.get_level_values(1)]

    return df_


def get_feature_train(log_pr, volu, x_begin_idx, x_end_idx, y_begin_idx, 
                        grp_idx=None, rm_outlier=False, print_cor=True):
    """
    Input:
    log_pr (pdSeries): train set
    volu (pdSeries): train set
    x_begin_idx (pdIndex): to truncate the NaNs
    grp_idx (dict): key is group idx, value is list of stock idx

    Returns:
    feature_dict (dict): key is group idx, value is a tuple of feature matrix and response
    """

    log_pr_df = log_pr.reset_index().melt(id_vars=['timestamp'])
    log_pr_df.columns = ['timestamp', 'stock', 'log_pr']
    log_pr_df = log_pr_df.set_index(['timestamp', 'stock']).sort_index()

    volu_df = volu.reset_index().melt(id_vars=['timestamp'])
    volu_df.columns = ['timestamp', 'stock', 'volu']
    volu_df = volu_df.set_index(['timestamp', 'stock']).sort_index()

    features = pd.DataFrame(index=log_pr_df.index)
    # features['trend'] = np.ones(log_pr_df.shape[0])

    # log_pr feature
    for i in [30]:
        features['log_pr_{}'.format(i)] = -log_pr_df.groupby(level='stock').log_pr.diff(i)

    k_period = 40
    d_period = 3
    ma_max = lambda x: x.rolling(k_period).max()
    ma_min = lambda x: x.rolling(k_period).min()
    mad = lambda x: x.rolling(d_period).mean()
    # msd = lambda x: x.rolling(d_period).sum()

    features['pr_min_40'] = log_pr_df.groupby(level='stock').log_pr.apply(ma_min)
    features['pr_max_40'] = log_pr_df.groupby(level='stock').log_pr.apply(ma_max)

    features['pr_so_40'] = (log_pr_df.log_pr - features['pr_min_40'])*100 / (features['pr_max_40'] - features['pr_min_40'])
    features['pr_so_40d3'] = features.groupby(level='stock').pr_so_40.apply(mad)

    # STD of log price
    for i in [10]:
        std = lambda x: x.rolling(i).std()
        features['log_pr_std_{}'.format(i)] = log_pr_df.groupby(level='stock').log_pr.apply(std)

    # RSI
    # features['rsi_20'] = log_pr_df.groupby(level='stock').log_pr.apply(rsi)
    features['rsi_30'] = log_pr_df.groupby(level='stock').log_pr.apply(rsi, periods=30)
    # features['rsi_50'] = log_pr_df.groupby(level='stock').log_pr.apply(rsi, periods=50)

    # volume feature
    log_fn = lambda x: np.log(x+1)
    features['log_volu'] = volu_df.groupby(level='stock').volu.apply(log_fn)

    # stdised volume in 2 hours backward rolling windows
    zscore_fn = lambda x: (x - x.rolling(window=240, min_periods=20).mean()) / x.rolling(window=240, min_periods=20).std()
    features['volu_z_score'] = volu_df.groupby(level='stock').volu.apply(zscore_fn)


    # drop min, max features
    features = features.drop(columns=['pr_min_40', 'pr_max_40', 'pr_so_40'])

    response = log_pr.diff(30)

    if grp_idx is not None:
        feature_dict = {}
        for key, idx_lis in grp_idx.items():
            feature_df_dropped = wide_format(features.loc[pd.IndexSlice[:,idx_lis],:])
            # transform back to wide format
            feature_dict[key] = (feature_df_dropped.iloc[x_begin_idx:x_end_idx], 
                                            response[idx_lis].iloc[y_begin_idx:])
        return feature_dict
    else:
        # transform back to wide format
        feature_df_dropped = wide_format(features).iloc[x_begin_idx:x_end_idx]
        # feature_df_dropped = feature_df[x_begin_idx:x_end_idx]
    
        if print_cor:
            for i in range(10):
                
                feature_train_0 = features.xs(i, level='stock').iloc[x_begin_idx:x_end_idx]
                print(feature_train_0.corrwith(response[i]))
                print(feature_train_0.isnull().sum())

        return feature_df_dropped, response.iloc[y_begin_idx:]




def rsi_test(log_pr, periods=20):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = log_pr.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # Use exponential moving average
    ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean().iloc[-1]
    ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean().iloc[-1]
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def wide_format_test(df):
    df_= df.reset_index()
    df_ = df_.pivot(columns ='index').apply(lambda s: s.dropna().reset_index(drop=True))
    df_.columns = df_.columns.get_level_values(0) + '_' +  [str(x) for x in df_.columns.get_level_values(1)]

    return df_

def get_feature_test(log_pr, volu, grp_idx=None):
    """
    Input: 
    log_pr (pdSeries): 1 day of log pr 
    volu (pdSeries): 1 day of volume

    Output:
    test data frame
    """
    features = pd.DataFrame(index=log_pr.columns)
    # print(log_pr.index[-1])
    # features['trend'] = np.ones(log_pr.shape[1])
    # backward return
    # print(-(log_pr.iloc[-1] - log_pr.iloc[-30]).values)
    # for i in [30]:
    features['log_pr_30'] = -(log_pr.iloc[-1] - log_pr.iloc[-31]).values
    
    # Oscilator
    k_period = 40
    d_period = 3
    pr_min_40 = log_pr.rolling(k_period).min().iloc[-d_period:].values
    pr_max_40 = log_pr.rolling(k_period).max().iloc[-d_period:].values
    pr_so_40 = (log_pr.iloc[-d_period:].values - pr_min_40)*100 / (pr_max_40 - pr_min_40)
    features['pr_so_40d3']  = pr_so_40.mean(0)

    # backward rolling std
    # features['log_pr_std_10'] = log_pr.iloc[-10:].std(0).values
    features['log_pr_std_30'] = log_pr.iloc[-30:].std(0).values
    
    # RSI
    features['rsi_30'] = log_pr.apply(rsi_test, periods=30)

    # volume features
    features['log_volu'] = np.log(volu.iloc[-1].values + 1)
    features['volu_z_score'] = ((volu.iloc[-1] - volu.iloc[-240:].mean())/volu.iloc[-240:].std()).values

    # print(volu.iloc[-240:].mean())

    if grp_idx is None:
        return wide_format_test(features)
    else:
        df_dict = {}
        for key, idx_lis in grp_idx.items():
            df_dict[key] = wide_format_test(features.loc[idx_lis])
        return df_dict


def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    # grp_idx = {0:[1,5,6,8], 1:[0,2,3,4,7,9]}
    grp_idx = {i:[i] for i in range(10)}
    x = get_feature_test(A, B, grp_idx=grp_idx)
    pred_dict = {i: model.predict(np.insert(x[i].values,0,1.)) for i, model in model_dict.items()}
    
    out = np.zeros(10)
    for keys, idx in grp_idx.items():
        out[idx] = pred_dict.get(keys)

    return out

