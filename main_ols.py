import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


models = []
with open("models.pckl", "rb") as f:
    while True:
        try:
            models.append(pickle.load(f))
        except EOFError:
            break
model_dict = {i: models[i] for i in range(len(models))}
features_dict = {0:[],1:[],2:[],3:[],}

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

    # backward return
    # print(-(log_pr.iloc[-1] - log_pr.iloc[-30]).values)
    for i in [10, 20, 30]:
        features['log_pr_{}'.format(i)] = -(log_pr.iloc[-1] - log_pr.iloc[-i]).values
    # backward rolling std
    features['log_pr_std_10'] = log_pr.iloc[-10:].std(0).values
    
    # volume features
    features['log_volu'] = np.log(volu.iloc[-1].values + 1)
    features['volu_z_score'] = ((volu.iloc[-1] - volu.iloc[-240:].mean())/volu.iloc[-240:].std()).values

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
    grp_idx = {i:[i] for i in range(10)}
    x = get_feature_test(A, B, grp_idx=grp_idx)
    pred_dict = {i: model.predict(x[i]) for i, model in model_dict.items()}
    
    out = np.zeros(10)
    for keys, idx in grp_idx.items():
        out[idx] = pred_dict.get(keys)

    return out