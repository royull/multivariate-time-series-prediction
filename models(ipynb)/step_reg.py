import pandas as pd
import statsmodels.api as sm

def forward_regression(X, y):
    '''
    Input
    X,y: training matrix (without intercept)
    Return
    model: ols model fitted with intercept on the selected features
    feature_selected: selected features
    '''
    initial_list = []
    included = list(initial_list)
    feature_num = len(X.columns)
    best_aics = pd.Series(index={i for i in range(feature_num)})
    best_features = list(it.repeat([],feature_num))
    for k in range(feature_num):
        excluded = list(set(X.columns)-set(included))
        new_aic = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_aic[new_column] = model.aic
        best_aic = new_aic.min()
        best_aics[k] = best_aic
        best_feature = new_aic.idxmin()
        included.append(best_feature)
        best_features[k] = included.copy()
    feature_selected = best_features[best_aics.idxmin()]
    model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[feature_selected]))).fit()
    return model,feature_selected
