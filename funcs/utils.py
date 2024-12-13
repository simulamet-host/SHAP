import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

def generate_missing_data(df, rate):
    # input: DataFrame
    # Create a binary mask with the same shape as df, with True values 
    # at locations we want to make missing, and False values elsewhere
    mask = np.random.rand(*df.shape) < rate
    
    # Create a copy of the DataFrame so we don't modify the original data
    df_missing = df.copy()
    
    # Apply the mask to the DataFrame, replacing True values with NaN
    df_missing[mask] = np.nan
    
    return df_missing


def shap_ranking_table(X_train, shap_values_ori, shap_values_xm, shap_values_mi, shap_values_mice,
                      shap_values_dimv, shap_values_mf, shap_values_soft, shap_values_gain):
    feature_names = X_train.columns

    vals = np.abs(shap_values_ori.values).mean(0)
    feature_importance_ori = pd.DataFrame(list(zip(feature_names, vals)),columns=['Ranking','FIV'])
    feature_importance_ori.sort_values(by=['FIV'],ascending=False, inplace=True)

    vals = np.abs(shap_values_xm.values).mean(0)
    feature_importance_xm = pd.DataFrame(list(zip(feature_names, vals)),columns=['Ranking','FIV'])
    feature_importance_xm.sort_values(by=['FIV'],ascending=False, inplace=True)    

    vals = np.abs(shap_values_mi.values).mean(0)
    feature_importance_mi = pd.DataFrame(list(zip(feature_names, vals)),columns=['Ranking','FIV'])
    feature_importance_mi.sort_values(by=['FIV'], ascending=False, inplace=True)

    vals = np.abs(shap_values_mice.values).mean(0)
    feature_importance_mice = pd.DataFrame(list(zip(feature_names, vals)),columns=['Ranking','FIV'])
    feature_importance_mice.sort_values(by=['FIV'],ascending=False, inplace=True)

    vals = np.abs(shap_values_dimv.values).mean(0)
    feature_importance_dimv = pd.DataFrame(list(zip(feature_names, vals)),columns=['Ranking','FIV'])
    feature_importance_dimv.sort_values(by=['FIV'], ascending=False, inplace=True)

    vals = np.abs(shap_values_mf.values).mean(0)
    feature_importance_mf = pd.DataFrame(list(zip(feature_names, vals)),columns=['Ranking','FIV'])
    feature_importance_mf.sort_values(by=['FIV'], ascending=False, inplace=True)

    vals = np.abs(shap_values_soft.values).mean(0)
    feature_importance_soft = pd.DataFrame(list(zip(feature_names, vals)),columns=['Ranking','FIV'])
    feature_importance_soft.sort_values(by=['FIV'],ascending=False, inplace=True)   

    vals = np.abs(shap_values_gain.values).mean(0)
    feature_importance_gain = pd.DataFrame(list(zip(feature_names, vals)),columns=['Ranking','FIV'])
    feature_importance_gain.sort_values(by=['FIV'],ascending=False, inplace=True)       
    
    combined_feature_importance = np.hstack((np.array(feature_importance_ori),np.array(feature_importance_xm),
                                             np.array(feature_importance_mi),np.array(feature_importance_mice),
                                             np.array(feature_importance_dimv), np.array(feature_importance_mf),
                                             np.array(feature_importance_soft),np.array(feature_importance_gain)))
    column_names = [('Original','Ranking'),('Original','FIV'),
                    ('xm','Ranking'),('xm','FIV'),
                    ('Mean Imputation','Ranking'),('Mean Imputation','FIV'),
                    ('MICE','Ranking'),('MICE','FIV'),
                    ('DIMV','Ranking'),('DIMV','FIV'),
                    ('missForest','Ranking'),('missForest','FIV'),
                   ('SOFT-IMPUTE','Ranking'),('SOFT-IMPUTE','FIV'),
                   ('GAIN','Ranking'),('GAIN','FIV')]
    combined_feature_importance = pd.DataFrame(combined_feature_importance)
    combined_feature_importance.columns =pd.MultiIndex.from_tuples(column_names)

    combined_feature_importance = pd.concat([feature_importance_ori, feature_importance_xm, feature_importance_mi,
              feature_importance_mice, feature_importance_dimv,
              feature_importance_mf, feature_importance_soft, feature_importance_gain], axis = 1)
    print('combined_feature_importance')
    combined_feature_importance = combined_feature_importance.drop(combined_feature_importance.columns[[2,4,6,8,10]], axis = 1) 
    combined_feature_importance.columns = ['Original','Xgb on missing data','Mean Imputation', 'MICE','DIMV','missForest','SOFT-IMPUTE', 'GAIN']
    combined_feature_importance.index = feature_importance_ori['Ranking']
    print(combined_feature_importance.to_latex(index=True,formatters={"name": str.upper},float_format="{:.3f}".format))
    
    # # get the ranking correlation 
    # get_spearmanr = lambda feature_importance_method: spearmanr(feature_importance_ori['FIV'], feature_importance_method['FIV'])
    # spearman_res = np.array([get_spearmanr(feature_importance_xm), get_spearmanr(feature_importance_mi), get_spearmanr(feature_importance_mice),
    #                          get_spearmanr(feature_importance_dimv), get_spearmanr(feature_importance_mf),
    #                          get_spearmanr(feature_importance_soft)])    
    # spearman_res = pd.DataFrame(spearman_res, columns = ['statistic','pvalue'], index = ['XgbM','Mean imputation','MICE','DIMV','MissForest','SOFT-IMPUTE'])
    # print('ranking correlation between feature importance')
    # print(spearman_res.to_latex(index=True,formatters={"name": str.upper},float_format="{:.1f}".format))

          
    return combined_feature_importance#, spearman_res

