import shap
import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from funcs.DIMV import DIMVImputation
from funcs.miss_forest import mf
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from fancyimpute import SoftImpute
from scipy.stats import spearmanr, kendalltau
# from .missingpy_git.missforest import MissForest 

# THIS IS JUST LIKE THE EXPLAIN.PY NOTEBOOK, EXCEPT IT RUNS FOR NUMPY INPUT
# HAVEN'T ADD GAIN IMPUTATION 

def one_run(X_train, X_train_star, y_train, X_test, X_test_star, y_test, chosen_model):
    
    ori_model = chosen_model
    ori_model.fit(X_train, y_train)
    explainer_ori = shap.Explainer(ori_model, X_test)
    shap_values_ori = explainer_ori(X_test)  
    ypred_ori = ori_model.predict(X_test)

    xm_model = chosen_model #xgboost directly on missing data
    xm_model.fit(X_train_star, y_train)
    explainer_xm = shap.Explainer(xm_model, X_test_star)
    shap_values_xm = explainer_ori(X_test_star)      
    ypred_xm = xm_model.predict(X_test_star)
    
    # impute X using mean imputation 
    X_train_mi = np.where(np.isnan(X_train_star), np.nanmean(X_train_star, axis=0), X_train_star)
    X_test_mi = np.where(np.isnan(X_test_star), np.nanmean(X_train_star, axis=0), X_test_star)
    # X_train_mi = pd.DataFrame(X_train_mi, columns=X_train.columns)
    # X_test_mi = pd.DataFrame(X_test_mi, columns=X_train.columns)
    model_mi = chosen_model
    model_mi.fit(X_train_mi, y_train)
    explainer_mi = shap.Explainer(model_mi, X_test_mi)
    shap_values_mi = explainer_mi(X_test_mi)
    ypred_mi = model_mi.predict(X_test_mi)
    
    # MICE imputation 
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputer.fit(X_train_star)
    X_train_mice = imputer.transform(X_train_star)
    X_test_mice = imputer.transform(X_test_star)
    # X_train_mice = pd.DataFrame(X_train_mice, columns=X_train.columns)
    # X_test_mice = pd.DataFrame(X_test_mice, columns=X_train.columns)
    model_mice = chosen_model
    model_mice.fit(X_train_mice, y_train)
    explainer_mice = shap.Explainer(model_mice, X_test_mice)
    shap_values_mice = explainer_mice(X_test_mice)
    ypred_mice = model_mice.predict(X_test_mice)

    # DIMV imputation 
    imputer = DIMVImputation()
    X_train_star_np, X_test_star_np = np.array(X_train_star), np.array(X_test_star)
    imputer.fit(X_train_star_np, initializing=False)
    X_train_dimv = imputer.transform(X_train_star_np)
    X_test_dimv = imputer.transform(X_test_star_np)
    # X_train_dimv = pd.DataFrame(X_train_dimv, columns=X_train.columns)
    # X_test_dimv = pd.DataFrame(X_test_dimv, columns=X_train.columns)
    model_dimv = chosen_model
    model_dimv.fit(X_train_dimv, y_train)
    explainer_dimv = shap.Explainer(model_dimv, X_test_dimv)
    shap_values_dimv = explainer_dimv(X_test_dimv)
    ypred_dimv = model_dimv.predict(X_test_dimv)

    # MissForest
    # mf = MissForest().fit_transform
    X_train_mf = mf(np.array(X_train_star))
    X_test_mf = mf(np.vstack((X_train_star, X_test_star)))[-len(X_test_star):]
    # X_train_mf = pd.DataFrame(X_train_mf, columns=X_train.columns)
    # X_test_mf = pd.DataFrame(X_test_mf, columns=X_train.columns)
    model_mf = chosen_model
    model_mf.fit(X_train_mf, y_train)
    explainer_mf = shap.Explainer(model_mf, X_test_mf)
    shap_values_mf = explainer_mf(X_test_mf)
    ypred_mf = model_mf.predict(X_test_mf)

    # SoftImpute
    X_train_soft = SoftImpute(verbose = False).fit_transform(X_train_star)
    X_test_soft = SoftImpute(verbose = False).fit_transform(np.vstack((X_train_star, X_test_star)))[-len(X_test_star):]
    # X_train_soft = pd.DataFrame(X_train_soft, columns=X_train.columns)
    # X_test_soft = pd.DataFrame(X_test_soft, columns=X_train.columns)
    model_soft = chosen_model
    model_soft.fit(X_train_soft, y_train)
    explainer_soft = shap.Explainer(model_soft, X_test_soft)
    shap_values_soft = explainer_soft(X_test_soft)
    ypred_soft = model_soft.predict(X_test_soft)
    
# def mse_imputation(X_test_imputed):
#     return np.mean((np.array(X_test_imputed)-np.array(X_test))**2)
# def mse_shap(computed_shap_values):
#     return np.mean((computed_shap_values - shap_values_ori.values)**2)

    mse_imputation = lambda X_test_imputed: np.mean((np.array(X_test_imputed)-np.array(X_test))**2)
    mse_imputation_all = np.array([mse_imputation(X_test_mi), mse_imputation(X_test_mice),
                        mse_imputation(X_test_dimv), mse_imputation(X_test_mf),
                        mse_imputation(X_test_soft)])

    mse_shap = lambda computed_shap_values: np.mean((computed_shap_values - shap_values_ori.values)**2)
    mse_shap_all = np.array([mse_shap(shap_values_xm.values),mse_shap(shap_values_mi.values), mse_shap(shap_values_mice.values),
                        mse_shap(shap_values_dimv.values), mse_shap(shap_values_mf.values), mse_shap(shap_values_soft.values)])

    mse_ypred = lambda ypred_method: np.mean((ypred_ori-ypred_method)**2)
    mse_ypred_all = np.array([mse_ypred(ypred_xm), mse_ypred(ypred_mi), mse_ypred(ypred_mice),
                              mse_ypred(ypred_dimv), mse_ypred(ypred_mf), mse_ypred(ypred_soft)])

    # mse_ypred_ytest = lambda ypred_method: np.mean((y_test-ypred_method)**2)
    # mse_ypred_ytest_all = np.array([mse_ypred_ytest(ypred_ori), mse_ypred_ytest(ypred_xm), mse_ypred_ytest(ypred_mi), mse_ypred_ytest(ypred_mice),
    #                           mse_ypred_ytest(ypred_dimv), mse_ypred_ytest(ypred_mf), mse_ypred_ytest(ypred_soft)])   

    # get the ranking correlation for spearman rank correlation between the predicted y on test set of original data and y predicted on imputed data
    cor_ypred = lambda ypred_method: spearmanr(ypred_ori, ypred_method)
    cor_ypred_all = np.array([cor_ypred(ypred_xm), cor_ypred(ypred_mi), cor_ypred(ypred_mice),
                              cor_ypred(ypred_dimv), cor_ypred(ypred_mf), cor_ypred(ypred_soft)])
    
    # get the ranking correlation for each feature 
    get_spearmanr = lambda shap_vals_method: np.array([spearmanr(shap_values_ori.values[:,i], shap_vals_method.values[:,i])[0] 
                                                       for i in range(shap_values_ori.values.shape[1])])
    spearman_res = np.array([get_spearmanr(shap_values_xm), get_spearmanr(shap_values_mi), get_spearmanr(shap_values_mice),
                             get_spearmanr(shap_values_dimv), get_spearmanr(shap_values_mf),
                             get_spearmanr(shap_values_soft)])
     
    
    
    #get the ranking correlation for each feature 
    shap_all = [shap_values_ori, shap_values_xm, shap_values_mi, shap_values_mice, shap_values_dimv, shap_values_mf, shap_values_soft]
    other_measures = [mse_imputation_all, mse_shap_all, mse_ypred_all, cor_ypred_all, spearman_res]

    return  shap_all, other_measures

def get_average_shap_vals(results, j, nruns):
    # get the average shap values from all runs for each imputation method or the original 
    current = results[0][0][j]
    for i in range(1, nruns):
        current.values += results[i][0][j].values
        current.base_values += results[i][0][j].base_values
        current.data += results[i][0][j].data  
    current.values = current.values/nruns
    current.base_values = current.base_values/nruns
    current.data = current.data/nruns
    return current

