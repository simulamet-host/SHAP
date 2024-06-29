import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer


def loss_function_categorical(Xnew, Xold, Xmissing):#-------------------CHECK------------
    return np.sum(Xnew != Xold)/np.sum(np.isnan(Xmissing))
    
def loss_function_continuous(Xnew, Xold):
    return np.sum((Xnew - Xold)**2)/np.sum(Xnew**2)

def loss_function(Xnew, Xold, X, cat_indices):
    if cat_indices == None:
        return loss_function_continuous(Xnew, Xold)
    else:
        if cate_indices == "all":
            return loss_function_categorical(Xnew, Xold, Xmissing)
        else:
            loss_cat = loss_function_categorical(Xnew[:,cat_indices], Xold[:,cat_indices], Xmissing[:,cat_indices])
            numerical_columns = np.setdiff1d(np.arange(X.shape[1]), cat_indices)
            loss_continuous = loss_function_continuous(Xnew[:,numerical_columns], Xold[:,numerical_columns])
            return (loss_cat + loss_continuous)/2
        
def mf(X,init_imputer =  SimpleImputer(), cat_indices = None,
               loss_function = loss_function, max_iter=10, gamma = 0.01, 
               regressor = RandomForestRegressor(n_estimators= 100), 
               classifier = RandomForestClassifier(n_estimators= 100)):
    """
    Impute missing values using the missForest algorithm for both numerical and categorical features.
   
    Categorical features must be stored in object
    
    Parameters:
    - X: Input data with missing values (numpy array or pandas DataFrame).
    - cat_indices: indices of categorical features     
    - max_iter: Maximum number of iterations.
    - gamma: threshold of the loss function to stop the loops
    Returns:
    - Imputed data.
    """ 
    # Calculate the number of missing values in each column
    missing_values_count = np.sum(np.isnan(X), axis=0)
    # Sort the column indices based on the number of missing values
    sorted_column_indices = np.argsort(missing_values_count)
    # Reorder the columns in X based on the sorted indices
    X_sorted = X[:, sorted_column_indices]
    
    # Identify numerical and categorical columns
#     numerical_columns = np.setdiff1d(np.arange(X.shape[1]), cat_indices)
#     categorical_columns = cat_indices

    # Initialization
    X_old = init_imputer.fit_transform(X_sorted) 
    X_new = X_old.copy()
    loss = np.inf
    
    # Iteratively impute missing values
    for iter_count in range(max_iter):

        # iterates through all the features 
        for i in np.arange(X.shape[1]):
            missing_row_indices = np.isnan(X_sorted[:, i])
            if np.sum(missing_row_indices) > 0:
                X_now = np.delete(X_old, i, axis=1)
                X_train = X_now[~missing_row_indices]
                X_test = X_now[missing_row_indices]                
                y_train = X_sorted[~missing_row_indices, i]
                
                if cat_indices == None: fit_model = regressor
                else: 
                    if i in cat_indices: fit_model = classifier 
                    else: fit_model = regressor
                    
                fit_model.fit(X_train, y_train)
                missing_values_predict = fit_model.predict(X_test)
                X_new[missing_row_indices, i] = missing_values_predict
        
        # Compute the loss
        loss = loss_function(X_new, X_old, X, cat_indices)
        print('loss',loss_function(X_new, X_old, X, cat_indices))    
        if loss < gamma: break
        X_old = X_new.copy()
    
    # Inverse sort the columns to get back the original order
    imputed_X = X_new[:, np.argsort(sorted_column_indices)]
    print('number of runs used by missForest:', iter_count+1)
    print('loss:', loss)
    return imputed_X