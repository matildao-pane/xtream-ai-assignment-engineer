import pickle
import preprocessing
from path_config import Path_Config
from params import Params
from datetime import datetime
from data_loader import DataLoader
from preprocessing import DatasetProcessor
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def create_experiment_dirs(main_dir):
    main_dir.mkdir(parents=True, exist_ok=True)

def plot_features(dir_path,X_train, X_test, df):
    X = df.drop(columns=['price'])
    X_train = pd.DataFrame(X_train, columns =['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x' ] )
    X_test = pd.DataFrame(X_test, columns =['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x' ])
    merged_df = pd.concat([X_train, X_test], ignore_index=True)

    #Plot original vs standardized features
    plt.figure(figsize=(14, 6))
    for i, feature in enumerate(['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x' ]):
        plt.subplot(2, 4, i+1)
        sns.histplot(X[feature], color='blue', label='Original')
        sns.histplot(merged_df[feature], color='orange', label='Standardized')
        plt.title(f'{feature} - Original vs Standardized')
        plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path+'/features.png')

def plot(y_test, y_pred, dir_path):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Plot 1: Correct Values vs Predicted Values
    ax1 = axes[0]
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1, color='blue', alpha=0.5)
    sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], ax=ax1, color='red', linestyle='--')
    plt.title(f' Predicted vs Ground Truth Values')
    ax1.set_xlabel('Correct Values')
    ax1.set_ylabel('Predicted Values')

    # Plot 2: Residuals vs Fitted Values
    ax2 = axes[1]
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals, ax=ax2, color='blue', alpha=0.5)
    ax2.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residuals vs Ground Truth Values')
    ax2.set_xlabel("Fitted Values")
    ax2.set_ylabel("Residuals")
    
    plt.tight_layout()
    fig.savefig(dir_path+'/training_plot.png')
    
def eval(X_test, y_test, dir): 
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print('\nResults:')
    print(f'\nRMSE: {rmse}, \nR2 Score: {r2}, \nMAE: {mae}' )
    plot(y_test, y_pred, dir)
 
 
if __name__ == '__main__':
    
    # SETUP
    paths = Path_Config()
    params = Params(paths.params_path)  
    
    #Before any new training a new folder is created to store model and parameters 
    now = datetime.now()
    date_and_time_str = now.strftime("%Y%m%d_%H%M%S")
    exp_id = date_and_time_str 
    params.exp_id = exp_id
    params.log_exp_path = paths.main_dir / 'logs_experiments' / str(params.exp_id)
    params.json_path = params.log_exp_path/ str(params.model_version + '_params.json')      
    
    if not params.log_exp_path.exists():
        create_experiment_dirs(params.log_exp_path)
    
    params.save(params.json_path)
    
    # DATA LOADING
    data_loader = DataLoader(paths.data_dir, params.last_dataset_update)  
    last_update = data_loader.load_new_data()
    
    # DATA PREPROCESSING
    ds = DatasetProcessor(paths,  params.last_dataset_update,  params.train_temp_split_percentage, params.rand_state)
    X_train, X_test, y_train, y_test, columns, df = ds.preprocess()
    print(X_train, X_test, y_train, y_test, columns)
    
    print(df)
    params.dataset_size = len(df)
    params.training_size = len(X_train)
    params.test_size = len(X_test)
    params.last_dataset_update =  date_and_time_str
    params.save(paths.params_path)
 
    plot_features(params.log_exp_path, X_train, X_test, df)  #TODO 
    
    # TRAINING  
    model = XGBRegressor(seed=params.rand_state, objective=params.objective, colsample_bytree= params.colsample_bytree, 
                     learning_rate=params.learning_rate, max_depth= params.max_depth, n_estimators=params.n_estimators, subsample=params.subsample)
    model.fit(X_train, y_train)

    # Save the model 
    with open( params.log_exp_path +(f'/{params.model_version}_.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # EVALUATION
    eval(X_test, y_test, params.log_exp_path)