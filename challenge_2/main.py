
"""
Automation: Create a script that automates the entire process, including data preprocessing, 
model training, evaluation, and saving. This script should be able to handle new data and retrain the model as needed.
"""
import pickle
import preprocessing
from config import Config_Path
from params import Params
from datetime import datetime
 
def create_experiment_dirs(main_dir):
    main_dir.mkdir(parents=True, exist_ok=True)

def train_model(new_data):
    # Preprocess the new data
    X_train, X_val, y_train, y_val = preprocess.preprocess_data(new_data)

    # Train the model
    model, scaler, preprocessor = train.train_regression_model(X_train, y_train)

    # Evaluate the model
    train.evaluate_model(model, X_val, y_val)

    # Save the trained model and its components
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)



if __name__ == '__main__':

    paths = Config_Path()
    params = Params(paths.params_path)
    
    now = datetime.now()
    date_and_time_str = now.strftime("%Y%m%d_%H%M%S")
    #exp_version = '1_ep' + str(params.num_epochs) + 'bs' + str(params.batch_size)  
    exp_id = date_and_time_str 
    params.exp_id = exp_id
    #params.exp_version = exp_version
    params.log_exp_path = paths.main_dir / 'logs_experiments' / str(params.exp_id)
    params.json_path = params.log_exp_path/ str(params.model_version + '_params.json')      
    params.last_dataset_update = date_and_time_str
    
    if not params.log_exp_path.exists():
        create_experiment_dirs(params.log_exp_path)
    

    #new_data = load_new_data()
    #train_model(new_data)
    print(params)
    params.save(params.json_path)