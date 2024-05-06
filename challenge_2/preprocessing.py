import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from path_config import Path_Config
from params import Params


columns =  ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']
columns_to_remove = ['y','z']
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = reversed([chr(i) for i in range(ord('D'), ord('Z')+1)])
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1','IF']


class DatasetProcessor:
    def __init__(self,paths, last_update, train_test_split_percentage, random_state):  
        self.paths = paths
        self.last_update = last_update
        self.test_size=train_test_split_percentage
        self.random_state=random_state
 
    def process_single_batch(self,df):
        """operations that can be performed in separate batches of data:
            remove 0 values
            remove na values
            remove negative prices
            remove duplicates 
            remove columns y e z
            transform price and carat columns to log
            label encoding
        """
        # Remove Na, =0, <0, duplicates, correlated columns
        df = df[df[columns] != 0]
        df = df.dropna(subset=columns)
        df = df[df['price'] >= 0]
        df = df.drop_duplicates()
        df = df.drop(columns=columns_to_remove)
        
        # Log trasformation
        df['price'] = np.log1p(df['price'])
        df['carat'] = np.log1p(df['carat'])
        
        # Label encoding
        df['cut'] = pd.Categorical(df['cut'], categories=cut_order, ordered=True)
        df['color'] = pd.Categorical(df['color'], categories=color_order, ordered=True)
        df['clarity'] = pd.Categorical(df['clarity'], categories=clarity_order, ordered=True)
        df['cut'] = df['cut'].cat.codes
        df['color'] = df['color'].cat.codes
        df['clarity'] = df['clarity'].cat.codes

        return df
 
    def load_clean_data(self):
        clean_data_dir = self.paths.data_dir / 'clean_data'
        df_list = [pd.read_csv(f) for f in clean_data_dir.glob('*.csv')]
        return pd.concat(df_list)
   
    def split_data(self, df):
        X = X = df.drop('price', axis=1)
        y = df['price']
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def scale_data(self, X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train = X_train_scaled
        X_test = X_test_scaled
        return X_train, X_test, scaler

    def save_scaler(self, scaler, scaler_file):
        scaler_path = self.paths.clean_data_dir / str(scaler_file + '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    def remove_outliers(self,df):
        # Calculate Q1, Q3, and IQR for each feature 
        selected_features = ['carat','depth','table','price']
        Q1 = df[selected_features].quantile(0.25)
        Q3 = df[selected_features].quantile(0.75)
        IQR = Q3 - Q1
        # Calculate lower and upper limits for each feature
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        # Remove outliers
        df_no_outliers = df[~((df[selected_features] < lower_limit).any(axis=1) | (df[selected_features] > upper_limit).any(axis=1))]
        
        return df_no_outliers

    def preprocess(self ):
        # Take the latest file from raw data directory
        csv_files = list(self.paths.raw_data_dir.glob('*.csv'))
        if not csv_files:
            print('No new data to clean')
        latest_file = max(csv_files, key=lambda file: file.stat().st_mtime)
        
        new_df = pd.read_csv(latest_file)
        
        # Preprocess the new single batch
        new_clean_df = self.process_single_batch(new_df)
        
        # Save in clean dir
        new_clean_df.to_csv(self.paths.clean_data_dir/latest_file.name, index=False)
         
        # Concatenate all datasets in the clean data directory
        df = self.load_clean_data()     
        
        # Perform operations that must be done aggregately:
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = self.split_data(df)

        # Scale the dataset
        X_train, X_test, scaler = self.scale_data(X_train, X_test)

        # Save the scaler
        self.save_scaler(scaler, latest_file.stem)

        return X_train, X_test, y_train, y_test, df.columns.tolist(), df
    
if __name__ == '__main__':
    paths =  Path_Config()
    params = Params(paths.params_path)

    ds = DatasetProcessor(paths,  params.last_dataset_update, params.train_temp_split_percentage , params.rand_state )
    X_train, X_test, y_train, y_test, col, df = ds.preprocess()
    print(X_train, X_test, y_train, y_test)
 
