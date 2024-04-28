import json
import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import Config_Path
from params import Params

class DataLoader:
    def __init__(self, dataset_dir: Path, last_update):
        self.dataset_dir = dataset_dir
        self.last_update = last_update
        self.column_order = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']

    def concatenate_csv_files(self, csv_files):
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            if df.columns.tolist() == self.column_order:
                dfs.append(df)
            else:
                raise ValueError(f"Incorrect columns order in file {csv_file}")

        if len(dfs) > 1:
            combined_df = pd.concat(dfs)
        elif len(dfs) == 1:
            combined_df = dfs[0]
        else:
            raise ValueError("No CSV files found in directory")
        return combined_df

    def get_newer_files(self, directory_path, last_update_date):
        newer_files = []
        for file in directory_path.glob('*.csv'):
            file_date = datetime.strptime(file.stem, '%Y%m%d_%H%M%S')
            if file_date > last_update_date:
                newer_files.append(file)
        
        if len(newer_files) == 0:
            raise ValueError(f"No newer files since {last_update_date}")
            #TODO FAI CONTINUARE IL PROGRAMMA
        return newer_files

    def load_new_data(self):
        last_update = datetime.strptime(self.last_update, '%Y%m%d_%H%M%S')
        new_files = self.get_newer_files(self.dataset_dir / 'new_data', last_update)
        df = self.concatenate_csv_files(new_files)
        now = datetime.now()
        filename = now.strftime("%Y%m%d_%H%M%S")
        df.to_csv((self.dataset_dir / 'raw_data') / str(filename + '.csv'), index=False)

        print(df.head(), len(df))
        
        return filename



if __name__ == '__main__':
    
    paths = Config_Path()
    params = Params(paths.params_path)

    data_loader = DataLoader(paths.data_dir, params.last_dataset_update)  
    last_update = data_loader.load_new_data()
    
    params.last_dataset_update = last_update
    params.save(paths.params_path)
    
    

