#in this file we have the directories paths 
from pathlib import Path

class Path_Config:
    def __init__(self):
        self.main_dir = Path('.').absolute()/'challenge_2'
        self.data_dir = self.main_dir/'datasets'
        self.new_data_dir = self.data_dir/'new_data'
        self.clean_data_dir = self.data_dir/'clean_data' 
        self.raw_data_dir = self.data_dir/'raw_data'
        self.logs_experiments = self.main_dir/'logs_experiments'
        self.params_path = self.main_dir/'params.json'
       
        
if __name__ == "__main__":
    config = Path_Config( )
    print(config.main_dir )
    print(config.data_dir)
    print(config.new_data_dir.is_dir())
    print(config.logs_experiments.is_dir())
    print(config.clean_data_dir.name)
    print(list(config.new_data_dir.rglob('*.csv')))
    print(config.params_path.exists())
    print(config.params_path)