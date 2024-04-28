
import json
from pathlib import Path
class Params():
    def __init__(self, json_path):
        self.update(json_path)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        self.convert_path_to_str()
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def __repr__(self):
        return str(self.__dict__)
    
    def convert_path_to_str(self):
        for k,v in self.__dict__.items():
            if isinstance(v,Path):
                setattr(self, k, str(v))


if __name__ == "__main__": 
 
    # Load the hyperparameters from the JSON file
    params = Params('./challenge_2/params.json')

    # Access the different entries
    print(params.model_version)  # baseline
    print(params.learning_rate)  # 0.001
    print(params.batch_size)     # 32
    print(params.num_epochs)     # 10
    
    params.batch_size = 3
    
    params.save('./challenge_2/paramstest.json')
