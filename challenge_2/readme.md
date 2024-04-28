# Challenge 2

## Directory Structure:

### `dataset/new_data`:
This directory stores periodically added new data. Each file's name represents the date of arrival. Example:
- `04/04.csv`
- `05/04.csv`
- `08/04.csv`
- `09/04.csv`

### `dataset/raw_data`:
Different versions of the dataset are stored here. Whenever new files are added to `new_data`, a new aggregated dataframe is created and saved in `raw_data`. This aggregated file is then used as input for the cleaning process. Example:
- `06/04.csv` (aggregates data from `new_data`: `04/04.csv` and `05/04.csv`)
- `11/04.csv` (aggregates data from `new_data`: `08/04.csv` and `09/04.csv`)

### `dataset/clean_data`:
Cleaned data, ready for preprocessing, is stored here. Each file corresponds to the cleaned version of the latest file in `raw_data`. Example:
- `06/04.csv` (derived from `raw_data` file `06/04.csv`)
- `11/04.csv` (derived from `raw_data` file `11/04.csv`)

Operations performed on the data in this folder include:
- Removing zero values
- Removing NaN values
- Removing negative prices
- Removing duplicates
- Removing columns 'y' and 'z'
- Log transformation
- Label encoding

## Classes:

### `DataLoader` class:
This class reads from `new_data` and writes to `raw_data`. It aggregates all files newer than the last update field in parameters and saves them in `raw_data` with the current date as the filename.

### `DataProcessor` class:
This class reads from `clean_data` and directly serves data to the model. It performs two types of operations:
1. Operations that can be performed on separate batches of data, as explained above.
2. Operations that must be performed on an aggregated form of the data, including removing outliers and calculating scalers.

## Logs and Experiments:

### `/logs_experiments`:
Different trainings made with various datasets are stored here. Each training folder contains:
- A file with useful variables and hyperparameters
- The trained model pickle
- Plots
