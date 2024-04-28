# Challenge 2

### dataset/new_data
new_data is the directory where periodically new data is added
the filenames represent the date of arrival 

example:
- 04/04
- 05/04
- 08/04
- 09/04

### in dataset/raw_data
There are different versions of the dataset, everytime that there are new files in the newdata directory, a new aggregated dataframe is created and saved in raw_data. it is given in input to the cleaning class
from raw_data the latest file will be taken and cleaned and saved with current date too

example:
- 06/04  (aggregates the data that that in new_data dir are: 04/04 and 05/04)
- 11/04 (aggregates the data that that in new_data dir are: 08/04 and 09/04)

### In dataset/clean_data:
is stored cleaned data ready to receive aggregated preprocessing.
At every training the whole clean_data is being aggregated to perform a new training.

example:
- 06/04  (comes from the raw_data 06/04)
- 11/04 (comes from the raw_data 11/04)

The data in this folder received operations that can be performed in separate batches of data (for each new batch of data separately):
- remove 0 values
- remove na values
- remove negative prices
- remove duplicates 
- remove columns y e z
- transform to log
- label encoding

### dataloader class:
all the files newer than the last update field in params will be aggregated and saved in raw data with current date as filename in raw data
this class reads from /new_data and writes inside /raw_data

### dataprocessor class:
This class reads from /clean_data and directly serve data to the model.

This class  perform two kinds of operations:

1) Operations that can be performed in separate batches of data (for each new batch of data separately) as seen in the clean_data directiory explanation above

2) Operations that must be performed in a aggregated form (aggregating all the clean data):
- remove outliers
- split and calculate scaler

### in logs_experiments
are stored different trainings made with different datasets.
inside each training folder there are:
- a file with usefull variables and hyperparameters
- the trained model pickle 
- plots

