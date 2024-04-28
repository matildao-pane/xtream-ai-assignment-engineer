datasets are organized like this:

new_data is the directory where periodically new data is stored
we have new data incoming each filenames is the date of arrival 

data loader class:
all the files newer than the last update field in params will be aggregated and saved in raw data with current date as filename in raw data

in raw_data
there are different versions of the dataset, everytime that there are new files in  the newdata directory, a new aggregated dataframe is created and saved in raw_data. it is given in input to the cleaning class
from raw_data the latest file will be taken and cleaned and saved with current date too

in clean_data
we have cleaned data ready to be   given in input to the model. the latest model will be taken and given 


operations that can be performed in separate batches of data:
-remove 0 values
-remove na values
-remove negative prices
-remove duplicates 
-remove columns y e z
-transform to log
-label encoding

operations that must be performed in a aggregated form:
-remove outliers
-split and calculate scaler

