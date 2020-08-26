# importing required packages
import pandas as pd
import datetime


# read the dataset and create a dataframe
dataset = pd.read_csv('dataset/WS_buildaccountdata.csv')


# helper function to change 'tm_submit_time' to binary class labels
# according to office time i.e if time is in between 0900 to 1800
# assign it to class label 1 else 0
def impute_time(x):
    if x >= datetime.time(9, 0, 0) and x<=datetime.time(18, 0, 0) :
        return 1
    else:
        return 0
dataset['tm_submit_time'] = dataset["tm_submit_time"].apply(lambda x : datetime.datetime.strptime(x, '%H:%M:%S').time())
dataset['tm_submit_time'] = dataset["tm_submit_time"].apply(impute_time)
dataset["dt_submit_date"] = dataset["dt_submit_date"].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').date().weekday())


# helper function to change 'dt_submit_date' to binary class labels
# according to office working days i.e from Monday to Saturday 
# assign it to class label 1 else 0
def impute_days(x):
    if x in([0,1,2,3,4,5]):
        return 1
    else:
        return 0
dataset["dt_submit_date"] = dataset["dt_submit_date"].apply(impute_days)   


# change the 'tx_revision' column to length of each distinct values in that column
dataset["tx_revision_len"] = dataset["tx_revision"].apply(lambda x : len(str(x)))

# change the 'id_infra_instan' column to length of each distinct values in that column
dataset["id_infra_nod"] = dataset["id_infra_instan"].apply(lambda x : len(str(x)))

# change the 'id_emp_id' column to length of each distinct values in that column
dataset["id_emp_nod"] = dataset["id_emp_id"].apply(lambda x : len(str(x)))


# adding the column 'nu_loc_flux' which will be the difference between 
# 'nu_loc_added' and 'nu_loc_removed' 
loc_flux = dataset['nu_loc_added'] - dataset['nu_loc_removed']
dataset = pd.concat([dataset, loc_flux], axis=1)
dataset.rename(columns = {0:'nu_loc_flux'}, inplace = True)


# drop all the columns from the dataset which have been modified respectively
dataset.drop(['nu_loc_added', 'nu_loc_removed', 'tx_revision', 'id_infra_instan', 'id_emp_id'], axis = 1, inplace = True)


# save the new dataset
dataset.to_csv('working_.csv')


# print the shape and columns name of the new dataframe
print(dataset.shape)
print('\n')
print(dataset.columns)