import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np


# load the model from disk
rfc = pickle.load(open('final_model.sav', 'rb'))


# input the validation csv file for predictions 
dataset = pd.read_csv('filename') 


# data preparation
def impu(x):
    if x >= datetime.time(9, 0, 0) and x<=datetime.time(18, 0, 0) :
        return 1
    else:
        return 0

dataset['tm_submit_time'] = dataset["tm_submit_time"].apply(lambda x : datetime.datetime.strptime(x, '%H:%M:%S').time())
dataset['tm_submit_time'] = dataset["tm_submit_time"].apply(impu)
dataset["dt_submit_date"] = dataset["dt_submit_date"].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').date().weekday())

def impu_days(x):
    if x in([0,1,2,3,4,5]):
        return 1
    else:
        return 0

dataset["dt_submit_date"] = dataset["dt_submit_date"].apply(impu_days)   
dataset["tx_revision_len"] = dataset["tx_revision"].apply(lambda x : len(str(x)))
dataset["id_infra_nod"] = dataset["id_infra_instan"].apply(lambda x : len(str(x)))
dataset["id_emp_nod"] = dataset["id_emp_id"].apply(lambda x : len(str(x)))


loc_flux = dataset['nu_loc_added'] - dataset['nu_loc_removed']
dataset.drop(['nu_loc_added', 'nu_loc_removed', 'tx_revision', 'id_infra_instan', 'id_emp_id'], axis = 1, inplace = True)
dataset = pd.concat([dataset, loc_flux], axis=1)
dataset.rename(columns = {0:'nu_loc_flux'}, inplace = True)

dat_org = dataset[['tx_revision_len','id_infra_nod', 'tx_state', 'tm_submit_time',
       'dt_submit_date','nu_loc_flux','nu_loc_total', 'nu_comment_flux', 'nu_comment_total',
       'nu_filesize_flux', 'nu_filesize_total', 'nu_cyclo_flux',
       'nu_cyclo_total', 'nu_fanout_flux',
       'nu_fanout_total', 'nu_halstead_flux', 'nu_halstead_total',
       'nu_nom_flux', 'nu_nom_total']]


data = np.zeros((len(dat_org), 3))
state = pd.DataFrame(data)
state.columns = ["Rollback","Skipped","Unknown"]
dat_org = pd.concat([dat_org, state], axis=1)

for i in range(len(dat_org['tx_state'])):
    if dat_org['tx_state'].iloc[i] == 'Skipped':
        dat_org['Skipped'].iloc[i] = 1 

    elif dat_org['tx_state'].iloc[i] == 'Rollback':
        dat_org['Rollback'].iloc[i] = 1

    elif dat_org['tx_state'].iloc[i] == 'Unknown':
        dat_org['Unknown'].iloc[i] = 1
    

df1 = dat_org.copy()

df1_xtrain = df1[['nu_loc_flux', 'nu_loc_total', 'nu_comment_flux', 'nu_comment_total',
          'nu_filesize_flux', 'nu_filesize_total', 'nu_cyclo_flux',
          'nu_cyclo_total', 'nu_fanout_flux', 'nu_fanout_total',
          'nu_halstead_flux', 'nu_halstead_total', 'nu_nom_flux', 'nu_nom_total',
          'tx_revision_len', 'id_infra_nod', 'dt_submit_date','Rollback', 'Skipped',
           'Unknown', 'tm_submit_time']]


df1_xtrain = df1_xtrain.fillna(df1_xtrain.median())

# get prediction for new input
new_output = rfc.predict(df1_xtrain)

# summarize input and output
print(new_output)
