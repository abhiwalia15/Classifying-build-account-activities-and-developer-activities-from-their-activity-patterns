
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

data = pd.read_csv("WS_buildaccountdata.csv")

#Correlation matrix
sns.heatmap(data.corr(), yticklabels=True,cbar=True,cmap='coolwarm', annot=True, fmt='.1g',linewidths=3, linecolor='black')

#Heatmap (To check the missing values present in the dataset)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Primary Plotting 

plt.plot(data.nu_cyclo_flux)
plt.show() #saved as fig7

plt.plot(data.nu_cyclo_total)
plt.show() #saved as fig8

plt.plot(data.nu_dac_flux)
plt.show() #saved as fig9

plt.plot(data.nu_dac_total)
plt.show() #saved as fig10

plt.plot(data.nu_fanout_flux)
plt.show() #saved as fig11

plt.plot(data.nu_fanout_total)
plt.show() #saved as fig12

plt.plot(data.nu_nom_flux)
plt.show() #saved as fig13

plt.plot(data.nu_nom_total)
plt.show() #saved as fig14

#dt_submit_Date to working day or not
temp = data[["dt_submit_date","label"]]

temp["dt_submit_date"] = temp["dt_submit_date"].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').date().weekday())
# helper function to change 'dt_submit_date' to binary class labels
# according to office working days i.e from Monday to Saturday 
# assign it to class label 1 else 0
def impute_days(x):
    if x in([0,1,2,3,4,5]):
        return 1
    else:
        return 0
temp["dt_submit_date"] = temp["dt_submit_date"].apply(impute_days) 

#To plot how many true labels for whether it is working day or not
temp.groupby("dt_submit_date").label.sum().plot(kind='bar') #saved as fig1



#tm_submit_time to working hours or not
temp = data[["tm_submit_time","label"]]
temp["tm_submit_time"] = temp["tm_submit_time"].apply(lambda x : datetime.datetime.strptime(x, '%H:%M:%S').time())
def impu(x):
    if x >= datetime.time(9, 0, 0) and x<=datetime.time(18, 0, 0) :
        return 1
    else:    
        return 0
        
temp["tm_submit_time"] = temp["tm_submit_time"].apply(impu) 

temp.groupby("tm_submit_time").label.sum().plot(kind='bar') #saved as fig2


#id_emp_id based on the number of digits
temp = data[["id_emp_id","label"]]

temp["id_emp_id"] = temp["id_emp_id"].apply(lambda x : len(str(x)))

temp.groupby("id_emp_id").label.sum().plot(kind='bar') #saved as fig3

#id_infra_instan

temp = data[["id_infra_instan","label"]]
temp["id_infra_instan"] = temp["id_infra_instan"].apply(lambda x : len(str(x)))
temp.groupby("id_infra_instan").label.sum().plot(kind='bar') #saved as fig4
#tx_state
temp = data[["tx_state","label"]]
temp.groupby("tx_state").label.sum().plot(kind='bar') #saved as fig5
#tx_revision

temp = data[["tx_revision","label"]]
temp["tx_revision"] = temp["tx_revision"].apply(lambda x : len(x))
temp.groupby("tx_revision").label.sum().plot(kind='bar') #saved as fig6