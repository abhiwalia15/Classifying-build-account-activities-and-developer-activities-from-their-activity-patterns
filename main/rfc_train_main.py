# import all the required packages for preprocessing, 
# training and evaluation of the model
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from  sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# read the modified dataframe
odf = pd.read_csv('working_.csv')


# select required columns and create a new separate dataframe
dat_org = odf[['tx_revision_len','id_infra_nod', 'tx_state', 'tm_submit_time',
       'dt_submit_date','nu_loc_flux','nu_loc_total', 'nu_comment_flux', 'nu_comment_total',
       'nu_filesize_flux', 'nu_filesize_total', 'nu_cyclo_flux',
       'nu_cyclo_total', 'nu_fanout_flux',
       'nu_fanout_total', 'nu_halstead_flux', 'nu_halstead_total',
       'nu_nom_flux', 'nu_nom_total','label']]


# One-Hot Encode the 'tx_state' feature variable
state = pd.get_dummies(dat_org['tx_state'], drop_first=True)
dat_org.drop(['tx_state'], axis=1, inplace=True)
dat_org = pd.concat([dat_org, state], axis=1)
dat_org.rename(columns = {'Rollback':'State_Rollback', 'Skipped':'State_Skipped', 'Unknown': 'State_Unknown'}, inplace = True)


# One-Hot Encode the 'label' target variable
label = pd.get_dummies(dat_org['label'], drop_first=True)
dat_org.drop(['label'], axis=1, inplace=True)
dat_org = pd.concat([dat_org, label], axis=1)
dat_org.rename(columns = {True:'label'}, inplace = True)

df1 = dat_org.copy()


# separate independent and dependent variables
X = df1.drop('label',axis=1)
y = df1['label']
# split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)


# fillin the missing values using iterative imputer
imp = IterativeImputer(n_nearest_features=15, max_iter=10, random_state=0)
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)


# convert back to dataframe
X_train = pd.DataFrame(X_train,columns=df1.columns[:-1])
X_test = pd.DataFrame(X_test,columns=df1.columns[:-1])


# separate the columns from dataframe and add them back to the dataframe after scaling the dataframe 
misc_xtrain = X_train[['tx_revision_len', 'id_infra_nod', 'dt_submit_date','State_Rollback', 'State_Skipped', 'State_Unknown', 'tm_submit_time']]
misc_xtest = X_test[['tx_revision_len', 'id_infra_nod', 'dt_submit_date','State_Rollback', 'State_Skipped', 'State_Unknown', 'tm_submit_time']]
X_train.drop(['tx_revision_len', 'id_infra_nod', 'dt_submit_date','State_Rollback', 'State_Skipped', 'State_Unknown', 'tm_submit_time'], inplace=True, axis=1)
X_test.drop(['tx_revision_len', 'id_infra_nod', 'dt_submit_date','State_Rollback', 'State_Skipped', 'State_Unknown', 'tm_submit_time'], inplace=True, axis=1)


# scale the dataframe using RobustScaler
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# add the columns back to the dataframe
col = ['nu_loc_flux', 'nu_loc_total', 'nu_comment_flux', 'nu_comment_total',
       'nu_filesize_flux', 'nu_filesize_total', 'nu_cyclo_flux',
       'nu_cyclo_total', 'nu_fanout_flux', 'nu_fanout_total',
       'nu_halstead_flux', 'nu_halstead_total', 'nu_nom_flux', 'nu_nom_total']


X_train = pd.DataFrame(X_train,columns=col)
X_test = pd.DataFrame(X_test,columns=col)
X_train = pd.concat([X_train, misc_xtrain], axis = 1)
X_test = pd.concat([X_test, misc_xtest], axis = 1)


# save the training and testing files
X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')


# train the model using RandomForestClassifier algorithm 
rfc = RandomForestClassifier(n_estimators=50,max_features=4)
rfc.fit(X_train, y_train)


# making predictions on the testing set 
y_pred_g_test = rfc.predict(X_test)

# evaluate the predictions on test_data
print('Confusion Matrix: ', confusion_matrix(y_test,y_pred_g_test))
print()

print('Classification Report: ', classification_report(y_test,y_pred_g_test))
print()

print('Accuracy Score: ', accuracy_score(y_test,y_pred_g_test))
print()

y_pred_g_test_ = rfc.predict_proba(X_test)
y_pred_g_test_ = [p[1] for p in y_pred_g_test_]
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred_g_test_) )
print()

print('---------------------------------------------------')

# Feature Importances
importance = rfc.feature_importances_
features = X_train.columns.to_list()
importance = importance.tolist()

for i,j in zip(importance,features):
    print('%s --> %.5f' % (j,i))

# Visual Representation of feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title("Features Importance Plot")
plt.xticks([x for x in range(len(importance))], features, rotation=45)
plt.show()

print('---------------------------------------------------')

# save the model
filename = 'final_model.sav'
pickle.dump(rfc, open(filename, 'wb'))