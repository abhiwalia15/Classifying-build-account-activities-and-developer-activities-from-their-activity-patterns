# import all the required packages for preprocessing, 
# training and evaluation of the model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from  sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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


# using stratifiedKFold cross validation technique for splitting the 
# data into 8 folds and shuffle each fold
kf = StratifiedKFold(n_splits=5, shuffle = True)
rfc = RandomForestClassifier(n_estimators=50,max_features=5)

# initialize an empty array to store the results for each fold
scores = []

for i in range(5):
    
    # split the dataset for training and testing
    result = next(kf.split(X, y), None)
    X_train = X.iloc[result[0]]
    X_test = X.iloc[result[1]]
    y_train = y.iloc[result[0]]
    y_test = y.iloc[result[1]]

    
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


    # train the model using Gaussian Naive Bayes algorithm 
    rfc.fit(X_train, y_train)

    # append the accuracy score/ r2 score and print it for every iteration 
    scores.append(rfc.score(X_test, y_test))
    
    print("Iteration " + str(i) + ' - ' + str(rfc.score(X_test, y_test)))

# print the mean of all the scores
print("Mean Accuracy Score: ", np.mean(scores))