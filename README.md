# Classifying-build-account-activities-and-developer-activities-from-their-activity-patterns

Data on the changes in static metrics between code revisions made in version control systems. This captures a range of activities, and commits can contain either human coding effort or automated processes, e.g. importing or generating existing code. 

Commits from automation are not necessarily reflective of the effort spent by a developer making changes to code.   To make their own lives easier, commonly non-human effort can be isolated to specific non-human accounts, and spotting this sort of activity automatically can make analysis easier. We have developed automated solutions which capture the majority of build accounts in order to tackle this problem. This work sample concerns creating a machine learning approach to solving that same problem.

“Useful_Graphs” folder contains the graphs which are mentioned in the report and were obtained
as a result of Exploratory Data Analysis

➢ “EDA_visualisation” contains the code for Exploratory Data Analysis

➢ “dataset_prep.py” – Contains the code for Data Preperation

➢ “rfc_train_main” – Contains the code for performing training on the preprocessed data and saving
the model

➢ “final_model.sav” the final model that is obtained after training

➢ “rfc_kfold” – Contains the code for k fold cross validation training technique (Just to check mean
accuracy score for 5 folds)

➢ “predictions.py” – Contains the code to predict the validation dataset
(Please change the filename to your path of the validation file)

Order of Execution of Files:

➢ Run the “dataset_prep.py” , generates “working.csv”.

➢ Next Run the “”rfc_train_main.py”,generates “final_model.sav”(trained model)

➢ Now run the “predicitons.py” by setting the filename(validation data) path.

Note:Follow all the steps to build from scratch/Do the step 3 to just check the
trained model
