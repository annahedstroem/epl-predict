# Import key packages.
import platform
print("python: {}".format(platform.__version__)) 
import sklearn
print("sklearn: {}".format(sklearn.__version__), ' /Note that version 0.18 is required')

# Numerical capacity.
import pandas as pd
print("pandas: {}".format(pd.__version__))
import numpy as np
print("numpy: {}".format(np.__version__),  '  /Note that version 1.11 is required')

# Plotly setup and other pretty graphs.
import plotly.plotly as py
import matplotlib.pyplot as plt

import plotly
plotly.offline.init_notebook_mode()
import visplots
import seaborn as sns

# Feature selection.
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# For modeling and validation.
from sklearn import preprocessing, metrics
from sklearn import cross_validation as cv
from sklearn.cross_validation import train_test_split
from sklearn.metrics.cluster import silhouette_score
from sklearn.cross_validation import cross_val_score
from sklearn import model_selection

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression


# Import the data and take a look.
epl = pd.read_csv("Users/annahedstrom/projects/epl-challenge/training-data.csv", index_col='ID', encoding='ISO-8859-1')
bookies = pd.read_csv("/Users/annahedstrom/projects/epl-challenge/epl-bookies.csv", index_col='ID', encoding='ISO-8859-1')
test = pd.read_csv("/Users/annahedstrom/projects/epl-challenge/epl-test.csv", encoding='ISO-8859-1')
header = epl.columns.values

# Ensure that dates is parsed correct.
epl['Date'] = pd.to_datetime(pd.Series(epl['Date']),format="%d/%m/%Y")
test['Date'] = pd.to_datetime(pd.Series(test['Date']),format="%d/%m/%Y")

# Inspect the data.
epl.tail()

# Inspect the data.
print(bookies.tail())
test.tail()

# Import the additional data and take a look.
added = pd.read_csv("added-data.csv", encoding='ISO-8859-1')

# Inspect the data.
added.head(10)

# Explore the dataset.
epl.info()
epl.describe()

# Some other ways I usually explore and look at data.
epl.columns
epl.index

# Explore the added dataset.
added.info()
added.ix[:3]

# Quick insight to the distribution of FTR using pd.Categorial.
distribution = pd.Categorical(epl.FTR)
distribution.describe()

# Quick insight to how different features correlates with the target value FTR. 
# This will tell me if I should dive more into any particular columns.
epl.corr()

# I gain insight to which values to convert by using dtypes.
epl.dtypes

# I replace the value of FTR and HTR with a numerical value using the replace function.
epl['FTR'] = epl['FTR'].replace('A', 1)
epl['FTR'] = epl['FTR'].replace('H', 2)
epl['FTR'] = epl['FTR'].replace('D', 0)
epl['HTR'] = epl['HTR'].replace('A', 1)
epl['HTR'] = epl['HTR'].replace('H', 2)
epl['HTR'] = epl['HTR'].replace('D', 0)

# Inspect results.
epl.tail()


# Append the added data to the epl data to ensure that the dataset is easy to operate.
epl['HT'] = added['HT']
epl['AT'] = added['AT']
epl['HF'] = added['HF']
epl['AF'] = added['AF']

# Craft a histogram of all the games in the FTR column.
plt.hist(epl["FTR"])

# Show the plot.
plt.show()

pd.DataFrame(epl.groupby(['HomeTeam', 'AwayTeam']).size().reset_index(name = "Group_Count"))

pd.DataFrame((epl.groupby(['HomeTeam'])['FTHG'].mean() / epl.groupby(['HomeTeam', 'AwayTeam']).size()).reset_index(name = "HomeTeam_Avg_Goals"))

pd.DataFrame((epl.groupby(['AwayTeam'])['FTHG'].mean() / epl.groupby(['HomeTeam', 'AwayTeam']).size()).reset_index(name = "HomeTeam_Avg_Goals"))

pd.DataFrame((epl.groupby(['HomeTeam'])['HST'].count() / epl.groupby(['HomeTeam', 'AwayTeam']).size()).reset_index(name = "HomeTeam_Avg_ShotsTarget"))

current = pd.concat(epl_final)
current.tail(10)


epl_final = []
epl_group = epl.groupby(['HomeTeam','AwayTeam'])
for key, item in epl_group:
    current_group = epl_group.get_group(key)
    current_group['Goal_diff'] = (current_group['FTHG'] - current_group['FTAG']).mean()
    current_group['Shots_diff'] = (current_group['HS'] - current_group['AS']).mean()
    current_group['ShotsTarget_diff'] = (current_group['HST'] - current_group['AST']).mean()
    current_group['Corner_diff'] = (current_group['HC'] - current_group['AC']).mean()
    current_group['YellowCard_diff'] = (current_group['HY'] - current_group['AY']).mean()
    current_group['RedCard_diff'] = (current_group['HR'] - current_group['AR']).mean()
    current_group['Fans'] = (current_group['HF'] > current_group['AF']).mean()
    current_group['Touches'] = (current_group['HT'] - current_group['AT']).mean()
    epl_final.append(current_group)
    
# Note that the purpose of the "SettingWithCopyWarning"  is to flag to the user that the assignment
# is carried out on a copy of the data frame slice instead of the original data frame itself. This is intented.

epl = current
epl_final = []
epl_group = epl.groupby(['HomeTeam'])
for key, item in epl_group:
    current_group = epl_group.get_group(key)
    current_group['Shots_game'] = current_group['FTHG'].mean() / current_group['HS'].mean()
    current_group['ShotsTarget_game'] = current_group['FTHG'].mean() / current_group['HST'].mean()
    epl_final.append(current_group)


# Join the new features to interim dataset.
current = pd.concat(epl_final)
current.tail(10)

epl = current
epl_final = []
epl_group = epl.groupby(['AwayTeam'])
for key, item in epl_group:
    current_group = epl_group.get_group(key)
    current_group['Shots_gameAway'] = current_group['FTAG'].mean() / current_group['AS'].mean()
    current_group['ShotsTarget_gameAway'] = current_group['FTAG'].mean() / current_group['AST'].mean()
    epl_final.append(current_group)

# Join the new features to interim dataset.
current = pd.concat(epl_final)
current.tail(10)


# Inspect the new features.
epl = pd.concat(epl_final)
epl


# Set the columns to false.
epl["HomeTeamLastWin"] = False
epl["AwayTeamLastWin"] = False

# Create a defaultdict that will default to zero the first time a team won their last game.
last_win = defaultdict(int)

# Create a HomeWin feature, representing if the home team won the game or not.
epl["HomeWin"] = epl["FTAG"] < epl["FTHG"]
FTR_true = epl["HomeWin"].values

# Iteratively loop over all the teams and games to see if the team won their last game.
for index, row in epl.iterrows():
    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]
    row["HomeTeamLastWin"] = last_win[home_team]
    row["AwayTeamLastWin"] = last_win[away_team]
    epl.ix[index] = row

last_win[home_team] = row["HomeWin"]
last_win[away_team] = not row["HomeWin"]

# Update the dictionary so the model know if the specific team won their last game or not.


# Inspect the results.
epl.tail()

# Set the  columns to zero.
epl["HomeTeam_WinStreak"] = 0
epl["AwayTeam_WinStreak"] = 0

# Create a defaultdict.
win_streak = defaultdict(int)

# Itertively, increment the counter for the winner and set it to zero for the loser.
for index, row in epl.iterrows():
    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]
    row["HomeTeam_WinStreak"] = win_streak[home_team]
    row["AwayTeam_WinStreak"] = win_streak[away_team]
    epl.ix[index] = row
    
    if row["HomeWin"]:
        win_streak[home_team] += 1
        win_streak[away_team] = 0
    
    else:
        win_streak[home_team] = 0
        win_streak[away_team] += 1

# Inspect the results.
print(epl.columns)
epl.tail()

# Convert date values to consecutive day-indices. 
epl['days'] = ((epl['Date'] - epl.iloc[-1]['Date']) / np.timedelta64(1, 'D')).astype(int)

# Convert float64 to integer.
epl[['Fans']] = epl[['Fans']].astype(int)
epl[['Touches']] = epl[['Touches']].astype(int)

# I concatenating the test and epl data frame to retrieve the features for the games to be predicted.
test_epl = pd.concat([epl, test])

# Inspect the new columns.
test_epl.columns

# I use sort_values and forward-fill to produce corresponding data points for the test set. 
test_epl.sort_values(['HomeTeam', 'AwayTeam','Date'], inplace=True)
test_epl.fillna(method='ffill',inplace=True)

# Inspect the test_test to ensure that the imputation was executed as expected.
test_epl[(test_epl['HomeTeam']=='Liverpool') & (test_epl['AwayTeam']=='Swansea')]

# Convert categorial values of HomeTeam and AwayTeam to binary ( and features).
epl_ready = pd.get_dummies(data=test_epl, columns=['HomeTeam', 'AwayTeam'])

# The isin function will help filter out all the rows inside the test dataframe.
test_data = epl_ready[epl_ready['Date'].isin(test.Date)]

# This remove all the rows which are inside test for the original train data.
epl_ready = epl_ready[~epl_ready['Date'].isin(test.Date)]

# Inspect the results.
epl_ready.head()

# Inspect the shapes pf the different datasets.
print(epl_ready.shape)
print(test_data.shape)

# Filter the columns and remove the ones unsuitable for model training, named remove_cols.
remove_cols = ['HomeWin', 'AT', 'HT', 'Referee','Date', 'FTHG', 'FTAG', 'HTHG', 'HTR', 'HTHG', 'HTAG','HS','AS', 'HST', 'AST', 'HF', 'AS', 'HC', 'AC', 'AF', 'HY','AY', 'HR','AR']
epl_ready = epl_ready.drop(remove_cols, axis=1)   
test_data = test_data.drop(remove_cols, axis=1)

# Drop FTR for test_data.
test_data = test_data.drop(['FTR'], axis=1)

# Print shapes. There should be one less column for test_data.
print(epl_ready.shape, test_data.shape)

# Split to input matrix X and class vector y.
X = epl_ready.drop(['FTR'], axis=1)
y = epl_ready.as_matrix(["FTR"]).ravel()
print(y)
X.columns
y.shape

# Create an array and confirm the dimensionality there are 2108 rows and 57 columns.
print("X dimensions:", X.shape)
print("y dimensions:", y.shape)
X.head(10)

Corr = round(epl_ready.corr()["FTR"],3)
Corr

#PCA

# I Initialise the model with specifying the number of clusters and random state. 
kmeans = KMeans(n_clusters=5, random_state=1)

# I only want the numeric columns from ready dataset.
num_columns = epl_ready._get_numeric_data()

# I fit the model using the numericals columns.
kmeans.fit(num_columns)

# Produce the cluster assignments where all rows are included.
labels = kmeans.labels_


# Create the PCA model.
pca_2 = PCA(2)

# I fit the principal component analysis model on the numeric columns produced earlier.
plot_columns = pca_2.fit_transform(num_columns)

# I make a scatterplot of each games. It is shaded after cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

# Print out the variance ratios for PCA.
pca = PCA(2)
pca.fit(num_columns, y)
print(pca.explained_variance_ratio_)

# Generate the training sets. I want reproducible results and therefore set random_state= 1. 
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=1, test_size=0.3)

# Inspect and confirm the results.
print("XTrain:", XTrain.shape, "yTrain:", yTrain.shape, "XTest:", XTest.shape, "yTest:",  yTest.shape)

# Building the classification model using a pre-defined parameter.
dtc = DecisionTreeClassifier(max_depth=3, random_state=1)

# Train the model.
dtc.fit(XTrain,yTrain)

# Test the model.
preddtc = dtc.predict(XTest)

# Report the metrics using metrics.classification_report.
print(metrics.classification_report(yTest, preddtc))
print("Overall Accuracy: ", round(metrics.accuracy_score(yTest, preddtc), 3))

# Get the confusion matrix for the classifier using metrics.confusion_matrix.
matrix_dtc = metrics.confusion_matrix(yTest, preddtc)
print("Confusion matrix Decision trees:",matrix_dtc)

# Parameters I can investigate include:
n_estimators = np.arange(5, 100, 25)
max_depth    = np.arange(1, 35, 5)

# Percentage of features to consider at each split.
max_features = np.linspace(.1, 1.,3)
parameters   = [{'n_estimators': n_estimators,
                 'max_depth': max_depth,
                 'max_features': max_features}]

# Building the grid search model.
gridCV = GridSearchCV(RandomForestClassifier(), param_grid=parameters, cv=5, n_jobs=4) 

# Train the model.
gridCV.fit(XTrain, yTrain)

best_n_estim      = gridCV.best_params_['n_estimators']
best_max_depth    = gridCV.best_params_['max_depth']
best_max_features = gridCV.best_params_['max_features']

# Print the best parameters.
print('Best parameters: n_estimators=', best_n_estim, 'max_depth=', best_max_depth, 'max_features=', best_max_features)

# Build a Random Forest classifier with best parameters.
rf = RandomForestClassifier(n_estimators=best_n_estim, max_depth=best_max_depth, max_features=best_max_features)

# Train the model.
rf.fit(XTrain, yTrain)

# Test the model.
predrf = rf.predict(XTest)
#probs = rf.predict_proba(XTest)
#print(probs)

# Report the metrics using metrics.classification_report and accuracy_score.
print(metrics.classification_report(yTest, predrf))
print("Overall Accuracy:", round(metrics.accuracy_score(yTest, predrf),3))

# Get the confusion matrix for the classifier using metrics.confusion_matrix.
matrix_rf = metrics.confusion_matrix(yTest, predrf)
print("Confusion matrix Random forest:",matrix_rf)


# Create a visplot.
visplots.rfAvgAcc(rfModel=rf, XTest=XTest, yTest=yTest)

# Stochastic Gradient Boosting Classification

# First I ensure that input matrix X and class vector y have remained the same.
X = epl_ready.drop(['FTR'], axis=1)
y = epl_ready.as_matrix(["FTR"]).ravel()

# I define number of number of folds, plus splits for the cross validation.
kfold = model_selection.KFold(n_splits=5, random_state=1)

# Build the model.
gb = GradientBoostingClassifier(n_estimators=30, random_state=1, max_depth=1, max_features=1.0)

# Fit the model.
gb.fit(XTrain, yTrain)

# Train the model.
predgb = model_selection.cross_val_score(gb, X, y, cv=kfold)

# Print results.
print(predgb.mean())

# Print the overall accuracy for the models.
print("Overall Accuracy Decision trees: ", round(metrics.accuracy_score(yTest, preddtc), 3))
print("Overall Accuracy Random forest:", round(metrics.accuracy_score(yTest, predrf), 3))
print("Overall Accuracy Stochastic Gradient Boosting:", round(predgb.mean(), 3))

import seaborn

# Create a heatmap out of the confusion matrix for Decision Tree Classifier.
seaborn.heatmap(metrics.confusion_matrix(yTest,preddtc), cmap="YlGnBu") 
plt.xlabel('True Labels DTC') 
plt.ylabel('Predicted Labels DTC') 

# Create the plot.
plt.show()

# Create a heatmap out of the confusion matrix for Decision Tree Classifier.
seaborn.heatmap(metrics.confusion_matrix(yTest,predrf), cmap="YlGnBu") 
plt.xlabel('True Labels RF') 
plt.ylabel('Predicted Labels RF')

# Create the plot.
plt.show()


# I have cerated a function that iteratively maps test_data to test data frame, to get index the for prediction classification.
def find_predict(test, test_data, model):
    predgb = model.predict(test_data)
    for index, row in test.iterrows():
        home, away = 'HomeTeam_' + row['HomeTeam'], 'AwayTeam_' + row['AwayTeam']
        ids = test_data[(test_data[home] == 1.0) & (test_data[away] == 1.0)].index.tolist()
        test.loc[index, 'FTR'] = predgb[ids[0]]
    return test


# I store the predictions in a pandas dataframe.
predictions = pd.DataFrame(find_predict(test, test_data, gb))

# Print the predictions.
predictions

# I replace the value of FTR and HTR with an numerical value using the replace function.
predictions['FTR'] = predictions['FTR'].replace(1, 'A')
predictions['FTR'] = predictions['FTR'].replace(2, 'H')
predictions['FTR'] = predictions['FTR'].replace(0, 'D')

# Print the final dataframe.
predictions

predictions.to_csv("submission.csv", index=False)

##### ##### ##### ##### ##### ##### ##### #####
# Others
# Building the classification model.
log = LogisticRegression()

# Train the model.
log = log.fit(XTrain, yTrain)

# Test the model.
predlog = log.predict(XTest)

# Report the metrics using metrics.classification_report.
print(metrics.classification_report(yTest, predlog))
print("Overall Accuracy:", round(metrics.accuracy_score(yTest, predlog),2))

# Get the confusion matrix for your classifier using metrics.confusion_matrix.
matrix_log = metrics.confusion_matrix(yTest, predlog)
print("Confusion matrix Logistic Regression:", matrix_log)
