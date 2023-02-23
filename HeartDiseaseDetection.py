"""
The dataset used here is the Cleveland Heart Disease dataset, which contains 14 attributes and a target
variable indicating the presence or absence of heart disease.
You can download the dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/Heart+Disease
"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

""" Load the dataset into a pandas dataframe and examine its features """
# Load the dataset from an url
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
df = pd.read_csv(url, header=None)

# Examine the features of the dataset
# print(df.head())
# print(df.info())
# print(df.describe())

""" Prepare the dataset for machine learning by converting categorical variables into dummy variables 
and scaling the continuous variables. """
# Add column names to the dataframe
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df.columns = columns

# Replace missing values with the mean value of the column
imputer = SimpleImputer(strategy='mean')
df = df.replace('?', np.nan)
df[['ca']] = imputer.fit_transform(df[['ca']])

# Convert categorical variables using ordinal encoding
ordinal_encoder = OrdinalEncoder()
df[['cp', 'thal']] = ordinal_encoder.fit_transform(df[['cp', 'thal']])

# Scale numerical variables using standardization:
scaler = StandardScaler()  # You can use standardization or normalization
# Scale the age, trestbps, chol, thalach, and oldpeak columns, so that they have a similar range of values
ct = ColumnTransformer([('scaler', scaler, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])], remainder='passthrough')
df_scaled = ct.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=columns)

# Examine the cleaned and preprocessed dataset
# print(df_scaled.head())
# print(df_scaled.info())
# print(df_scaled.describe())


""" Split the dataset into training and testing sets """
# Split the data into features and target variables
features = df.drop('target', axis=1)
targets = df['target']
# Convert categorical variables to dummy variables
features = pd.get_dummies(features, columns=['cp', 'restecg', 'slope', 'thal'])

seed = random.randint(1, 1000)  # seed = 42
# 70% go to training and 30% to testing
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=seed)

# Scale the features using standard scaling to ensure that they are on the same scale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

""" Train several classification algorithms on the training set, including logistic regression, decision trees, 
random forests, support vector machines, and k-nearest neighbors. """
# Define a list of classifiers to be used
classifiers = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), SVC(probability=True),
               KNeighborsClassifier()]
accuracies = {}
precisions = {}
recalls = {}
f1s = {}

# Train each classifier on the training set and test on the testing set
for classifier in classifiers:
    classifier.fit(x_train, y_train)
    # Predict labels on the testing set
    y_prediction = classifier.predict(x_test)
    y_prediction_prob = classifier.predict_proba(x_test)[:, 1]
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction, average='macro', zero_division=1)
    recall = recall_score(y_test, y_prediction, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_prediction, average='macro', zero_division=1)
    # Store the results
    accuracies[classifier.__class__.__name__] = accuracy
    precisions[classifier.__class__.__name__] = precision
    recalls[classifier.__class__.__name__] = recall
    f1s[classifier.__class__.__name__] = f1
    # Print evaluation metrics
    # print('----- '+classifier.__class__.__name__+' -----')
    # print('Accuracy:', accuracy)
    # print('Precision:', precision)
    # print('Recall:', recall)
    # print('F1 score:', f1)
    # print()

# Choose the best classifier based on performance
best_classifier = max(classifiers, key=lambda clf: accuracy_score(y_test, clf.predict(x_test)))
# print(f"The best classifier is: {type(best_classifier).__name__}")

# Display some plots to show the performance differences
# ax1 = plt.subplot(221)
# ax2 = plt.subplot(222)
# ax3 = plt.subplot(223)
# ax4 = plt.subplot(224)

# for name, value in recalls.items():
#     print(name, value)
#     ax1.plot(np.arange(0, 100, 1), value*100, 'b')

# # Create lists of classifier names and accuracy scores
# classifiers = list(recalls.keys())
# scores = list(recalls.values())

# Create a bar plot of the accuracy scores
# plt.bar(classifiers, scores)
# plt.xticks(rotation=30, ha='center')
# # Adjust the margins of the plot
# plt.subplots_adjust(bottom=0.25, left=0.1)
#
# # Add a title and labels for the axes
# plt.title('Classifier Performances')
# # plt.xlabel('Classifier')
# plt.ylabel('Recalls')

# Display the plot
# plt.show()
#
#

classifiers_names = list(accuracies.keys())
index = np.arange(len(classifiers_names))

plt.figure(figsize=(12, 8))

plt.bar(index, list(accuracies.values()), width=0.2, color='b', label='Accuracy')
plt.bar(index + 0.2, list(recalls.values()), width=0.2, color='r', label='Recall')
plt.bar(index + 0.4, list(precisions.values()), width=0.2, color='g', label='Precision' )
plt.bar(index + 0.6, list(f1s.values()), width=0.2, color='y', label='F1 score' )

plt.title("Classifiers Performances")
plt.xlabel("Classifier")
plt.ylabel("Performances")
plt.xticks(index+0.2, classifiers_names)
plt.legend(loc='upper right')

plt.show()
