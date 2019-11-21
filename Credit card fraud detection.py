import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
import random

# Before reading the dataset one package need to import to set the working directory by the below code

import os
os.getcwd()
os.chdir('C:\\Users\\aruns\\Downloads\\CCRD')
os.getcwd()

# Now we can load our dataset deireclty without giving any paths that where we save our data.

df = pd.read_csv('creditcard_sampledata.csv')
df.describe()
df.head()

# To calculate the total number of rows and columns by the below code

df.shape

# To check the Null values if any by the below code

df.isnull().values.any()

# In our dataset totally we have 20,350 transactions so let us calculate how many fraud transacation occured in our total transaction by
the below code.

class_names = {0:'Not Fraud', 1:'Fraud'}
print(df.Class.value_counts().rename(index = class_names))

# We have a highly imbalanced dataset on our hands. Normal transactions overwhelm the fraudulent ones by a large margin. Let's look at the 
two types of transactions:

count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
LABELS = ["Normal", "Fraud"]
plt.xticks(range(5), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");

# How different are the amount of money used in different transaction classes?

frauds.Amount.describe()
non_frauds.Amount.describe()

rauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]
print("We have total 20,350 transactions in that", len(frauds), "fraud transactions and", len(non_frauds), "non-fraudulent transations.")

# Lets plot the above details to visualize for more clarity.

ax = frauds.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')
non_frauds.plot.scatter(x='Amount', y='Class', color='Blue', label='Non-Fraud',ax=ax)

# Let's zoom in our fraud transactions to see the range of amounts.

ax = frauds.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')

# As per above plot most of the transactions are happened between 0 to 500.

# Do fraudulent transactions occur more often during certain time?

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')


ax1.scatter(frauds.Time, frauds.Amount)
ax1.set_title('Fraud')

ax2.scatter(non_frauds.Time, non_frauds.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

# Logistic Regression
# Before build the model import the required packages and divide the "X" & "Y" variables.
# Once we classified lets split our data for train and test dataset as 80/20.

from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state = 0)
print("Train and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
print("Total number of frauds:", len(y.loc[df['Class'] == 1]))
print("Number of frauds on y_test:", len(y_test.loc[df['Class'] == 1]))
print("Number of frauds on y_train:", len(y_train.loc[df['Class'] == 1]))

# Now we can calculate the Logistic score by the below code.

logistic = LogisticRegression(random_state=100)
logistic.fit(X_train, y_train)
print("Score: ", logistic.score(X_test, y_test))

y_predicted = logistic.predict(X_test)
y_actual = y_test

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_actual, y_predicted)
pd.crosstab(y_actual, y_predicted, rownames=['Actual'], colnames=['Predicted'],margins=True)

# 3987 transactions were classified as valid that were actually valid
# 03 transactions were classified as fraud that were actually valid (type 1 error)
# 17 transactions were classified as valid that were fraud (type 2 error)
# 63 transactions were classified as fraud that were fraud

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print ('Accuracy Score :',accuracy_score(y_actual, y_predicted) )
print ('Report : ')
print (classification_report(y_actual, y_predicted))

# Classification Rate/Accuracy: Accuracy = (TP + TN) / (TP + TN + FP + FN)= (3987+63) /(3987+63+03+17)= 0.99
# Recall: Recall gives us an idea about when it’s actually yes, how often does it predict yes. Recall=TP / (TP + FN)=3987/(3987+17)=0.99
# Precision: Precsion tells us about when it predicts yes, how often is it correct. Precision = TP / (TP + FP)=3987/ (3987+03)=0.99
# F-measure: Fmeasure=(2RecallPrecision)/(Recall+Presision)=(20.990.99)/(0.99+0.99)=0.99

import seaborn as sns
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
conf_matrix = confusion_matrix(y_actual,y_predicted)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('Actual class')
plt.xlabel('Predicted class')
plt.show()

We usally think that we got 99% of its predictions correct. That is true, except if you look closely at the confusion matrix you will 
see the
15 transactions were classified as fraud that were actually valid (type 1 error)
19 transactions were classified as valid that were fraud (type 2 error)
So, accuracy is not the reliable measure of a model’s effectiveness. In our case all the other measures like Precision, Recall, and F1 
scores are also good but totally we have 34 wrong predictions from our above model.May be if we drill our dataset more with Random Forest 
or the other options as below.One is over-sampling the sample of fraud records or, conversely, under-sampling the sample of good records.
Over-sampling means adding fraud records to our training sample, thereby increasing the overall proportion of fraud records. Conversely, 
under-sampling is removing valid records from the sample, which has the same effect. Changing the sampling makes the algorithm more 
“sensitive” to fraud transactions.¶


