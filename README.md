# QL - Fraud Detection with Machine Learning

This script performs automated machine learning in detecting the potential fraud.

It means when we runs the script, the script will run through all the machine learning algorithm with the evaluation, often with accuracy, so that we can choose the best machine learning algorithm. 

To begin with, the script just require a cleaned data with selected feature, and put it in the current directory.

1. Clone the git project
```
git clone https://github.com/TeohYx/QL-CashTransactionDetector.git
```

2. Provide a cleaned dataset <br>
Step to have a cleaned dataset: <br>
1. Remove all the missing / incomplete data <br>
    This can be done in the Excel.
2. Obtain the normal and fraud dataset. <br>
    As the models predict the outcome through classification, therefore the dataset for each class is needed. <br>
    In this case, it is "Normal" and "Fraud" <br>
    This means that the dataset that are known to be "Normal" and "Fraud" is required. <br>
3. Choose the features <br>
    The features that are think to be good contribution on predicting fraud can be used. <br>
    Besides, feature engineering can also be performed. <br>
    For example, there are "Transaction Close time" and "Transaction Start time". Therefore the "Transaction duration" can be obtained by subtracting "Transaction Close time" and "Transaction Start time".
```
Transaction duration = Transaction Close time - Transaction Start time
```
![Example Dataset](screenshot/Example%20Dataset.png)

3. Run the script
```
python Fraud.py
```