import sys
import numpy as np
import pandas as pd # type: ignore
from sklearn.linear_model import SGDClassifier, LogisticRegression  # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # type: ignore

data = pd.read_csv('..\\data\\traindata.csv')
testData = pd.read_csv('..\\data\\testdata.csv')
labels = pd.read_csv('..\\data\\trainlabel.txt', header=None, names=['label'])['label']
data.insert(0, 'income', labels)

# one-hot encoding
cat_col = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
num_col = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
encode_data = pd.get_dummies(data, columns=cat_col)
encode_data_test = pd.get_dummies(testData, columns=cat_col)
encode_data.drop(['education', 'fnlwgt'], axis=1, inplace=True)
encode_data_test.drop(['education', 'fnlwgt'], axis=1, inplace=True)

# nomalization
for col in num_col:
    encode_data[col] = (encode_data[col] - encode_data[col].mean()) / encode_data[col].std()
    encode_data_test[col] = (encode_data_test[col] - encode_data_test[col].mean()) / encode_data_test[col].std()

# make the col to the same
for col in encode_data.columns:
    if col not in encode_data.columns:
        encode_data_test.drop(col, axis=1, inplace=True)

modelChose = sys.argv[2]


if modelChose == 'help' or modelChose == 'Help' or modelChose == '--help' or modelChose == '-h':
    print('Usage: python main.py [model]')
    print('model: [RD, DT, SGD, LR, RF, GB]')
    sys.exit(0)
elif modelChose == 'RD':
    print('Random')
    pred_label = np.zeros(encode_data_test.shape[0])
    np.savetxt('Random.txt', pred_label, fmt='%d')
elif modelChose == 'SGD':
    print('Stochastic Gradient Descent')
    model = SGDClassifier()
    model.fit(encode_data.drop('income', axis=1), encode_data['income'])
    pred_label = model.predict(encode_data_test)
    np.savetxt('SGD.txt', pred_label, fmt='%d')
elif modelChose == 'DT':
    print('Decision Tree')
    model = DecisionTreeClassifier()
    model.fit(encode_data.drop('income', axis=1), encode_data['income'])
    pred_label = model.predict(encode_data_test)
    np.savetxt('DecisionTree.txt', pred_label, fmt='%d')
elif modelChose == 'LR':
    print('Logistic Regression')
    model = LogisticRegression()
    model.fit(encode_data.drop('income', axis=1), encode_data['income'])
    pred_label = model.predict(encode_data_test)
    np.savetxt('LogisticRegression.txt', pred_label, fmt='%d')
elif modelChose == 'RF':
    print('Random Forest')
    model = RandomForestClassifier()
    model.fit(encode_data.drop('income', axis=1), encode_data['income'])
    pred_label = model.predict(encode_data_test)
    np.savetxt('RandomForest.txt', pred_label, fmt='%d')
elif modelChose == 'GB':
    print('Gradient Boosting')
    model = GradientBoostingClassifier()
    model.fit(encode_data.drop('income', axis=1), encode_data['income'])
    pred_label = model.predict(encode_data_test)
    np.savetxt('GradientBoosting.txt', pred_label, fmt='%d')
else:
    print('No such model: ', modelChose)
    print('Usage: python main.py [model]')
    print('model: [RD, DT, SGD, LR, RF, GB]')
    sys.exit(0)