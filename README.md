# Adult Census Income Problem

## Introduction
### about the problem
A model on an Adult Census Income  dataset, and the goal is to predict whether income exceeds $50K/yr based on census  data.

### structure of the repository
```
.
├─data
│   ├─testdata.csv
│   ├─traindata.csv
│   └─trainlabel.txt
├─src
│   ├─visualization.ipynb
│   ├─dataProcessing.ipynb
│   └─main.py
├─report
├─test_result.txt
├─requirements.txt
├─.gitignore
└─README
```

## How to build
to build this, you need
```
pip install -r requirements.txt
python main.py
```
you can generate the answer with the model you like, and you can run `visualization.ipynb` to see the distribution of the data and run `dataProcessing.ipynb` to see the performance of the models.

