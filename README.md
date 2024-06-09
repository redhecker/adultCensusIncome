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
├─report.pdf
├─prediction.txt
├─requirements.txt
├─.gitignore
└─README
```

## How to build
you need to follow the steps to build this repository:
first, clone the repository
```
git clone https://github.com/redhecker/adultCensusIncome.git
```
then, create a new virtual environment and install the requirements
```
conda create -n test python==3.10
pip install -r requirements.txt
```
finally, you can run the main prrogram
```
python main.py
```
you can generate the answer with the model you like, and you can also run `visualization.ipynb` to see the distribution of the data and run `dataProcessing.ipynb` to see the performance of the models.

