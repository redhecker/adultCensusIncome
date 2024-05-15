import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('..\\data\\traindata.csv')
labels = pd.read_csv('..\\data\\trainlabel.txt', header=None, names=['label'])['label']
data.insert(0, 'income', labels)
