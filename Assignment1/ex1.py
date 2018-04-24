import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/markklibanov/PycharmProjects/AndrewNgProblemSet/Assignment1/ex1data1.txt'

data = pd.read_csv(path, header=None, names=['Population','Profit'])

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show(block=True)

data.T




