import numpy as np 
import pandas as pd
import sklearn.decomposition
import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sb 

np.seterr(divide='ignore',invalid='ignore')

stocks = pd.read_csv('supercolumns-elements-nasdaq-nyse-otcbb-general-UPDATE-2017-03-01.csv')

print(stocks.head())

str_list = []
for colname,colvalue in stocks.iteritems():
	if type(colvalue[1]) == str:
		str_list.append(colname)

num_list = stocks.columns.difference(str_list)
stocks_num = stocks[num_list]
print(stocks_num.head())
