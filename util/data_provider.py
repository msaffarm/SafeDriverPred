import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
DATA_DIR = CURRENT_DIR + "../data/"

TEST_TRAIN_RATIO = 0.3
RANDOM_STATE = 100

class DataProvider(object):

	def __init__(self):

		self._data = None
		self._train_data = None
		self._train_label = None
		self._test_data = None
		self._test_label = None


	def clean_data(self,data):
		data.replace(-1,np.nan,inplace=True)


	def read_data(self,file):
		self._data =  pd.read_csv(DATA_DIR + file)
		self.clean_data(self._data)


	def get_data(self):
		return self._data


	def _create_test_train_data(self,features=None,target="target"):
		categorical_features = [x for x in features if "cat" in x or "bin" in x]
		if target in features:
			features.remove(target)

		X_vectorized = pd.get_dummies(self._data[features],
			columns=categorical_features,prefix_sep='_')
		y = self._data[[target]]
	
		self._train_data,self._test_data,self._train_label,self._test_label=\
		train_test_split(X_vectorized,y,test_size=TEST_TRAIN_RATIO,\
			random_state=RANDOM_STATE)

		self._train_label = np.ravel(self._train_label)
		self._test_label = np.ravel(self._test_label)


	def get_test_train_data(self,features=None,target="target"):
		if not self._train_data or not self._test_data:
			self._create_test_train_data(features=features,target=target)
		return self._train_data,self._test_data,self._train_label,self._test_label


def main():
	dp = DataProvider()
	dp.read_data("train.csv")
	data = dp.get_data()
	cols = list(data.columns.values)
	cols.remove("id")
	# print(cols)
	# dtypes = [data[x].dtype for x in cols]
	# print(dict(zip(cols,dtypes)))
	tr_data,tst_data,tr_label,tst_label = dp.get_test_train_data(features=cols)
	print(sorted(tr_data.columns.values))
	print(tr_data.shape)


if __name__ == '__main__':
	main()