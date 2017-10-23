import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os, sys

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
UTIL_DIR = CURRENT_DIR + "../util/"
sys.path.append(UTIL_DIR)
from data_provider import DataProvider
MODEL_DIR = CURRENT_DIR + "trainedModels/"

IGNORE_FEATURES = ["id","target"]


def create_svr_model(dp):

	# extract features to be used in classifier
	features = dp.get_all_features()
	for x in IGNORE_FEATURES:
		features.remove(x)
	print("Loading test/train data...")
	# X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features)

	X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features,
		method="resample")[0]
	X_train = X_train.interpolate()
	X_test = X_test.interpolate()

	# Create regression model
	print("Creating SVR model...")
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	# svr_lin = SVR(kernel='linear', C=1e3)
	# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	
	# fit the regression model
	print("Fitting model...")
	trained_rbf = svr_rbf.fit(X_train, y_train)
	# TO DO: save model
	# TO DO: save_model(trained_model,MODEL_DIR)
	# y_lin = svr_lin.fit(X, y).predict(X)
	# y_poly = svr_poly.fit(X, y).predict(X)

	# make predictions
	print("Making predictions on train/test data...")
	train_pred = trained_rbf.predict(X_train)
	test_pred = trained_rbf.predict(X_test)


	# Look at the results
	print("Plotting the results...")
	lw = 2
	plt.scatter(X_train, y_train, color='darkorange', label='data')
	plt.plot(X_train, train_pred, color='navy', lw=lw, label='RBF model')

	# plt.plot(X_train, y_lin, color='c', lw=lw, label='Linear model')
	# plt.plot(X_train, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
	plt.xlabel('data')
	plt.ylabel('target')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

def main():
	# read data
	dp = DataProvider()
	dp.read_data("train.csv")
	# if not os.path.exists(MODEL_DIR)
		# os.makedirs(MODEL_DIR)
	create_svr_model(dp)

if __name__ == '__main__':
	main()