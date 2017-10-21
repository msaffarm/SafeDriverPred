import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle as pk
from sklearn.model_selection import GridSearchCV

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
UTIL_DIR = CURRENT_DIR + "../util/"
sys.path.append(UTIL_DIR)
from data_provider import DataProvider
MODEL_DIR = CURRENT_DIR + "/trainedModel"

IGNORE_FEATURES = ["id","target"]


def save_model(model,save_path,model_name=None):

	if not model_name:
		model_name = "nest={}-lr={}-maxd={}".format(model.n_estimators,
			model.learning_rate,model.max_depth)
	with open(save_path + model_name,'wb') as out:
		pk.dump(model,out)
	print("Model saved successfully in " + str(save_path)+str(model_name))



def xgb_cv(model):
	pass



def tune_hyperparams_scikit(model,X_train,y_train):
	"""
	Tuning parameters of model using scikit learn GridSearch
	"""

	params = [{"learning_rate":[0.1],
	"n_estimators":[10],
	"seed":[100],
	"max_depth":[3],
	"min_child_weight":[1]
	}]

	print("Running GridSearch")
	gscv = GridSearchCV(model,params,cv=3,n_jobs=-1,
		scoring="accuracy",verbose=3,error_score=0)
	gscv.fit(X_train.as_matrix(),y_train)

	# save CV results
	gscv_resutls = pd.DataFrame(gscv.cv_results_)
	gscv_resutls.to_csv("GridSearchRes.csv",index=False)


def create_xgbmodel(dp):

	# extract features to be used in classifier
	features = dp.get_all_features()
	for x in IGNORE_FEATURES:
		features.remove(x)
	print("Reading test/train data")
	X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features)

	# print(X_train.shape)
	# print("size = {} MB".format(sys.getsizeof(X_train)/1e+6))
	# xtrain_mat = X_train.as_matrix()
	# print(xtrain_mat.shape)
	# print(xtrain_mat.dtype)
	# print("size = {} MB".format(sys.getsizeof(xtrain_mat)/1e+6))


	n2p_ratio = len(y_train[y_train==0])/len(y_train[y_train==1])

	# create model
	xgb_model = xgb.XGBClassifier(learning_rate=0.1,min_child_weight=1,
		n_estimators=1000,silent=0,seed=100,max_depth=3,scale_pos_weight=n2p_ratio,
		nthread=8,objective='binary:logistic')

	# Dmat_train = xgb.DMatrix(X_train,label=y_train)
	# Dmat_test = xgb.DMatrix(X_test,label=y_test)

	# cv task
	tune_hyperparams_scikit(xgb_model,X_train,y_train)
	return

	# fit model
	print("Fitting model")
	trained_model = xgb_model.fit(X_train,y_train)

	#get predictions
	train_pred = trained_model.predict(X_train)
	test_pred = trained_model.predict(X_test)
	print(np.bincount(y_train))
	print(np.bincount(train_pred))

	# save model
	save_model(trained_model,MODEL_DIR)



def main():

	# read data
	dp = DataProvider()
	dp.read_data("train.csv")
	if os.path.exists("trainedModel"):
		os.path.makedirs(CURRENT_DIR + "trainedModel")
	create_xgbmodel(dp)
	


if __name__ == '__main__':
	main()
