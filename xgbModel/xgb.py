import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle as pk
from sklearn.model_selection import GridSearchCV
import datetime

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
UTIL_DIR = CURRENT_DIR + "../util/"
sys.path.append(UTIL_DIR)
from data_provider import DataProvider
from measurement import Measurement
MODEL_DIR = CURRENT_DIR + "trainedModels/"

IGNORE_FEATURES = ["id","target"]


def save_model(model,save_path,model_name=None,prefix=None):

	if not model_name:
		model_name = "nest={}-lr={}-maxd={}".format(model.n_estimators,
			model.learning_rate,model.max_depth)
	if prefix:
		model_name = prefix+model_name

	with open(save_path + model_name,'wb') as out:
		pk.dump(model,out)
	print("Model saved successfully in " + str(save_path)+str(model_name))



def tune_hyperparams_scikit(model,X_train,y_train):
	"""
	Tuning parameters of model using scikit learn GridSearch
	"""
	print("Started CV task at "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	
	# create gini scorer
	measurer = Measurement()
	gini_score = measurer.get_gini_scorer()

	params = [{"learning_rate":[0.05,0.1],
	"n_estimators":[10,100,200,400,800],
	"seed":[100],
	"max_depth":[3,4],
	"min_child_weight":[1]
	}]

	print("Running GridSearch")
	gscv = GridSearchCV(model,params,cv=5,n_jobs=-1,
		scoring=gini_score,verbose=3,error_score=0)
	gscv.fit(X_train,y_train)
	best_model = gscv.best_estimator_
	# save best model
	save_model(best_model,MODEL_DIR,prefix="CV5-bestModel-")
	# save CV results
	gscv_resutls = pd.DataFrame(gscv.cv_results_)
	gscv_resutls.to_csv("GridSearchRestest.csv",index=False)
	
	print("Finished CV task at "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))



def create_xgbmodel(dp):

	# extract features to be used in classifier
	features = dp.get_all_features()
	for x in IGNORE_FEATURES:
		features.remove(x)
	print("Reading test/train data")
	# X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features)

	X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features,
		method="resample")[0]

	saved_model = "CV5-bestModel-nest=200-lr=0.1-m3"
	if not os.path.exists(MODEL_DIR + saved_model):
		print("Training new model")
		n2p_ratio = len(y_train[y_train==0])/len(y_train[y_train==1])
		# create model
		xgb_model = xgb.XGBClassifier(learning_rate=0.05,min_child_weight=1,
			n_estimators=400,silent=1,seed=100,max_depth=3,scale_pos_weight=1,
			nthread=8,objective='binary:logistic')

		# Dmat_train = xgb.DMatrix(X_train,lsaved_modelabel=y_train)
		# Dmat_test = xgb.DMatrix(X_test,label=y_test)

		# cv task
		tune_hyperparams_scikit(xgb_model,X_train,y_train)
		return

		# fit model
		print("Fitting model")
		trained_model = xgb_model.fit(X_train,y_train)
		# save model
		save_model(trained_model,MODEL_DIR)

	# else load model
	else:
		print("Loading model")
		with open(MODEL_DIR + saved_model,'rb') as inp:
			trained_model = pk.load(inp)

	#get predictions
	print("Making predictions!")
	train_pred = trained_model.predict(X_train)
	train_pred_prob = trained_model.predict_proba(X_train)
	test_pred = trained_model.predict(X_test)
	test_pred_prob = trained_model.predict_proba(X_test)
	# print metrics
	measurer = Measurement()
	print("Train resutls:")
	measurer.print_measurements(y_train,train_pred,train_pred_prob[:,1])
	print("Test resutls:")
	measurer.print_measurements(y_test,test_pred,test_pred_prob[:,1])





def main():
	# read data
	dp = DataProvider()
	dp.read_data("train.csv")
	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	create_xgbmodel(dp)


if __name__ == '__main__':
	main()
