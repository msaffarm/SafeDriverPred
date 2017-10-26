import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle as pk
from sklearn.model_selection import GridSearchCV
import datetime
from hyperopt import hp,STATUS_OK

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
UTIL_DIR = CURRENT_DIR + "../util/"
sys.path.append(UTIL_DIR)
from data_provider import DataProvider
from measurement import Measurement
from model import Model
from optimizer import Optimizer

MODEL_DIR = CURRENT_DIR + "trainedModels/"

IGNORE_FEATURES = ["id","target"]


class XGBClassifier(Model):

	def __init__(self):
		super().__init__()


	def get_train_test_Dmat(self):
		X_train,X_test,y_train,y_test = self._data
		train_dmat = xgb.DMatrix(X_train,label=y_train)
		test_dmat = xgb.DMatrix(X_test,label=y_test)
		return train_dmat, test_dmat


	def tune_hyperparams_scikit(self,model,X_train,y_train):
		"""
		Tuning parameters of model using scikit learn GridSearch
		"""
		print("Started CV task at "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		
		# create gini scorer
		measurer = Measurement()
		gini_score = measurer.get_gini_scorer()

		params = [{"learning_rate":[0.01,0.05],
		"n_estimators":[400,800,1000],
		"seed":[100],
		"max_depth":[2,3,4],
		"min_child_weight":[1,2,5],
		"subsample":[1,0.8]
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


	def make_predictions(self,trained_model,data):
		X_train,X_test,y_train,y_test = data
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


def create_xgbmodel(dp,xgb_model):
	X_train,X_test,y_train,y_test = get_data(dp)
	xgb_model.set_data((X_train,X_test,y_train,y_test))

	saved_model = "CV5-bestModel-nest=200-lr=0.1-m3"
	if not os.path.exists(MODEL_DIR + saved_model):
		print("Training new model")
		n2p_ratio = len(y_train[y_train==0])/len(y_train[y_train==1])
		# create model
		xgb_model_base = xgb.XGBClassifier(learning_rate=0.05,min_child_weight=1,
			n_estimators=400,silent=1,seed=100,max_depth=3,scale_pos_weight=1,
			nthread=8,objective='binary:logistic')
		xgb_model.set_base_model(xgb_model_base)
		# # cv task
		# xgb_model.tune_hyperparams_scikit(xgb_model_base,X_train,y_train)
		# return

		# fit model
		print("Fitting model")
		trained_model = xgb_model.get_base_model().fit(X_train,y_train)
		xgb_model.set_model(trained_model)
		# save model
		xgb_model.save_model(save_path=MODEL_DIR)

	# else load model
	else:
		train_model = xgb_model.load_model(save_path=MODEL_DIR,model_name=saved_model)
		xgb_model.set_model(trained_model)

	# make predictions
	xgb_model.make_predictions(xgb_model.get_model(),xgb_model.get_data())


def get_data(dp):
	# extract features to be used in classifier
	features = dp.get_all_features()
	for x in IGNORE_FEATURES:
		features.remove(x)
	print("Reading test/train data")
	# X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features)
	X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features,
		method="resample")[0]
	return (X_train,X_test,y_train,y_test)


def objective(params):
	xgb_model,measurer = params["xgb_model"],params["measurer"]
	# print(params)
	train_dmat,test_dmat = xgb_model.get_train_test_Dmat()
	trained_model = xgb.train(params,train_dmat,
		num_boost_round=int(params["n_estimators"]))
	test_pred = trained_model.predict(test_dmat)
	return {"loss":-measurer.normalized_gini(xgb_model.get_data()[1],test_pred),
	"status":STATUS_OK}


def tune_with_TPE(dp,xgb_model,opt):
	# add xgb_model,measurer
	X_train,X_test,y_train,y_test = get_data(dp)
	xgb_model.set_data(get_data(dp))

	# set optimizer parameters
	opt.set_max_eval(500)

	search_space = {"n_estimators": hp.quniform("n_estimators",100,1500,10),
	"eta": hp.quniform("eta",0.025,0.2,0.025),
	"max_depth": hp.choice("max_depth",list(range(2,8))),
	"min_child_weight": hp.quniform('min_child_weight',1,6,1),
	"subsample": hp.quniform("subsample",0.6,1.0,0.1),
	"colsample_bytree": hp.quniform("colsample_bytree",0.6,1.0,0.1),
	"gamma": hp.quniform("gamma",0,0.5,0.05),
	# "alpha": hp.choice("alpha",[1e-5,1e-2,1,10]),
	"nthread":8,
	"silent":1,
	"seed":100,
	"measurer": Measurement(),
	"xgb_model": xgb_model
	}

	opt.set_search_space(search_space)
	opt.set_objective(objective)
	# start optimization
	opt.optimize()
	opt.save_trials(CURRENT_DIR)


def main():
	# read data
	dp = DataProvider()
	xgb_model = XGBClassifier()
	dp.read_data("train.csv")
	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	# create_xgbmodel(dp,xgb_model)
	opt = Optimizer()
	tune_with_TPE(dp,xgb_model,opt)


if __name__ == '__main__':
	main()
