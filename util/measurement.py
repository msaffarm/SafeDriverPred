import os
import sys
import numpy as np
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score

class Measurement(object):

	def print_measurements(self,true_label,pred_label,pred_prob):
		print("ACCURACY= {}".format(self.accu(true_label,pred_label)))
		print("AUC= {}".format(self.AUC(true_label,pred_prob)))
		print("SKLEARN REPORT:")
		self.sklearn_report(true_label,pred_label)


	def AUC(self,true_label,pred_prob):
		return roc_auc_score(true_label, pred_prob)


	def accu(self,true_label,pred_label):
		return accuracy_score(true_label,pred_label)


	def sklearn_report(self,true_label,pred_label,
		classes=["negative(0)","positive(1)"]):
		print(classification_report(true_label,pred_label,target_names=classes))
