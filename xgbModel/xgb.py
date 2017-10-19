# import os
# import pandas as pd
# import xgboost as xgb

from sklearn import metrics
import numpy as np


def main():
	y_hat = np.array([1,1,2,1,2,2])
	y = np.array([1,2,1,1,2,1])
	fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=1)	
	print(fpr)
	print(tpr)
	print(thresholds)


if __name__ == '__main__':
	main()