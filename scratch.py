import pandas as pd
import numpy as np


def main():
	d = pd.DataFrame([[1,2,3],[-1,4,3]],columns=['a','b','c'])
	print(d)
	# r = dict(a={-1:np.nan},b={-1:np.nan},c={-1:np.nan})
	d.replace(-1,np.nan,inplace=True)
	print(d)


if __name__ == '__main__':
	main()