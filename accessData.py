#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 07th of May 2020      #
# picard.baptiste@laposte.net       #
#                                   #
#####################################

# Imports 
# Usefull librairies
import numpy as np
import pandas as pd 
import pandas_datareader as pdw # Recover data from web APIs
import seaborn as sns # Data visualization
import matplotlib.pyplot as plt # Data visualization

# Preprocessing librairies
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class AccessData :
	def __init__(self, stock, source, start_date, end_date, test_size=0.1) :
		self.stock = stock
		self.df =  pdw.DataReader(self.getStock(), data_source=source, start=start_date, end=end_date)
		self.test_size = test_size 
		self.scaler = None
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None
		self.create() #Update the value of df, n_rows, n_cols

	def getStock(self) :
		return self.stock

	def getTestSize(self) :
		return self.test_size

	def getDF(self) :
		return self.df

	def setDF(self, value) :
		self.df = value

	def getScaler(self) :
		return self.scaler

	def setScaler(self, value) :
		self.scaler = value

	def getTrainTestSet(self) :
		return self.X_train, self.y_train, self.X_test, self.y_test

	def setTrainTestSet(self, X_train, y_train, X_test, y_test) : 
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

	def create(self) :
		df = self.getDF()
		for column in df.columns :
			scaler = MinMaxScaler()
			df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
			if(column == 'Close') :
				self.setScaler(scaler)
		self.setDF(df)
		targets = df['Close'].values
		df = df.drop('Close', axis=1).values
		X_train, X_test, y_train, y_test = train_test_split(df, targets, test_size=self.getTestSize(), random_state=42)
		self.setTrainTestSet(X_train, y_train, X_test, y_test)

	def getNAValues(self) :
		df = self.getDF()
		na_values = df.isnull().sum().sum()
		object_cols = df.dtypes[df.dtypes == np.object].count()
		integer_cols = df.dtypes[df.dtypes == np.int64].count()
		float_cols = df.dtypes[df.dtypes == np.float64].count()
		print("The dataset is composed by : {} rows/columns.".format(df.shape))
		print("There are {} na values of the {} values.".format(na_values, df.shape[0] * df.shape[1]))
		print("This dataset is composed by {} object cols / {} int64 cols / {} float64 cols.".format(object_cols, integer_cols, float_cols))

	def showSampleDF(self) :
		print("Showing the dataFrame.")
		print(self.getDF()[:5])

	def showClosePriceEvolution(self) :
		df = self.getDF()
		plt.figure()
		plt.title("Visualization the closing price of "+self.getStock())
		plt.plot(df['Close'])
		plt.xlabel('Date')
		plt.ylabel('Closing price')
		plt.show()