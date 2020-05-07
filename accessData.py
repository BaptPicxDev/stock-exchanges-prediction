#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 07th of May 2020      #
# picard.baptiste@laposte.net       #
#                                   #
#####################################

# Imports 
# Usefull librairies
import pandas_datareader as pdw # Recover data from web APIs

class AccessData :
	def __init__(self, stock, source, start_date, end_date) :
		self.stock = stock
		self.source = source
		self.start_date = start_date
		self.end_date = end_date
		self.df =  None 
		self.n_rows = None
		self.n_cols = None
		self.update() #Update the value of df, n_rows, n_cols

	def getStock(self) :
		return self.stock

	def getSource(self) :
		return self.source

	def getStartDate(self) :
		return self.start_date

	def getEndDate(self) :
		return self.end_date

	def getDF(self) :
		return self.df

	def setDF(self, value) :
		self.df = value

	def getNRows(self) :
		return self.n_rows

	def setNRows(self, value) :
		self.n_rows = value

	def getNCols(self) :
		return self.n_cols

	def setNCols(self, value) :
		self.n_cols = value

	def update(self) :
		df = pdw.DataReader(self.getStock(), data_source=self.getSource(), start=self.getStartDate(), end=self.getEndDate())
		self.setDF(df)
		n_rows, n_cols = df.shape
		self.setNRows(n_rows)
		self.setNCols(n_cols)

	def getNAValues(self) :
		df = self.getDF()
		na_values = df.isnull().sum().sum()
		object_cols = df.dtypes[df.dtypes == np.object].count()
		integer_cols = df.dtypes[df.dtypes == np.int64].count()
		float_cols = df.dtypes[df.dtypes == np.float64].count()
		print("The dataset is composed by : {} rows/columns.".format(df.shape))
		print("There are {} na values of the {} values.".format(na_values, n_rows * n_cols))
		print("This dataset is composed by {} object cols / {} int64 cols / {} float64 cols.".format(object_cols, integer_cols, float_cols))