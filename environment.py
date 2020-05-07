#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 07th of May 2020      #
# picard.baptiste@laposte.net       #
#                                   #
#####################################

# Imports 
# Usefull librairies
import sys
import keras
import sklearn
import matplotlib

class Environment :
	def __init__(self, xgb, pd, np, pdw, sns) :
		self.xgb = xgb
		self.pd = pd
		self.np = np
		self.pdw = pdw
		self.sns = sns

	def getEnvironmentVersion(self) :# Get environment version.
	    print("Python version : {}".format(sys.version))
	    print("Numpy version : {}".format(self.np.__version__))
	    print("Pandas version : {}".format(self.pd.__version__))
	    print("Pandas datareader version : {}".format(self.pdw.__version__))
	    print("Keras version : {}".format(keras.__version__))
	    print("Sklearn version : {}".format(sklearn.__version__))
	    print("Seaborn version : {}".format(self.sns.__version__))
	    print("Matplotlib version : {}".format(matplotlib.__version__))
