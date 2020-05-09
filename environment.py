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
import numpy
import pandas
import pandas_datareader 
import seaborn
import keras
import sklearn
import matplotlib
import tensorflow

class Environment :

	def getEnvironmentVersion(self) :# Get environment version.
	    print("Python version : {}".format(sys.version))
	    print("Numpy version : {}".format(numpy.__version__))
	    print("Pandas version : {}".format(pandas.__version__))
	    print("Pandas datareader version : {}".format(pandas_datareader.__version__))
	    print("Tensorflow version : {}".format(tensorflow.__version__))
	    print("Keras version : {}".format(keras.__version__))
	    print("Sklearn version : {}".format(sklearn.__version__))
	    print("Seaborn version : {}".format(seaborn.__version__))
	    print("Matplotlib version : {}".format(matplotlib.__version__))
