#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 03th of May 2020      #
# picard.baptiste@laposte.net       #
#                                   #
# Price prediction of stocks.       #
# Using machine learning.           #
#####################################

# Imports 
# Usefull modules
import os 
import sys
import json # Use .json file
import time # Time calculation
import joblib # Saving and loading models
import warnings # Avoid warnings
import datetime # get the date
import numpy as np # np.array structure
import pandas as pd # DataFrame structure
import seaborn as sns # Data visualization
import matplotlib
import matplotlib.pyplot as plt # Data visualization
import pandas_datareader as pdw # Recover data from web APIs

# Preprocessing modules
import keras
import sklearn
from keras.callbacks import ReduceLROnPlateau # Learning rate reduction
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, max_error # score calculation methods
from sklearn.model_selection import train_test_split # Split data
from sklearn.preprocessing import MinMaxScaler # Scaler to shape data

# Models
from keras.models import Sequential, model_from_json # Basis for neural networks model implementation
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten # Standard layers 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # Gradient Boosting, Random Forest
import xgboost as xgb # X Gradient Boosting (most powerfull)

# Environment
plt.close('all')
plt.style.use('seaborn')
warnings.filterwarnings('ignore') # Avoid warnings

# CONSTANTS
JSON_FILE = './data/config.json'
GRAFS = True
SUMMARY = True
TODAY = datetime.date.today()

def getEnvironmentVersion() :# Get environment version.
    print("Python version : {}".format(sys.version))
    print("Numpy version : {}".format(np.__version__))
    print("Pandas version : {}".format(pd.__version__))
    print("Keras version : {}".format(keras.__version__))
    print("Sklearn version : {}".format(sklearn.__version__))
    print("Seaborn version : {}".format(sns.__version__))
    print("Matplotlib version : {}".format(matplotlib.__version__))
    print("Datareader version : {}".format(pdw.__version__))

if __name__ == "__main__" :
	print("Script starting : {}.".format(datetime.datetime.now()))
	start = time.time()
	my_json = json.load(open(JSON_FILE))
	getEnvironmentVersion()
	print("End of this script in {} seconds.".format(time.time() - start))
