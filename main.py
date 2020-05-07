#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 01th of May 2020      #
# picard.baptiste@laposte.net       #
#                                   #
# Price prediction of stocks.       #
# Using machine learning.           #
#####################################

# Imports 
# Usefull librairies
import json # Use .json file
import time # Time calculation
import joblib # Saving and loading models
import warnings # Avoid warnings
import datetime # get the date
import numpy as np # np.array structure
import pandas as pd # DataFrame structure
import seaborn as sns # Data visualization
import matplotlib.pyplot as plt # Data visualization

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Preprocessing librairies
from keras.callbacks import ReduceLROnPlateau # Learning rate reduction
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, max_error # score calculation methods
from sklearn.model_selection import train_test_split # Split data
from sklearn.preprocessing import MinMaxScaler # Scaler to shape data

# Models
import xgboost as xgb # X Gradient Boosting (most powerfull)
from keras.models import Sequential, model_from_json # Basis for neural networks model implementation
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten # Standard layers 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # Gradient Boosting, Random Forest

# Own modules
from environment import *
from accessData import *

# Environment
plt.close('all')
plt.style.use('seaborn')
warnings.filterwarnings('ignore') # Avoid warnings

# CONSTANTS
JSON_FILE = './data/config.json'
GRAFS = True
SUMMARY = True
TODAY = datetime.date.today()

if __name__ == "__main__" :
	print("Script starting : {}.".format(TODAY))
	start = time.time()
	# env = Environment(xgb, np, pd, pdw, sns)
	# env.getEnvironmentVersion()
	my_json = json.load(open(JSON_FILE))
	AD = AccessData(my_json["STOCK_NAME"], my_json["SOURCE"], my_json["START_DATE"], TODAY) 
	print("End of this script in {} seconds.".format(time.time() - start))
