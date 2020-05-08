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
import time # Time calculation
import warnings # Avoid warnings
import datetime # get the date
import matplotlib.pyplot as plt # Drawing figures

# Own modules
from stockPricePrediction import *

# Environment
plt.close('all')
plt.style.use('seaborn')
warnings.filterwarnings('ignore') # Avoid warnings
GRAFS = True
SUMMARY = True

# CONSTANTS
JSON_FILE = './data/config.json'
TODAY = datetime.date.today()

if __name__ == "__main__" :
	print("Script starting : {}.\n".format(TODAY))
	start = time.time()
	SPP =  StockPricePrediction(JSON_FILE, TODAY)
	print("End of this script in {} minutes.".format((time.time() - start)/60))
