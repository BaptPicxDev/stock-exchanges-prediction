#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 07th of May 2020      #
# picard.baptiste@laposte.net       #
#                                   #
#####################################

# Imports 
# Usefull librairies
import json # Use .json file

# My modules 
from accessData import * 
from model import * 

class StockPricePrediction :
	def __init__(self, json_file, end_date) : 
		self.my_json = json.load(open(json_file))
		self.access_data = AccessData(self.my_json["STOCK_NAME"], self.my_json["SOURCE"], self.my_json["START_DATE"], end_date)
		self.model = Model(self.my_json["STOCK_NAME"], self.access_data.scaler, self.getAccessData().getTrainTestSet())

	def getAccessData(self) :
		return self.access_data

	def getModel(self) :
		return self.model
		