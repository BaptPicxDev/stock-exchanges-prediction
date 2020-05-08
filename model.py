#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 07th of May 2020      #
# picard.baptiste@laposte.net       #
#                                   #
#####################################

# Imports 
# Usefull librairies
import time
import numpy as np 

# Librairies for building machine learning models and process
from keras.callbacks import ReduceLROnPlateau # Learning rate reduction during training
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, max_error # Evaluation

# My modules 
from machine_learning import * # models

class Model :
	def __init__(self, scaler, train_test_data) :
		self.scaler = scaler
		self.X_train = train_test_data[0]
		self.y_train = train_test_data[1]
		self.X_test = train_test_data[2]
		self.y_test = train_test_data[3]
		self.n_rows_train = train_test_data[0].shape[0]
		self.n_rows_test = train_test_data[2].shape[0]
		self.n_cols = train_test_data[0].shape[1]
		self.input_shape_MLP = (self.n_cols,)
		self.input_shape_Conv1D = (self.n_cols, 1)
		self.input_shape_LSTM = (1, self.n_cols)
		self.models = self.updateModels()
		self.best_model = self.filterModels()

	# Getters / Setters
	def getScaler(self) :
		return self.scaler

	def getModels(self) :
		return self.models

	def getBestModel(self) :
		return self.best_model

	# Rows and columns
	def getNRowsTrain(self) : 
		return self.n_rows_train

	def getNRowsTest(self) : 
		return self.n_rows_test

	def getNCols(self) :
		return self.n_cols

	# X_train, y_train, X_test, y_test
	def getX_train(self) :
		return self.X_train

	def setX_train(self, value) :
		self.X_train = value

	def gety_train(self) :
		return self.y_train

	def sety_train(self, value) :
		self.y_train = value

	def getX_test(self) :
		return self.X_test

	def setX_test(self, value) :
		self.X_test = value

	def gety_test(self) :
		return self.y_test

	def sety_test(self, value) :
		self.y_test = value

	# Input shapes
	def getInputShapeMLP(self) :
		return self.input_shape_MLP

	def getInputShapeConv1D(self) :
		return self.input_shape_Conv1D

	def getInputShapeLSTM(self) :
		return self.input_shape_LSTM

	# Reshapes
	def reshapeForMLP(self) : # At the beginning, the X_train, y_train, X_test, y_test data are in this shape.
		self.setX_train(self.getX_train().reshape(self.getNRowsTrain(), self.getNCols()))
		self.setX_test(self.getX_test().reshape(self.getNRowsTest(), self.getNCols()))
		self.sety_train(self.gety_train().reshape(self.getNRowsTrain()))
		self.sety_test(self.gety_test().reshape(self.getNRowsTest()))

	def reshapeForConv1D(self) : 
		self.setX_train(self.getX_train().reshape(self.getNRowsTrain(), self.getNCols(), 1))
		self.setX_test(self.getX_test().reshape(self.getNRowsTest(), self.getNCols(), 1))
		self.sety_train(self.gety_train().reshape(self.getNRowsTrain(), 1))
		self.sety_test(self.gety_test().reshape(self.getNRowsTest(), 1))

	def reshapeForLSTM(self) : 
		self.setX_train(self.getX_train().reshape(self.getNRowsTrain(), 1, self.getNCols()))
		self.setX_test(self.getX_test().reshape(self.getNRowsTest(), 1, self.getNCols()))
		self.sety_train(self.gety_train().reshape(self.getNRowsTrain(), 1))
		self.sety_test(self.gety_test().reshape(self.getNRowsTest(), 1))

	def updateModels(self) : 
		# Principal function of this class.
		# The aim is to train several models and to determine which one is the best  
		# to predict the price of a stock.
		start_training = time.time()
		models =    [
				{"model_name" : 'LSTM', 'model' : getModelLSTM(self.getInputShapeLSTM()), 'history' : None, 'score' : None, 'training_time' : 0},
				{"model_name" : 'Conv1D', 'model' : getModelConv1D(self.getInputShapeConv1D()), 'history' : None, 'score' : None, 'training_time' : 0},
				{"model_name" : 'MLP', 'model' : getModelMLP(self.getInputShapeMLP()), 'history' : None, 'score' : None, 'training_time' : 0},
				{"model_name" : 'RandomForest', 'model' : getModelRandomForest(), 'history' : None, 'score' : None, 'training_time' : 0},
				{"model_name" : 'XGB', 'model' : getModelXGB(), 'history' : None, 'score' : None, 'training_time' : 0},
				{"model_name" : 'GradientBoost', 'model' : getModelGBR(), 'history' : None, 'score' : None, 'training_time' : 0}
			]
		for index, model in enumerate(models) : 
		    print("{}/{} -> Training {}".format(index+1, len(models), model['model_name']))
		    if(model['model_name'] == 'LSTM') :
		        self.reshapeForLSTM()
		    elif(model['model_name'] == 'Conv1D') :
		        self.reshapeForConv1D()
		    else : 
		    	self.reshapeForMLP()
		    start = time.time()
		    if(model['model_name'] == 'LSTM' or model['model_name'] == 'Conv1D' or model['model_name'] == 'MLP') :
		        learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
		                                                    patience=3, 
		                                                    verbose=1, 
		                                                    factor=0.2, 
		                                                    min_lr=0.0001)
		        model['model'].fit(self.getX_train(), self.gety_train(), batch_size=1, verbose=0, epochs=10, callbacks=[learning_rate_reduction])
		    else : 
		        model['model'] = model['model'].fit(self.getX_train(), self.gety_train())
		    model['training_time'] = (time.time() - start)/60
		    MSE_score = mean_squared_error(self.gety_test(), model['model'].predict(self.getX_test()))
		    MAE_score = mean_absolute_error(self.gety_test(), model['model'].predict(self.getX_test())) 
		    ME_score = max_error(self.gety_test(), model['model'].predict(self.getX_test()))
		    EA_score = explained_variance_score(self.gety_test(), model['model'].predict(self.getX_test()))
		    model['score'] = {"MSE" : MSE_score, "MAE" : MAE_score, "ME" : ME_score, "EA" : EA_score}
		    print("{}/{} -> {} trained in {} minutes.".format(index+1, len(models), model['model_name'], (time.time() - start)/60))
		print("It takes {} minutes to train {} models.".format((time.time() - start_training)/60, len(models)))
		return models

	def filterModels(self) :
		print('Filtering models and get the best one.')
		for model in self.getModels() :
			if(model['model_name'] == 'XGB') :
				print("The best model is {}.".format(model['model_name']))
				return model['model']

		