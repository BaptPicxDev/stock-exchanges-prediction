#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 08th of May 2020      #
# picard.baptiste@laposte.net       #
#                                   #
# Provide machine/deep learning 	# 
# models.							#
#####################################

# Imports
# Librairies for building deep learning model
from keras.models import Sequential # Basis for neural networks model implementation
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten # Standard layers 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # Gradient Boosting, Random Forest
import xgboost as xgb # X Gradient Boosting (most powerfull)

def getModelMLP(input_shape) : # Develop the model for close price prediction. Dense and Dropout
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu')) 
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(16, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(1)) 
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def getModelConv1D(input_shape) : # Develop the model for close price prediction. Dense, Conv1D, MaxPolling, Dropout, Flatten.
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv1D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=4, padding='same')) 
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=4, padding='same')) 
    model.add(Flatten()) # Don't forget this line !!!!
    model.add(Dropout(0.25)) 
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(8, activation='relu')) 
    model.add(Dropout(0.25))
    model.add(Dense(1)) 
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def getModelLSTM(input_shape) : # Develop the model for close price prediction. Dense, LSTM, Dropout
    model = Sequential()
    model.add(LSTM(40, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(40, return_sequences=True, activation='relu')) # , input_shape=input_shape))
    model.add(LSTM(8, return_sequences=False, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(8, activation='relu')) 
    model.add(Dropout(0.25))
    model.add(Dense(1)) 
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def getModelRandomForest() :
    return RandomForestRegressor(n_estimators=400, max_depth=12, ccp_alpha=0.3)

def getModelXGB() :
    return xgb.XGBRegressor(n_estimators=400, max_depth=12, ccp_alpha=0.3, learning_rate=0.1)

def getModelGBR() :
    return GradientBoostingRegressor(n_estimators=400, max_depth=12, ccp_alpha=0.3, learning_rate=0.1)