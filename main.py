
# Import the required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from rdkit.Chem import AllChem
from keras.layers import Dense, SimpleRNN,GRU
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
from rdkit.Chem import rdMolDescriptors
import time

# data
data = pd.read_csv('data.txt', sep='\t', encoding='utf-16')

#Normalize temperature and pressure
scaler = MinMaxScaler()
data[['T_1','T_2','TP_1','TP_2','TC_1','TC_2','pressure','temperature']] = scaler.fit_transform(data[['T_1','T_2','TP_1','TP_2','TC_1','TC_2','pressure','temperature']])

X_smiles_1 = data['smiles_1']# Compound 1
X_smiles_2 = data['smiles_2']# Compound 2
T_1 = data['T_1']
TC_1 = data['TC_1']
TP_1 = data['TP_1']
TV_1 = data['TV_1']
hot1 = data['one_hot1']
T_2 = data['T_2']
TC_2 = data['TC_2']
TP_2 = data['TP_2']
TV_2 = data['TV_2']
hot2 = data['one_hot2']
X_temp = data['temperature']
X_pressure = data['pressure']
comp1=data['comp1']#compl
comp2=data['comp2']#compv
# Converting SMILES code columns to molecular objects
#X_mol_1 = [Chem.MolFromSmiles(smiles) for smiles in data['smiles_1']]
#X_mol_2 = [Chem.MolFromSmiles(smiles) for smiles in data['smiles_2']]
#X_smiles_1 = pd.get_dummies(data['smiles_1'])
#X_smiles_2 = pd.get_dummies(data['smiles_2'])

hot1_encoded = pd.get_dummies(data['one_hot1'])
hot2_encoded = pd.get_dummies(data['one_hot2'])

data = pd.concat([data, hot1_encoded, hot2_encoded], axis=1)
data.drop(['one_hot1', 'one_hot2'], axis=1, inplace=True)
# print(type(hot1_encoded))
# print(len(hot1_encoded))
# print(type(hot2))
# Update X with new column
X = np.column_stack((hot1_encoded, hot2_encoded, T_1, T_2, TP_1, TP_2, TC_1, TC_2, X_temp, comp1))
y = data[['pressure','comp1']].values #Output changes according to predicted target
# Divide the training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle= False)
# Calculate the size of the training set
#train_size = int(0.8 * len(X))

kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Divide the training set and test set in order
#X_train, X_test = X[:train_size], X[train_size:]
#y_train, y_test = y[:train_size], y[train_size:]

# model
model = Sequential()

model.add(SimpleRNN(32, input_shape=(1,X_train.shape[1]),  activation='relu',return_sequences=True))
#model.add(Dropout(0.005))
model.add(SimpleRNN(16,  activation='relu',return_sequences=True))
#model.add(Dropout(0.02))
model.add(Dense(8, activation='relu'))
#model.add(Dropout(0.05))
model.add(Dense(2, activation='linear'))



# Output Model Architecture
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


start_time = time.time()
# Run your model training code here
#loss function:rmse
# def root_mean_squared_error(y_true, y_pred):
#     mse = tf.reduce_mean(tf.square(y_pred - y_true))
#
#     rmse = tf.sqrt(mse)
#     return rmse

#
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mse')
# training model
epochs=150
batch_size=8

#with tf.device('/GPU:1'):
X_train_reshaped = X_train.reshape(X_train.shape[0], 1,X_train.shape[1])
y_train_reshaped = y_train.reshape(y_train.shape[0], 1,y_train.shape[1])
print("shape of x_train:", y_train_reshaped.shape)


history = model.fit(X_train_reshaped, y_train_reshaped, batch_size=batch_size, epochs=epochs, validation_split=0.2)
#history = model.fit(X_train_reshaped,  y_train_reshaped, batch_size=batch_size, epochs=epochs)

# Record the model end time
end_time = time.time()
#  Calculate model runtime
execution_time = end_time - start_time
print("The model runtime was: {:.2f} 秒".format(execution_time))
# Projected results
y_pred = model.predict(X_test.reshape(X_test.shape[0], 1,X_test.shape[1]))
y_pred = np.squeeze(y_pred, axis=1)

# Output evaluation indicators
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mae_0 = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
mae_1 = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
#r2 = r2_score(y_test[:, 1], y_pred[:, 1])
r2 = r2_score(y_test, y_pred)


print('epochs:',epochs)
print('batch_size:',batch_size)
print('MAE:', mae)
print('MAE0:', mae_0)
print('MAE1:', mae_1)
print('R²:', r2)
print('RMSE:', rmse)


# Draw a graph of the loss curve during training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


plt.scatter(y_test[:, 0], y_pred[:, 0], label='pressure/temperature', color='blue')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()

plt.scatter(y_test[:, 1], y_pred[:, 1], label='CompL/CompV', color='orange')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()

# Save predicted and actual values to an excel sheet
df = pd.DataFrame({'Actual Comp1': y_test[:, 0], 'Predicted Comp1': y_pred[:, 0], 'Actual Comp2': y_test[:, 1], 'Predicted Comp2': y_pred[:, 1]})
df.to_excel('predictions.xlsx', index=False)
