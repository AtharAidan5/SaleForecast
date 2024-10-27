# SaleForecast

!pip install dataprep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from google.colab import drive

drive.mount('/content/drive/')

DATA_FILES = [
    'CodeChallenge_Dataset_2021-2023_Set_1.csv',
    'CodeChallenge_Dataset_2021-2023_Set_2.csv'
]

def load_and_preprocess_data(file_name):
    data = pd.read_csv(f'/content/drive/My Drive/athar/data-sales/{file_name}', sep=',')
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', inplace=True)
    data.fillna(method='ffill', inplace=True)
    df_grouped = data.groupby('Date').agg({'Quantity': 'sum', 'Amount': 'sum'}).reset_index()
    max_quantity = df_grouped['Quantity'].quantile(0.95)
    df_grouped['Quantity'] = df_grouped['Quantity'].clip(upper=max_quantity)
    return df_grouped

def create_dataset(data, time_step=1):
    X, Y_quantity, Y_amount = [], [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y_quantity.append(data[i + time_step, 0])
        Y_amount.append(data[i + time_step, 1])
    return np.array(X), np.array(Y_quantity), np.array(Y_amount)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(2))  # Output layer for both quantity and amount
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

train_dataset = load_and_preprocess_data(DATA_FILES[0])

train_data = train_dataset[train_dataset['Date'].dt.year < 2023]
val_data = train_dataset[train_dataset['Date'].dt.year == 2023]

scaler = RobustScaler()
scaled_train_data = scaler.fit_transform(train_data[['Quantity', 'Amount']])
scaled_val_data = scaler.transform(val_data[['Quantity', 'Amount']])

time_step = 3
X_train, Y_train_quantity, Y_train_amount = create_dataset(scaled_train_data, time_step)
X_val, Y_val_quantity, Y_val_amount = create_dataset(scaled_val_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

model = build_lstm_model((time_step, 1))
history = model.fit(X_train, np.column_stack((Y_train_quantity, Y_train_amount)), validation_data=(X_val, np.column_stack((Y_val_quantity, Y_val_amount))), epochs=100, batch_size=1, verbose=1)

train_predict = model.predict(X_train)
val_predict = model.predict(X_val)

train_predict = scaler.inverse_transform(train_predict)
val_predict = scaler.inverse_transform(val_predict)

plt.figure(figsize=(15, 6))
plt.plot(train_data['Date'], train_data['Quantity'], label='2021-2022 Data Quantity', color='blue')
plt.plot(val_data['Date'].iloc[time_step:], val_predict[:, 0], label='2023 Data Quantity', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Quantity Data')
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(train_data['Date'], train_data['Amount'], label='2021-2022 Data Amount', color='blue')
plt.plot(val_data['Date'].iloc[time_step:], val_predict[:, 1], label='2023 Data Amount', color='red')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.title('Amount Data')
plt.legend()
plt.show()

test_dataset = load_and_preprocess_data(DATA_FILES[1])

scaled_test_data = scaler.transform(test_dataset[['Quantity', 'Amount']])

X_test, Y_test_quantity, Y_test_amount = create_dataset(scaled_test_data, time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

test_predict = model.predict(X_test)

test_predict = scaler.inverse_transform(test_predict)

plt.figure(figsize=(15, 6))
plt.plot(test_dataset['Date'].iloc[time_step:], test_predict[:, 0], label='Test Predictions', color='green')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Test Quantity')
plt .legend()
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(test_dataset['Date'].iloc[time_step:], test_predict[:, 1], label='Test Predictions', color='green')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.title('Test Amount')
plt.legend()
plt.show()
