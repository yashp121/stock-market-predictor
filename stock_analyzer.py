import math
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def stock_analysis(stock_name):
    plt.style.use("fivethirtyeight")

    #Grabs the Apple stock quote
    df = web.DataReader(stock_name, data_source="yahoo", start="2012-01-01", end="2020-07-11")

    #Closing price history visualization
    plt.figure(figsize = (16,8))
    plt.title("Closing Price History")
    plt.plot(df["Close"])
    plt.xlabel("Date", fontsize = 18)
    plt.ylabel("Closing Price (USD $)", fontsize = 18)
    plt.show()

    #Creating a new data frame with only the Closing price column
    data = df.filter(["Close"])

    #Converting the dataframe into a numpy array
    dataset = data.values

    #Get the num of rows we want to train on (90%)
    training_data_len = math.ceil(len(dataset) * .90)


    data_scaler = MinMaxScaler(feature_range = (0,1)) # Creates scaler object for later use
    scaled_data = data_scaler.fit_transform(dataset) #Scaling the Data

    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
      x_train.append(train_data[i - 60:i, 0])   #60 previous days
      y_train.append(train_data[i, 0])          #61st day is predicted

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Building the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss = "mean_squared_error")

    model.fit(x_train, y_train, batch_size = 1, epochs = 1) ##Model training

    #Create test data
    #New array of scaled values from index 1791 to 1891
    test_data = scaled_data[training_data_len - 100: , : ]

    x_test = []
    y_test = dataset[training_data_len:, :]  ## actual values we want our model to predict

    for i in range(100, len(test_data)):
      x_test.append(test_data[i - 100: i, 0])

    x_test = np.array(x_test)

    #Data reshape after making it numpy array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = data_scaler.inverse_transform(predictions)

    #Get RMSE value for accuracy
    RMSE = np.sqrt(np.mean(predictions - y_test) ** 2)
    print("Outputted RMSE value: " + str(RMSE))

    #Data plotting
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid["Predictions"] = predictions

    plt.figure(figsize = (16,8))
    plt.title("Model")
    plt.xlabel("Date", fontsize = 18)
    plt.ylabel("Closing Price (USD $)", fontsize = 18)
    plt.plot(train["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Train", "Val", "Predictions"], loc = "lower left")
    plt.show()

