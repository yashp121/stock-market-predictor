# Stock Market Predictor
### This project aimed towards building a Deep Sequential Neural Network that accurately predicts the closing prices of major stocks.Made using tensorflow, keras, pandas, and sklearn. Written on Google Colaboratory.

## What it's Supposed To Do
My model aims to predict the closing stock price of a company using the data from the previous 100 days prior to the specified day. The model does this for each day up until the current day.

## The Model
- A Long Short Term Memory (LSTM) neural network, which is a form of recurrent neural networks (RNN) that adjust their predictions based on data given to them.
  - This model consists of 4 layers: Two LSTM layers and Two Dense Layers
    -The Dense Layers use a linear activation function to apply to the input data
#### Why Sequential?
- Sequential Models are typically used when:
  - The model itself and every layer has only one input and output (In this case, its just the closing price)
  - You want a linear hierarchy
  - Dont need layer sharing
- This model is only looking at the closing prices of each stock, so the sequential model is best suited for this NN


## How it works

After calling the function and providing the stock code of the company you wish to analyze, the training data is collected from Yahoo finance API.

The training data:
Independent - A 3D numpy array with each each element containing the previous 100 days of closing price data prior to the day being predicted
Dependent - The value of that day's closing price


