# Stock Market Predictor
### This project aimed towards building a Deep Sequential Neural Network that accurately predicts the closing prices of major stocks.Made using tensorflow, keras, pandas, and sklearn. Written on Google Colaboratory.

## What it's Supposed To Do
My model aims to predict the closing stock price of a company using the data from the previous 60 days prior to the specified day. The model does this for each day up until the current day.

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
Independent - A 3D numpy array with each each element containing the previous 60 days of closing price data prior to the day being predicted
Dependent - The value of that day's closing price

The training data is then processed and formatted to make it available to the NN.
The model is created, trained, and then tested to return an RMSE value that tells us how accurate its predictions were.


## Results

I began by testing it on a fairly stable yet upward-trending stock in Apple (AAPL). The returned RMSE came out to 1.2906, which was much more accurate than I was expecting:

![AAPL Result](https://github.com/yashp121/stock-market-predictor/blob/master/img/NN%20AAPL%20Result.png)

I then tested a more volatile stock in Telsa (TSLA), but to my surprise the model maintained a decent accuracy level:

The last step was to show the graph of the stock with its predictions:



Overall, the accuracy of the models was better than I expected and I am pleasantly surprised with how much I learned from this project.


