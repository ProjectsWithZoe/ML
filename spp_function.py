import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
today = date.today()

d1 = today.strftime('%Y-%m-%d')
end_date = d1

d2 = today - timedelta(days=5000) 
d2 = d2.strftime('%Y-%m-%d')
start_date = d2 
best_stocks_today = ['TSLA', 'META', 'AMZN', 'NFLX', 'AMD']

data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
#print(data)

data['Date'] = data.index
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
data.reset_index(drop=True, inplace=True)

#print(data.head())
#print(data.tail())

import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
figure.update_layout(title='Stock Price analysis', xaxis_rangeslider_visible = False)
#figure.show()
#the close column is the target column

correlation = data.corr()
#print(correlation['Close'].sort_values(ascending=False))

#looks at the correlation of the close column to the other columns

X=data[['Open', 'High', 'Low', 'Volume']]
y= data['Close']

X= X.to_numpy()
y= y.to_numpy()
y = y.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)


#Neural network preparation
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])
model.fit(X_train, y_train, batch_size=30, epochs=10)

last_day_data = data.iloc[-1][['Open', 'High', 'Low', 'Volume']].values
last_day_data = last_day_data.astype('float32')
last_day_data = last_day_data.reshape((1,4,1))

predicted_closing_value = model.predict(last_day_data)
#print(predicted_closing_value)
#print(data.iloc[-1]['Close'])
#print(data.iloc[-1]['Date'])

print(f"Date is {data.iloc[-1]['Date']} \n Predicted closing price : {predicted_closing_value} \n Actual closing price : {data.iloc[-1]['Close']}")
#print(type(last_day_data))

#predicted_closing_values = []
#actual_date = data['Date'].iloc[-1]

#for i in range(3):  # Predict for the next 3 days
   # predicted_closing_value = model.predict(last_day_data)
    #predicted_closing_values.append(predicted_closing_value[0][0])

   # actual_date += timedelta(days=0)

    # Update last_day_data to include the predicted value for the next day
    #last_day_data = np.array([predicted_closing_value], dtype='float32').reshape(1, 1, 1)

#for i, value in enumerate(predicted_closing_values, start=1):
 #   next_day = (actual_date + timedelta(days=i)).strftime('%Y-%m-%d')
#  print(f'Predicted closing for {next_day} : {value}')


#    predicted_closing_values.append(predicted_closing_value)
#print(predicted_closing_value)
#print(data.iloc[-1]['Close'])

#predicted_closing_values.append(predicted_closing_value[0][0])

#print(last_day_data)
#CODE NOTES
#When drop is set to True, it means that the current index of the DataFrame will be removed and not added as a new column in the DataFrame.
#When inplace is set to True, it means that the operation is performed directly on the data DataFrame or Series, and no new DataFrame or Series is created.

#to_numpy was used instead of np.array because the data is already in pandas format aka series/dataframe
# y is reshaped to convert it into a 2D array with one column

#(X_train.shape[1], 1) essentially represents a 2D shape where the first dimension is determined by the number of features, and the second dimension is fixed at 1, indicating that you're working with single time step sequences. This shape is often used when defining the input shape for recurrent neural networks (RNNs), such as LSTMs, for tasks like time series prediction or sequence classification.

#The choice of evaluation metric for a Long Short-Term Memory (LSTM) network, or any neural network, depends on the specific problem you are trying to solve. Different metrics are suitable for different tasks. Here are some commonly used metrics for various types of problems when using LSTMs:
#Regression Problems (e.g., Time Series Prediction):
#Mean Squared Error (MSE): This metric measures the average squared difference between predicted and actual values. It's commonly used when you want to minimize the error between predicted and actual values.
#Root Mean Squared Error (RMSE): RMSE is the square root of the MSE. It provides an interpretable measure of the error in the same units as the target variable.
#Mean Absolute Error (MAE): MAE measures the average absolute difference between predicted and actual values. It's robust to outliers and provides a more interpretable error measure compared to MSE.

#.reshape(1, 1, 1): It reshapes the NumPy array into a 3D array with dimensions (1, 1, 1). This reshaping is typically done to match the expected input shape of your LSTM model. LSTM models often expect input data in the form of sequences, and a shape of (batch_size, timesteps, features) is common. In this case, you're reshaping it to a batch size of 1, one timestep, and one feature, which corresponds to a single prediction.