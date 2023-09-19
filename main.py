import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta

today = date.today()

d1 = today.strftime('%Y-%m-%d')
end_date = d1

d2 = today - timedelta(days=5000) 
d2 = d2.strftime('%Y-%m-%d')
start_date = d2 

data = yf.download('AAPL', start=start_date, end=end_date, progress=False)
#print(data)

data['Date'] = data.index
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
data.reset_index(drop=True, inplace=True)
#print(data)
print(data.head())
print(data.tail())

#When drop is set to True, it means that the current index of the DataFrame will be removed and not added as a new column in the DataFrame.
#When inplace is set to True, it means that the operation is performed directly on the data DataFrame or Series, and no new DataFrame or Series is created.