import quandl  # quandl to get stock market data
import pandas as pd  # pandas for processing csv files
import numpy as np  # numpy for processing table formed data
import fbprophet  # facebook data science library for python to predict stock market
import matplotlib.pyplot as plt  # matplotlib pyplot for plotting

class Stocker():
    
    def __init__(self, stock_name, exchange='WIKI'):
        
        self.stock_name = stock_name.upper() # Capitalize stock name
        
        quandl.ApiConfig.api_key = 'TcKtBtukcckAFsspxbkS' # API Key for quandl

        stock = quandl.get('%s/%s' % (exchange, stock_name)) # Fetch stock data from quandl
        
        # Set the index to a column called Date
        stock = stock.reset_index(level=0)
        
        # Columns required for prophet
        stock['ds'] = stock['Date']

        if ('Adj. Close' not in stock.columns):
            stock['Adj. Close'] = stock['Close']
            stock['Adj. Open'] = stock['Open']
        
        stock['y'] = stock['Adj. Close']
        stock['Daily Change'] = stock['Adj. Close'] - stock['Adj. Open']
        
        # Data assigned as class attribute
        self.stock = stock.copy()

        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05 
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None

    # Create a prophet model without training
    def create_model(self):

        # Make the model
        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,  
                                  weekly_seasonality=self.weekly_seasonality, 
                                  yearly_seasonality=self.yearly_seasonality,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                  changepoints=self.changepoints)
        
        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        
        return model
    
    def predict_future(self, days=30):
        
        # Use past self.training_years years for training
        train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=3))]
        
        model = self.create_model()
        
        model.fit(train)
        
        # Future dataframe with specified number of days to predict
        future = model.make_future_dataframe(periods=days, freq='D')
        future = model.predict(future)
        
        # Only concerned with future dates
        future = future[future['ds'] >= max(self.stock['Date'])]
        
        # Calculate whether increase or not
        future['diff'] = future['yhat'].diff()
    
        future = future.dropna() # Drop empty rows
        
        # Set up plot

        plt.plot(future['ds'],future['yhat'], label='Prediction')
        plt.xticks(rotation = '45') # Rotate dates by 45 degree for visibility
        plt.ylabel('Predicted Stock Price (US $)')
        plt.xlabel('Date')
        plt.title('Predictions for %s' % self.stock_name)
        plt.show()                                                             

def get_input_data():

    stock_name = input(
        "Enter stock name(Eg.AMZN for amazon, MSFT for Microsoft): ")

    number_of_days_to_predict = input(
        "Enter number of days to predict(Number of days is after 2018-04-10):")

    return stock_name, number_of_days_to_predict


def predict_stocks(stock_name, number_of_days_to_predict):

    stock_object = Stocker(stock_name)

    stock_object.predict_future(days=int(number_of_days_to_predict))


stock_name, number_of_days_to_predict = get_input_data()
predict_stocks(stock_name, number_of_days_to_predict)