The code fetches stock data from quandl.
Then trains a fb prophet model. This machine learning model is facebook data science model for predictions.
After the model has been trained prediction data is plotted using matplotlib.

Run the following code to run the program in linux environment(Ubuntu inside windows is not supported because gui for linux is not supported since graphs need gui, use either a virtual machine or actual linux, can also use jupyter notebook):

    sudo apt-get install python3-pip
    sudo pip3 install virtualenv 
    virtualenv venv 
    source venv/bin/activate
    pip3 install -r requirements.txt
    python3 stock_predictor.py
