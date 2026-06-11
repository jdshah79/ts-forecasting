Enterprise Demand Forecasting Dashboard

This is a Forecasting dashboard which can be used to compare historical data of sales and the forecasting. Identify anamolies where there is high difference between the expected sales and the unit sold. 

A sample data set has been included from https://www.kaggle.com/datasets/raminhuseyn/demand-forecasting-dataset
The data contains demand_forecasting with the following set of information.


The data provided has information according to various regions, hence the previous Dashboard was static has been modified to provide option for user to filter according to various regions. To start the app need to run 

uvicorn app_mod/app/app_mod1:app --reload --port 8000
