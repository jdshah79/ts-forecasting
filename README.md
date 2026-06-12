Enterprise Demand Forecasting Dashboard

Forecasting dashboard which can be used to compare historical data of sales and the forecasting. Identify anamolies where there is high difference between the expected sales and the unit sold. 

A sample data set has been included from https://www.kaggle.com/datasets/raminhuseyn/demand-forecasting-dataset The data contains demand_forecasting with the following set of information.

This data is available in data folder this has been used to create a Dashboard which is dynamic. This will allow user to select the target regions and data limit in terms of days.

The anamoly is displayed accordingly

To run the app, need to install the requirement packages and then

uvicorn app_mod:app --reload --port 8000

There are three app files app.py, app_mod, and app_old each with its own unique version of files
