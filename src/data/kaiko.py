# importing src directory
import sys
sys.path.append('..')
# library imports
import os
import requests
import csv
import pandas as pd
from datetime import datetime
# calbiration imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
# local imports
from api_key.my_api_key import api_key




def save_to_csv(data, filename):
    # Directory is set inside the function to maintain encapsulation
    directory = "/data/crypto_data/"
    # os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    file_path = os.path.join(directory, filename)  # Construct the full file path
    data.to_csv(file_path, index=False) 

def fetch_data(api_key, pair, start_dt, end_dt, frequency):
    """
    get aggregated vwap data from Kaiko API
    api_key: str, your Kaiko API key
    pairs: str, always 'asset-numeraire' format
    interval: str, the frequency -> suffixes are s (second), m (minute), h (hour) and d (day)
    start_dt: str, the start time in ISO 8601 format
    end_dt: str, the end time in ISO 8601 format
    exchange: str, exchange alias
    """
    filename = f'{pair}_{start_dt[0:10]}_{end_dt[0:10]}_{frequency}.csv' # define filename
    if os.path.exists(filename): # check if data already exists
        return pd.read_csv(filename) # read and return data
    url = f'https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/cbse/spot/{pair}/aggregations/vwap' # url for aggregated vwap (volume weighted average price)
    params = { 'start_time': start_dt, 'end_time': end_dt, 'interval': frequency, 'page_size': 100000, 'sort': 'asc'} # define request parameters
    headers = {'X-Api-Key': api_key,'Accept': 'application/json'} # define request headers 
    response = requests.get(url, headers=headers, params=params) # call kaiko request for data
    if response.status_code == 200: 
        data = pd.DataFrame(response.json()['data'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['price'] = data['price'].astype(float)
        # save_to_csv(data, filename) # save data to csv  # # NEED TO FIX # #
        return data # if successful return data
    else: raise Exception(f'Failed to retrieve data for {pair}: {response.status_code}, {response.text}') # if failed raise exception

def main():
    user_api_key = api_key
    pair = 'eth-usd'  # define pair (asset-numeraire format)
    start_dt = '2023-03-01T00:00:00Z' # define start date
    end_dt = '2024-03-01T00:00:00Z' # define end date
    frequency = '1d' # define frequency


if __name__ == '__main__':
    main()