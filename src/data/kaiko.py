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
    if os.path.exists(f'crypto_data/{pair}_{start_dt[0:10]}_{end_dt[0:10]}_{frequency}.csv'): # check if data already exists
        data = pd.read_csv(f'crypto_data/{pair}_{start_dt[0:10]}_{end_dt[0:10]}_{frequency}.csv') # read data
        return data # return data
    url = f'https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/cbse/spot/{pair}/aggregations/vwap' # url for aggregated vwap (volume weighted average price)
    params = { 'start_time': start_dt, 'end_time': end_dt, 'interval': frequency, 'page_size': 100000, 'sort': 'asc'} # define request parameters
    headers = {'X-Api-Key': api_key,'Accept': 'application/json'} # define request headers 
    response = requests.get(url, headers=headers, params=params) # call kaiko request for data
    if response.status_code == 200: 
        save_to_csv(response.json()['data'], pair, f'crypto_data/{pair}_{start_dt[0:10]}_{end_dt[0:10]}_{frequency}.csv') # save data to csv
        return response.json()['data'] # if successful return data
    else: raise Exception(f'Failed to retrieve data for {pair}: {response.status_code}, {response.text}') # if failed raise exception

def save_to_csv(data, pair, filename):
    headers = ['pair', 'timestamp', 'price'] # define headers
    with open(filename, 'w', newline='') as file: # open file for writing
        writer = csv.writer(file) # create csv writer
        writer.writerow(headers) # write headers
        for row in data: # write data
            timestamp = datetime.fromtimestamp(row.get('timestamp') / 1000).isoformat() if row.get('timestamp') else '' # get timestamp
            writer.writerow([pair, timestamp, row.get('price', '')])  # write row
    print(f'Data successfully saved to {filename}') # print success message


def main():
    user_api_key = api_key
    pair = 'eth-usd'  # define pair (asset-numeraire format)
    start_dt = '2023-03-01T00:00:00Z' # define start date
    end_dt = '2024-03-01T00:00:00Z' # define end date
    frequency = '1d' # define frequency


if __name__ == '__main__':
    main()