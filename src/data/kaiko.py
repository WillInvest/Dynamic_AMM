# importing src directory
import sys
sys.path.append('..')
# library imports
import os
import requests
import pandas as pd
# calbiration imports
import pandas as pd
# local imports
from api_key.my_api_key import api_key




def save_to_csv(data, filename):
    # Directory is set inside the function to maintain encapsulation
    directory = "../data/crypto_data/"
    # os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    file_path = os.path.join(directory, filename)  # construct the full file path
    data.to_csv(file_path, index=False) 

def fetch_data(api_key, asset, start_dt, end_dt, frequency):
    """
    get aggregated vwap data from Kaiko API
    api_key: str, your Kaiko API key
    asset: str, asset get data for
    interval: str, the frequency -> suffixes are s (second), m (minute), h (hour) and d (day)
    start_dt: str, the start time in ISO 8601 format
    end_dt: str, the end time in ISO 8601 format
    exchange: str, exchange alias
    """
    filename = f'{asset}_{start_dt[0:10]}_{end_dt[0:10]}_{frequency}.csv' # define filename
    if os.path.exists(filename): # check if data already exists
        return pd.read_csv(filename) # read and return data

    url = f'https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/cbse/spot/{asset}-usd/aggregations/vwap' # url for aggregated vwap (volume weighted average price)
    params = { 'start_time': start_dt, 'end_time': end_dt, 'interval': frequency, 'page_size': 100000, 'sort': 'asc'} # define request parameters
    headers = {'X-Api-Key': api_key,'Accept': 'application/json'} # define request headers 
    response = requests.get(url, headers=headers, params=params) # call kaiko request for data
    if response.status_code == 200: 
        # convert json response to dataframe
        returned_df = pd.DataFrame(response.json()['data'])
        # create empty dataframe & convert timestamp to datetime & price to numeric
        data = pd.DataFrame()
        data['timestamp'] = pd.to_datetime(returned_df['timestamp'], unit='ms')
        data[f'{asset}_mrkt_price'] = returned_df['price'].astype(float)
        # save data to csv
        save_to_csv(data, filename) # save data to csv  # # NEED TO FIX # #
        return data # if successful return data
    else: raise Exception(f'Failed to retrieve data for {asset}: {response.status_code}, {response.text}') # if failed raise exception

def main():
    user_api_key = api_key
    # asset = 'eth'  # define pair (asset-numeraire format)
    # start_dt = '2023-03-01T00:00:00Z' # define start date
    # end_dt = '2024-03-01T00:00:00Z' # define end date
    # frequency = '1d' # define frequency


if __name__ == '__main__':
    main()