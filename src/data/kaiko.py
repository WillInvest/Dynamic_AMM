import os
import requests
import csv
from datetime import datetime
from api_key.my_api_key import api_key


def fetch_data(api_key, pair, start_dt, end_dt, frequency):
    """
    api_key: str, your Kaiko API key
    pairs: str, always 'asset-numeraire' format
    interval: str, the frequency -> suffixes are s (second), m (minute), h (hour) and d (day)
    start_dt: str, the start time in ISO 8601 format
    end_dt: str, the end time in ISO 8601 format
    exchange: str, exchange alias
    """
    # check if data already exists
    if os.path.exists(f'crypto_data/{pair}_{start_dt[0:10]}_{end_dt[0:10]}_{frequency}.csv'):
        #__TODO__
        pass
    # url for aggregated vwap (volume weighted average price)
    url = f'https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/cbse/spot/{pair}/aggregations/vwap'


    params = {
        'start_time': start_dt,
        'end_time': end_dt,
        'interval': frequency,
        'page_size': 100000,
        'sort': 'asc'
    }
    headers = {'X-Api-Key': api_key,
               'Accept': 'application/json'
    }
    # call request
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:        
        return response.json()['data']
    else:
        raise Exception(f'Failed to retrieve data for {pair}: {response.status_code}, {response.text}')


def save_to_csv(data, pair, filename):
    headers = ['pair', 'timestamp', 'price']
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in data:
            timestamp = datetime.fromtimestamp(row.get('timestamp') / 1000).isoformat() if row.get('timestamp') else ''
            writer.writerow([
                # pair.replace('-', '').lower(), 
                pair,
                timestamp,
                row.get('price', '')
            ])


def main():
    api_key = api_key
    pair = 'eth-usd'  # define pair (asset-numeraire format)
    start_dt = '2023-03-01T00:00:00Z'
    end_dt = '2024-03-01T00:00:00Z'
    frequency = '1d'
    filename = f'crypto_data/{pair}_{start_dt[0:10]}_{end_dt[0:10]}_{frequency}.csv'

    try:
        data = fetch_data(api_key, pair, start_dt, end_dt, frequency)


        if 'data' in data:
            save_to_csv(data, pair, filename)
            print(f'Data successfully saved to {filename}')
        else:
            print("No 'data' key in response.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()