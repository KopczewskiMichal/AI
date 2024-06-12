import pandas as pd
import numpy as np
import certifi
from datetime import timedelta, datetime
import ssl

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
ssl._create_default_https_context = ssl._create_unverified_context

import yfinance as yf

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365)

def get_sp500_companies() -> list[str]:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0] # Pierwsza tabela zawiera listę spółek
    return sp500_table['Symbol'].tolist()

def download_tickers_history(tickers: list[str]) -> None:
	data = []
	for company_ticker in tickers:
		company_data = yf.download(tickers=company_ticker, start = START_DATE, end = END_DATE, interval = "1d")
		company_data['date'] = company_data.index
		company_data['ticker'] = company_ticker
		company_data['Adj Close'] = company_data['Adj Close']
		data.append(company_data)
	
	raw_data = pd.concat(data)
	raw_data = raw_data[['Open', 'High', 'Low', 'Close', 'Volume', 'date', 'ticker', 'Adj Close']]
	raw_data.to_csv("resources/new_data.csv")

def download_SP500():
  raw_data = yf.download (tickers = "^GSPC", start = START_DATE,
                              end = END_DATE, interval = "5m")
  raw_data.to_csv("resources/SP500_history.csv")
  
  
if __name__ == "__main__":
  download_tickers_history(get_sp500_companies())
  # download_SP500()
  