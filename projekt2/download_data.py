import pandas as pd
from datetime import timedelta, datetime
import ssl
import yfinance as yf
ssl._create_default_https_context = ssl._create_unverified_context

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365*3)

def get_sp500_companies() -> list[str]:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0] # Pierwsza tabela zawiera listę spółek
    return sp500_table['Symbol'].tolist()

def download_tickers_history(tickers: list[str]) -> None:
    company_data = yf.download(tickers=tickers, start = START_DATE, end = END_DATE, interval = "1d")["Close"]
    company_data = company_data.dropna(axis=1, how='all')
    print(company_data.head())
    company_data.to_csv("resources/stock_data.csv")


def download_sp500():
  df = yf.download (tickers="^GSPC", start=START_DATE,
                              end=END_DATE, interval="1d")
  df.columns = df.columns.get_level_values(0)
  df.reset_index(inplace=True)
  df.to_csv("resources/SP500_history.csv", index=False)
  
  
if __name__ == "__main__":
  # download_tickers_history(get_sp500_companies())
  download_sp500()
  