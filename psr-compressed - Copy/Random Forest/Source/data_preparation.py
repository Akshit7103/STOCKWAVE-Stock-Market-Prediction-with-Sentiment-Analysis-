import os
import talib as ta
import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option("display.max_columns", None)


def get_data(symbol: str, start: str = "2019-12-30", end: str = "2021-12-31", auto_adjust: bool = False) -> pd.DataFrame:
  """
    Fetches the historical data for given symbol

    Parameters
    ----------
    symbol: str
      Stock symbol to be fetched for historical date
    start: str
      Default: 2019-12-30
      Valid: "YYYY-MM-DD"
      Starting period of historical data for given stock symbol
    end: str
      Default: 2021-12-31
      Valid: "YYYY-MM-DD
      Ending period of historical data for given stock symbol
    auto_adjust: bool
      Default: False
      Adjust all OHLC automatically ?

    Returns
    -------
    pd.DataFrame
  """
  stock = yf.Ticker(symbol)
  # hist = stock.history(start = start, end = end, auto_adjust = auto_adjust)
  hist = stock.history(period="5y", auto_adjust = auto_adjust)
  hist["Symbol"] = symbol

  return hist


def get_nifty_500(NIFTY500_SYMBOLS: pd.Series, start: str = "2019-12-30", end: str = "2021-12-31", auto_adjust: bool = False) -> pd.DataFrame:
  """
    Fetches the data of Nifty 500 symbols from list

    Parameters
    ----------
    NIFTY_500_SYMBOLS: pd.Series
      Series of Nifty 500 symbols to fetch
    start: str
      Default: 2019-12-30
      Valid: "YYYY-MM-DD"
      Starting period of historical data for given stock symbol
    end: str
      Default: 2021-12-31
      Valid: "YYYY-MM-DD
      Ending period of historical data for given stock symbol
    auto_adjust: bool
      Default: False
      Adjust all OHLC automatically ?

    Returns
    -------
    pd.DataFrame
  """
  PER_STOCK_LIST = list()
  for SYM in NIFTY500_SYMBOLS:
    print(SYM, "Fetched")
    stock = get_data(SYM+".NS")
    PER_STOCK_LIST.append(stock)

  return pd.concat(PER_STOCK_LIST)


def download_csv(df: pd.DataFrame, file_name: str = "NIFTY500_DATA.csv", index: bool = True) -> None:
  """
    Downloads .csv file of given df

    Parameters
    ----------
    df: pd.DataFrame
      Dataframe which needs to be converted to .csv file
    file_name: str
      Default: "NIFTY500_DATA.csv"
      Filename of downloaded .csv file
    index: bool
      Default: True
      Whether you need index of dataframe in your .csv file or not ?
  """
  df.to_csv(file_name, index = index)
