import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)


def getPriceDifference(df: pd.DataFrame, prd: int):
    
    """
    Calculates price difference for past 1 row and past 5 rows 
    (i.e It will be price difference between past day and pass week if data is having 1-day interval)
    
    Params:
    df: pd.DataFrame
        Dataset in form of dataframe
    pdr: int
        Period to calculate returns
    
    Returns:
    Tuple of pd.Series
    """
    
    # shifted_open_1 = df["Open"].shift(-1)
    # shifted_open_prd = df["Open"].shift(-prd)
    
    # calc_fr_1_rw = ((df["High"] - shifted_open_1) / shifted_open_1) * 100
    # calc_fr_prd_rw = ((df["High"] - shifted_open_prd) / shifted_open_prd) * 100
    
    shifted_high_1 = df["High"].shift(-1)
    shifted_high_prd = df["High"].shift(-prd)
    
    # BullBro
    calc_fr_1_rw = ((shifted_high_1 - df["Low"]) / df["Low"]) * 100
    calc_fr_prd_rw = ((shifted_high_prd - df["Low"]) / df["Low"]) * 100
    
    # ShortStan - Doesn't work !
    # calc_fr_1_rw = ((df["Low"] - shifted_high_1) / shifted_high_1) * 100
    # calc_fr_prd_rw = ((df["Low"] - shifted_high_prd) / shifted_high_1) * 100

    return calc_fr_1_rw, calc_fr_prd_rw


def getRSI(df: pd.DataFrame, timeperiod: int = 14, bands: tuple = (30, 70)) -> pd.DataFrame:
    """
    MOMENTUM INDICATOR
    Calculates relative strength index indicator with features

    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 14 rows
        Time-periods in rows to consider while calculating RSI values
    bands: int, default = (30, 70)
        Bands representing rsi values to acknowledge oversold and overbought signals respectively
      
    Returns:
    pd.DataFrame of rsi, oversold, overbought & price difference in percentage
    """
    
    df_fn = pd.DataFrame(columns = ["rsi" + str(timeperiod) + "_RSI", "oversold_RSI", "overbought_RSI", "div_RSI", "conv_RSI", 
                                    "div_1_price_diff_values_RSI", "conv_1_price_diff_values_RSI",
                                    "div_5_price_diff_values_RSI", "conv_5_price_diff_values_RSI"],
                         index = df.index)
    
    rsi = pd.Series(ta.RSI(df["Close"], timeperiod = timeperiod))
    oversold = pd.Series(np.where(rsi <= bands[0], 1, 0)).values # upward momentum
    overbought = pd.Series(np.where(rsi >= bands[1], 1, 0)).values # downward momentum
    
    shifted_rsi = rsi.shift(1)
    div_RSI = pd.Series(np.where((rsi > bands[1]) & (shifted_rsi <=  bands[1]), 1, 0))
    conv_RSI = pd.Series(np.where((rsi < bands[0]) & (shifted_rsi >=  bands[0]), 1, 0))
    shifted_div_RSI = div_RSI.shift(1).values
    shifted_conv_RSI = conv_RSI.shift(1).values
    
    div_1_price_diff_values_RSI = pd.Series(np.where(shifted_div_RSI == 1, df["calc_fr_1_rw"], 0)).values
    conv_1_price_diff_values_RSI = pd.Series(np.where(shifted_conv_RSI == 1, df["calc_fr_1_rw"], 0)).values
    div_5_price_diff_values_RSI = pd.Series(np.where(shifted_div_RSI == 1, df["calc_fr_5_rw"], 0)).values
    conv_5_price_diff_values_RSI = pd.Series(np.where(shifted_conv_RSI == 1, df["calc_fr_5_rw"], 0)).values
    
    asgn_list = [rsi, oversold, overbought, div_RSI.values, conv_RSI.values,
                 div_1_price_diff_values_RSI, conv_1_price_diff_values_RSI,
                 div_5_price_diff_values_RSI, conv_5_price_diff_values_RSI]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
    
    return df_fn


def getSMA(df: pd.DataFrame, timeperiod: int = 7, other_sma: int = 21) -> pd.DataFrame:
    """
    TREND INDICATOR
    Calculates simple moving average indicator with features

    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 7 rows
        Time-periods in rows to consider while calculating SMA values
    other_sma: int, default = 21 rows
        Other SMA to calculate percentage price difference
    
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = [str(timeperiod) + "_SMA", str(other_sma) + "_SMA", "div_SMA", "conv_SMA", 
                                    "div_1_price_diff_values_SMA", "conv_1_price_diff_values_SMA",
                                    "div_5_price_diff_values_SMA", "conv_5_price_diff_values_SMA",
                                    "ratio_" + str(timeperiod) + "_SMA", "ratio_" + str(other_sma) + "_SMA",
                                    "std_" + str(timeperiod) + "_SMA", "std_" + str(other_sma) + "_SMA",
                                    "trend_" + str(timeperiod) + "_" + str(other_sma) + "_SMA"],
                         index = df.index)
    
    sma1 = pd.Series(ta.SMA(df["Close"], timeperiod = timeperiod))
    sma2 = pd.Series(ta.SMA(df["Close"], timeperiod = other_sma))
    
    shifted_sma1 = sma1.shift(1)
    shifted_sma2 = sma2.shift(1)
    div_SMA = pd.Series(np.where((sma1 < sma2) & (shifted_sma1 > shifted_sma2), 1, 0))
    conv_SMA = pd.Series(np.where((sma1 > sma2) & (shifted_sma1 < shifted_sma2), 1, 0))
    shifted_div_SMA = div_SMA.shift(1).values
    shifted_conv_SMA = conv_SMA.shift(1).values
    
    div_1_price_diff_values_SMA = pd.Series(np.where(shifted_div_SMA == 1, df["calc_fr_1_rw"], 0)).values
    conv_1_price_diff_values_SMA = pd.Series(np.where(shifted_conv_SMA == 1, df["calc_fr_1_rw"], 0)).values
    div_5_price_diff_values_SMA = pd.Series(np.where(shifted_div_SMA == 1, df["calc_fr_5_rw"], 0)).values
    conv_5_price_diff_values_SMA = pd.Series(np.where(shifted_conv_SMA == 1, df["calc_fr_5_rw"], 0)).values
    
    sma1_ratio = df["Close"] / sma1
    sma2_ratio = df["Close"] / sma2
    std1 = sma1.rolling(5).std()
    std2 = sma2.rolling(5).std()
    trend = pd.Series(np.where(sma1 > sma2, 1, 0)).values
    
    asgn_list = [sma1, sma2, div_SMA.values, conv_SMA.values,
                 div_1_price_diff_values_SMA, conv_1_price_diff_values_SMA,
                 div_5_price_diff_values_SMA, conv_5_price_diff_values_SMA,
                 sma1_ratio, sma2_ratio,
                 std1, std2,
                 trend]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
        
    return df_fn


def getEMA(df: pd.DataFrame, timeperiod: int = 7, other_ema: int = 21) -> pd.DataFrame:
    """
    TREND INDICATOR
    Calculates exponential moving average indicator with features

    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 7 rows
        Time-periods in rows to consider while calculating EMA values
    other_ema: int, default = 21 rows
        Other EMA to calculate percentage price difference
    
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = [str(timeperiod) + "_EMA", str(other_ema) + "_EMA", "div_EMA", "conv_EMA", 
                                    "div_1_price_diff_values_EMA", "conv_1_price_diff_values_EMA",
                                    "div_5_price_diff_values_EMA", "conv_5_price_diff_values_EMA",
                                    "ratio_" + str(timeperiod) + "_EMA", "ratio_" + str(other_ema) + "_EMA",
                                    "std_" + str(timeperiod) + "_EMA", "std_" + str(other_ema) + "_EMA",
                                    "trend_" + str(timeperiod) + str(other_ema) + "_EMA"],
                         index = df.index)
    
    ema1 = pd.Series(ta.EMA(df["Close"], timeperiod = timeperiod))
    ema2 = pd.Series(ta.EMA(df["Close"], timeperiod = other_ema))
    
    shifted_ema1 = ema1.shift(1)
    shifted_ema2 = ema2.shift(1)
    div_EMA = pd.Series(np.where((ema1 < ema2) & (shifted_ema1 > shifted_ema2), 1, 0))
    conv_EMA = pd.Series(np.where((ema1 > ema2) & (shifted_ema1 < shifted_ema2), 1, 0))
    shifted_div_EMA = div_EMA.shift(1).values
    shifted_conv_EMA = conv_EMA.shift(1).values
    
    div_1_price_diff_values_EMA = pd.Series(np.where(shifted_div_EMA == 1, df["calc_fr_1_rw"], 0)).values
    conv_1_price_diff_values_EMA = pd.Series(np.where(shifted_conv_EMA == 1, df["calc_fr_1_rw"], 0)).values
    div_5_price_diff_values_EMA = pd.Series(np.where(shifted_div_EMA == 1, df["calc_fr_5_rw"], 0)).values
    conv_5_price_diff_values_EMA = pd.Series(np.where(shifted_conv_EMA == 1, df["calc_fr_5_rw"], 0)).values
    
    ema1_ratio = df["Close"] / ema1
    ema2_ratio = df["Close"] / ema2
    std1 = ema1.rolling(5).std()
    std2 = ema2.rolling(5).std()
    trend = pd.Series(np.where(ema1 > ema2, 1, 0)).values
    
    asgn_list = [ema1, ema2, div_EMA.values, conv_EMA.values,
                 div_1_price_diff_values_EMA, conv_1_price_diff_values_EMA,
                 div_5_price_diff_values_EMA, conv_5_price_diff_values_EMA,
                 ema1_ratio, ema2_ratio,
                 std1, std2,
                 trend]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
        
    return df_fn


def getDEMA(df: pd.DataFrame, timeperiod: int = 7, other_dema: int = 21) -> pd.DataFrame:
    """
    TREND INDICATOR
    Calculates double exponential moving average indicator with features

    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 7 rows
        Time-periods in rows to consider while calculating DEMA values
    other_dema: int, default = 21 rows
        Other DEMA to calculate percentage price difference
    
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = [str(timeperiod) + "_DEMA", str(other_dema) + "_DEMA", "div_DEMA", "conv_DEMA", 
                                    "div_1_price_diff_values_DEMA", "conv_1_price_diff_values_DEMA",
                                    "div_5_price_diff_values_DEMA", "conv_5_price_diff_values_DEMA",
                                    "ratio_" + str(timeperiod) + "_DEMA", "ratio_" + str(other_dema) + "_DEMA",
                                    "std_" + str(timeperiod) + "_DEMA", "std_" + str(other_dema) + "_DEMA",
                                    "trend" + str(timeperiod) + str(other_dema) + "_DEMA"],
                         index = df.index)
    
    dema1 = pd.Series(ta.DEMA(df["Close"], timeperiod = timeperiod))
    dema2 = pd.Series(ta.DEMA(df["Close"], timeperiod = other_dema))
    
    shifted_dema1 = dema1.shift(1)
    shifted_dema2 = dema2.shift(1)
    div_DEMA = pd.Series(np.where((dema1 < dema2) & (shifted_dema1 > shifted_dema2), 1, 0))
    conv_DEMA = pd.Series(np.where((dema1 > dema2) & (shifted_dema1 < shifted_dema2), 1, 0))
    shifted_div_DEMA = div_DEMA.shift(1).values
    shifted_conv_DEMA = conv_DEMA.shift(1).values
    
    div_1_price_diff_values_DEMA = pd.Series(np.where(shifted_div_DEMA == 1, df["calc_fr_1_rw"], 0)).values
    conv_1_price_diff_values_DEMA = pd.Series(np.where(shifted_conv_DEMA == 1, df["calc_fr_1_rw"], 0)).values
    div_5_price_diff_values_DEMA = pd.Series(np.where(shifted_div_DEMA == 1, df["calc_fr_5_rw"], 0)).values
    conv_5_price_diff_values_DEMA = pd.Series(np.where(shifted_conv_DEMA == 1, df["calc_fr_5_rw"], 0)).values
    
    dema1_ratio = df["Close"] / dema1
    dema2_ratio = df["Close"] / dema2
    std1 = dema1.rolling(5).std()
    std2 = dema2.rolling(5).std()
    trend = pd.Series(np.where(dema1 > dema2, 1, 0)).values
    
    asgn_list = [dema1, dema2, div_DEMA.values, conv_DEMA.values,
                 div_1_price_diff_values_DEMA, conv_1_price_diff_values_DEMA,
                 div_5_price_diff_values_DEMA, conv_5_price_diff_values_DEMA,
                 dema1_ratio, dema2_ratio,
                 std1, std2,
                 trend]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
        
    return df_fn


def getBBANDS(df: pd.DataFrame, sma: int = 20) -> pd.DataFrame:
    """
    MOMENTUM INDICATOR
    Calculates bollinger bands ranging bands indicator with features
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    sma = int, default = 20 rows
        Time-periods to look back while calculating SMA values
        
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = ["lower_BBANDS", "upper_BBANDS", "pos_momentum_BBANDS", "neg_momentum_BBANDS",
                                    "positive_momentum_1_price_diff_BBANDS", "negative_momentum_1_price_diff_BBANDS",
                                    "positive_momentum_5_price_diff_BBANDS", "negative_momentum_5_price_diff_BBANDS"],
                         index = df.index)
    
    sma_val = df["Close"].rolling(sma).mean()
    lower = sma_val - 2 * (df["Close"].rolling(sma).std())
    upper = sma_val + 2 * (df["Close"].rolling(sma).std())
    
    pos_momentum = pd.Series(np.where(df["Close"] < lower, 1, 0))
    neg_momentum = pd.Series(np.where(df["Close"] > upper, 1, 0))
    shifted_pos_momentum = pos_momentum.shift(1).values
    shifted_neg_momentum = neg_momentum.shift(1).values
    
    positive_momentum_1_price_diff_BBANDS = pd.Series(np.where(shifted_pos_momentum == 1, df["calc_fr_1_rw"], 0)).values
    negative_momentum_1_price_diff_BBANDS = pd.Series(np.where(shifted_neg_momentum == 1, df["calc_fr_1_rw"], 0)).values
    positive_momentum_5_price_diff_BBANDS = pd.Series(np.where(shifted_pos_momentum == 1, df["calc_fr_5_rw"], 0)).values
    negative_momentum_5_price_diff_BBANDS = pd.Series(np.where(shifted_neg_momentum == 1, df["calc_fr_5_rw"], 0)).values
    
    asgn_list = [lower, upper, pos_momentum.values, neg_momentum.values,
                 positive_momentum_1_price_diff_BBANDS, negative_momentum_1_price_diff_BBANDS,
                 positive_momentum_5_price_diff_BBANDS, negative_momentum_5_price_diff_BBANDS]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
    
    return df_fn


def getCCI(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    TREND INDICATOR
    Calculates commodity channel index indicator with features
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 14 rows
        Time-periods in rows to consider while calculating CCI values
        
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = [str(timeperiod) + "_CCI", "div_CCI", "conv_CCI",
                                    "div_1_price_diff_values_CCI", "conv_1_price_diff_values_CCI",
                                    "div_5_price_diff_values_CCI", "conv_5_price_diff_values_CCI",
                                    "trend_CCI"],
                         index = df.index)
    
    cci = pd.Series(ta.CCI(df["High"], df["Low"], df["Close"], timeperiod = timeperiod))
    
    shifted_cci = cci.shift(1).values
    div_CCI = pd.Series(np.where((shifted_cci > 100) & (cci < 100), 1, 0))
    conv_CCI = pd.Series(np.where((shifted_cci < -100) & (cci > -100), 1, 0))
    shifted_div_CCI = div_CCI.shift(1).values
    shifted_conv_CCI = conv_CCI.shift(1).values

    div_1_price_diff_values_CCI = pd.Series(np.where(shifted_div_CCI == 1, df["calc_fr_1_rw"], 0)).values
    conv_1_price_diff_values_CCI = pd.Series(np.where(shifted_conv_CCI == 1, df["calc_fr_1_rw"], 0)).values
    div_5_price_diff_values_CCI = pd.Series(np.where(shifted_div_CCI == 1, df["calc_fr_5_rw"], 0)).values
    conv_5_price_diff_values_CCI = pd.Series(np.where(shifted_conv_CCI == 1, df["calc_fr_5_rw"], 0)).values
    
    trend = cci.diff().values
    
    asgn_list = [cci, div_CCI.values, conv_CCI.values,
                 div_1_price_diff_values_CCI, conv_1_price_diff_values_CCI,
                 div_5_price_diff_values_CCI, conv_5_price_diff_values_CCI,
                 trend]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
    
    return df_fn


def getMACD(df: pd.DataFrame, fast_ema: int = 12, slow_ema: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    MOMENTUM INDICATOR
    Calcualtes moving average convergence and divergence with features
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    fast_ema: int, default = 12 rows
        Fast EMA lookback period to calculate
    slow_ema: int, default = 26 rows
        Slow EMA lookback period to calculate
    signal_period: int, default = 9 rows
        EMA lookback period of macd line
    
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = ["macd_MACD", "signal_MACD", "hist_MACD", "diff_MACD", "div_MACD", "conv_MACD",
                                    "div_1_price_diff_values_MACD", "conv_1_price_diff_values_MACD",
                                    "div_5_price_diff_values_MACD", "conv_5_price_diff_values_MACD",],
                         index = df.index)
    
    macd, signal, hist = ta.MACD(df["Close"], fastperiod = fast_ema, slowperiod = slow_ema,
                                 signalperiod = signal_period)
    
    diff = abs(macd - signal)
    
    shifted_macd = macd.shift(1)
    shifted_signal = signal.shift(1)
    div_MACD = pd.Series(np.where((macd < signal) & (shifted_macd > shifted_signal), 1, 0))
    conv_MACD = pd.Series(np.where((macd > signal) & (shifted_macd < shifted_signal), 1, 0))
    shifted_div_MACD = div_MACD.shift(1).values
    shifted_conv_MACD = conv_MACD.shift(1).values
    
    div_1_price_diff_values_MACD = pd.Series(np.where(shifted_div_MACD == 1, df["calc_fr_1_rw"], 0)).values
    conv_1_price_diff_values_MACD = pd.Series(np.where(shifted_conv_MACD == 1, df["calc_fr_1_rw"], 0)).values
    div_5_price_diff_values_MACD = pd.Series(np.where(shifted_div_MACD == 1, df["calc_fr_5_rw"], 0)).values
    conv_5_price_diff_values_MACD = pd.Series(np.where(shifted_conv_MACD == 1, df["calc_fr_5_rw"], 0)).values
    
    asgn_list = [macd, signal, hist, diff, div_MACD.values, conv_MACD.values,
                div_1_price_diff_values_MACD, conv_1_price_diff_values_MACD,
                div_5_price_diff_values_MACD, conv_5_price_diff_values_MACD]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
    
    return df_fn


def getROC(df: pd.DataFrame, timeperiod: int = 10) -> pd.DataFrame:
    """
    MOMENTUM INDICATOR
    Calculates rate of change indicator with features
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 10 rows
        Lookback rows to calculate ROC values
        
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = [str(timeperiod) + "_ROC", "div_ROC", "trend_ROC"],
                         index = df.index)
    
    roc = pd.Series(ta.ROC(df["Close"], timeperiod = timeperiod))
    
    div_ROC = df["Close"] - roc
    
    shifted_roc = roc.shift(1)
    trend = np.sign(roc - shifted_roc)
    
    asgn_list = [roc, div_ROC, trend]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
    
    return df_fn


def getSTOCH(df: pd.DataFrame, sma_period: int = 14, sma_period_for_k: int = 3, bands: tuple = (20, 80)) -> pd.DataFrame:
    """
    MOMENTUM INDICATOR
    Adds STOCH indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    sma_period: int, default = 14 rows
        Lookback rows to calculate SMA values
    sma_period_for_k: int, default = 3 rows
        Lookback period to calculate moving average of k-line
    bands: tuple, default = (20, 80)
        Threshold range to consider oversold and overbought
        
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = ["%K_STOCH", "%D_STOCH", "%KCross_STOCH", "%KDiv_STOCH", "%DDiv_STOCH",
                                    "%KOversold_STOCH", "%KOverbought_STOCH", "%DOversold_STOCH", "%DOverbought_STOCH",
                                    "%KRoc_STOCH", "%DRoc_STOCH"],
                         index = df.index)
    
    sma = ta.SMA(df["Close"], timeperiod = sma_period)
    high = sma.max()
    low = sma.min()
    
    k = 100 * ((df["Close"] - low) / (high - low))
    d = k.rolling(window = sma_period_for_k).mean()

    shifted_k = k.shift(1)
    shifted_d = k.shift(1)
    pos_k_cross = pd.Series(np.where((k > d) & (shifted_k <= shifted_d), 1, 0)).values
    neg_k_cross = pd.Series(np.where((k < d) & (shifted_k >= shifted_d), -1, 0)).values
    
    shifted_close = df["Close"].shift(1)
    pos_k_div = pd.Series(np.where((k < df["Close"]) & (shifted_k >= shifted_close), 1, 0)).values
    neg_k_div = pd.Series(np.where((k > df["Close"]) & (shifted_k <= shifted_close), -1, 0)).values
    pos_d_div = pd.Series(np.where((d < df["Close"]) & (shifted_d >= shifted_close), 1, 0)).values
    neg_d_div = pd.Series(np.where((d > df["Close"]) & (shifted_d <= shifted_close), -1, 0)).values
    
    k_oversold = pd.Series(np.where(k < bands[0], 1, 0)).values
    k_overbought = pd.Series(np.where(k > bands[1], 1, 0)).values
    d_oversold = pd.Series(np.where(d < bands[0], 1, 0)).values
    d_overbought = pd.Series(np.where(d > bands[1], 1, 0)).values
    
    k_roc = k.pct_change()
    d_roc = d.pct_change()

    
    asgn_list = [k, d, pos_k_cross + neg_k_cross, pos_k_div + neg_k_div, pos_d_div + neg_d_div,
                k_oversold, k_overbought, d_oversold, d_overbought,
                k_roc, d_roc,
                ]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
    
    return df_fn


def getMFI(df: pd.DataFrame, timeperiod: int = 14, pct_change_period: int = 5, bands: tuple = (20, 80)) -> pd.DataFrame:
    """
    MOMENTUM INDICATOR
    Adds MFI indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 14 rows
        Lookback rows to calculate MFI values
    pct_change_period: int, default = 5 rows
        Rows to consider when calculating percentage change in MFI values
     bands: tuple, default = (20, 80)
        Threshold range to consider oversold and overbought
    
    Returns:
    pd.DataFrame with calculated features
    """
    
    df_fn = pd.DataFrame(columns = ["mfi" + str(timeperiod) + "_MFI", "typicalPrc_MFI", "moneyFlow_MFI", "posMoneyFlow_MFI", "negMoneyFlow_MFI",
                                   "moneyFlowRatio_MFI", "oversold_MFI", "overbought_MFI", "change_MFI", "div_MFI",
                                   "movingAverage_MFI", "crossOverAbove_MFI", "momentum_MFI"],
                         index = df.index)
    
    mfi = pd.Series(ta.MFI(df["High"], df["Low"], df["Close"], df["Volume"], timeperiod = timeperiod))
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    
    shifted_typical_price = typical_price.shift(1)
    money_flow = typical_price * df["Volume"]
    pos_money_flow = pd.Series(np.where(typical_price > shifted_typical_price, money_flow, 0))
    neg_money_flow = pd.Series(np.where(typical_price <= shifted_typical_price, money_flow, 0))
    money_flow_ratio = pos_money_flow.rolling(window = 14).sum().values / neg_money_flow.rolling(window = 14).sum().values
    
    oversold = pd.Series(np.where(mfi < bands[0], 1, 0)).values
    overbought = pd.Series(np.where(mfi > bands[1], 1, 0)).values
    change = mfi.pct_change(periods = pct_change_period)
    
    div = mfi - df["Close"]
    moving_avg = mfi.rolling(window = 5).mean()
    crossover_above = pd.Series(np.where(mfi > moving_avg, 1, 0)).values
    momentum = mfi.diff(periods = 5)
    
    asgn_list = [mfi, typical_price, money_flow, pos_money_flow.values, neg_money_flow.values,
                 money_flow_ratio, oversold, overbought, change, div,
                 moving_avg, crossover_above, momentum]
    
    for cols, vals in zip(list(df_fn.columns), asgn_list):
        df_fn[cols] = vals
    
    return df_fn



def addRSI(df: pd.DataFrame, timeperiod: int = 14, bands: tuple = (30, 70)) -> None:
    """
    Adds RSI indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 14 rows
        Time-periods in rows to consider while calculating RSI values
    bands: int, default = (30, 70)
        Bands representing rsi values to acknowledge oversold and overbought signals respectively
    """
    
    # get RSI related values from RSI indicator
    df_fn = getRSI(df, timeperiod, bands)
    # fill RSI related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
    
    # print("addRSI() finished ...")


def addSMA(df: pd.DataFrame, timeperiod: int = 7, other_sma: int = 21) -> None:
    """
    Adds SMA indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 7 rows
        Time-periods in rows to consider while calculating SMA values
    other_sma: int, default = 21 rows
        Other SMA to calculate percentage price difference
    """
    
    # get SMA related values from SMA indicator
    df_fn = getSMA(df, timeperiod, other_sma)
    # fill SMA related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
        
    # print("addSMA() finished ... ")


def addEMA(df: pd.DataFrame, timeperiod: int = 7, other_ema: int = 21) -> None:
    """
    Adds EMA indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 7 rows
        Time-periods in rows to consider while calculating EMA values
    other_dema: int, default = 21 rows
        Other EMA to calculate percentage price difference
    """
    
    # get EMA related values from EMA indicator
    df_fn = getEMA(df, timeperiod, other_ema)
    # fill EMA related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
        
    # print("addEMA() finished ... ")


def addDEMA(df: pd.DataFrame, timeperiod: int = 7, other_dema: int = 21) -> None:
    """
    Adds DEMA indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 7 rows
        Time-periods in rows to consider while calculating DEMA values
    other_dema: int, default = 21 rows
        Other DEMA to calculate percentage price difference
    """
    
    # get DEMA related values from DEMA indicator
    df_fn = getDEMA(df, timeperiod, other_dema)
    # fill DEMA related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
        
    # print("addDEMA() finished ... ")


def addBBANDS(df: pd.DataFrame, sma: int = 20) -> None:
    """
    Adds BBANDS indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    sma = int, default = 20 rows
        Time-periods to look back while calculating SMA values
    """
    
    # get BBANDS related values from BBANDS indicator
    df_fn = getBBANDS(df, sma)
    # fill BBANDS related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
    
    # print("addBBANDS() finished ... ")


def addCCI(df: pd.DataFrame, timeperiod: int = 14) -> None:
    """
    Adds CCI indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 14 rows
        Time-periods in rows to consider while calculating CCI values
    """
    
    # get CCI related values from CCI indicator
    df_fn = getCCI(df, timeperiod)
    # fill CCI related values in our original dataframe
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
    
    # print("addCCI() finished ... ")


def addMACD(df: pd.DataFrame, fast_ema: int = 12, slow_ema: int = 26, signal_period: int = 9) -> None:
    """
    Adds MACD indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    fast_ema: int, default = 12 rows
        Fast EMA lookback period to calculate
    slow_ema: int, default = 26 rows
        Slow EMA lookback period to calculate
    signal_period: int, default = 9 rows
        EMA lookback period of macd line
    """
    
    # get MACD related values from MACD indicator
    df_fn = getMACD(df, fast_ema, slow_ema, signal_period)
    # fill MACD related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
    
    # print("addMACD() finished ... ")


def addROC(df: pd.DataFrame, timeperiod: int = 10):
    """
    Adds ROC indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 10 rows
        Lookback rows to calculate ROC values
    """
    
    # get ROC related values from ROC indicator
    df_fn = getROC(df, timeperiod)
    # fill ROC related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
    
    # print("addROC() finished ... ")


def addSTOCH(df: pd.DataFrame, sma_period: int = 14, sma_period_for_k: int = 3, bands: tuple = (20, 80)) -> None:
    """
    Adds STOCH indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    sma_period: int, default = 14 rows
        Lookback rows to calculate SMA values
    sma_period_for_k: int, default = 3 rows
        Lookback period to calculate moving average of k-line
    bands: tuple, default = (20, 80)
        Threshold range to consider oversold and overbought
    """
    
    # get STOCH related values from STOCH indicator
    df_fn = getSTOCH(df, sma_period, sma_period_for_k, bands)
    # fill STOCH related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
    
    # print("addSTOCH() finished ... ")


def addMFI(df: pd.DataFrame, timeperiod: int = 14, pct_change_period: int = 5, bands: tuple = (20, 80)) -> None:
    """
    Adds MFI indicator based on index
    
    Params:
    df = pd.DataFrame
        Dataset in form of dataframe
    timeperiod: int, default = 14 rows
        Lookback rows to calculate MFI values
    pct_change_period: int, default = 5 rows
        Rows to calculate percentage change in MFI values
     bands: tuple, default = (20, 80)
        Threshold range to consider oversold and overbought    
    """
    
    # get MFI related values from MFI indicator
    df_fn = getMFI(df, timeperiod)
    # fill MFI related values in our original dataframe    
    for column in list(df_fn.columns):
        df.loc[df.index.isin(df_fn.index), column] = df_fn[column].values
    
    # print("addMFI() finished ... ")


def addAllIndicators(df):
    addRSI(df) # 9 features
    addSMA(df) # 11 features
    addEMA(df) # 11 features
    addDEMA(df) # 11 features
    addBBANDS(df) # 7 features
    addCCI(df) # 8 features
    addMACD(df) # 10 features
    addROC(df) # 3 features
    addMFI(df) # 13 features
    addSTOCH(df) # 13 features
    
    
    # TODO: Other technical indicators
    # addWILLR(df)
    # addWCLPRICE(df)
    # addOBV(df)


# volume
# candle length
# gap-ups
# sentiments