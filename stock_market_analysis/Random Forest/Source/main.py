import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from globals import PRICE_CHANGE, PREDICT_PERIOD
from data_preparation import get_nifty_500
from feature_engineering import getPriceDifference, addAllIndicators
from model_building import pre_processing, build_model, predict

load_dotenv()
FEATURES=[
    "div_1_price_diff_values_SMA",
    # "std_21_SMA",
    # "std_7_SMA",
    # "21_SMA",
    # "7_SMA",
    # "ratio_21_SMA",
    # "ratio_7_SMA",

    "div_1_price_diff_values_EMA",
    "div_5_price_diff_values_EMA",
    "conv_5_price_diff_values_EMA",
    # "std_21_EMA",
    # "std_7_EMA",
    # "21_EMA",
    # "7_EMA",
    # "ratio_21_EMA",
    # "ratio_7_EMA",

    "div_1_price_diff_values_DEMA",
    "conv_5_price_diff_values_DEMA",
    # "std_21_DEMA",
    # "std_7_DEMA",
    # "21_DEMA",
    # "7_DEMA",
    # "ratio_21_DEMA",
    # "ratio_7_DEMA",

    "conv_1_price_diff_values_RSI",
    "div_1_price_diff_values_RSI",
    "div_5_price_diff_values_RSI",
    "conv_5_price_diff_values_RSI",
    # "rsi14_RSI",

    "moneyFlowRatio_MFI",
    "change_MFI",
    # "div_MFI",
    "momentum_MFI",
    "negMoneyFlow_MFI",
    "posMoneyFlow_MFI",
    # "typicalPrc_MFI",
    # "moneyFlow_MFI",
    "mfi14_MFI",
    # "movingAverage_MFI",

    "div_1_price_diff_values_MACD",
    "hist_MACD",
    "conv_5_price_diff_values_MACD",
    # "macd_MACD",
    "signal_MACD",
    "diff_MACD",

    "conv_1_price_diff_values_CCI",
    "trend_CCI",
    "conv_5_price_diff_values_CCI",
    # "14_CCI",
    
    "positive_momentum_1_price_diff_BBANDS",
    "positive_momentum_5_price_diff_BBANDS",
    # "lower_BBANDS",
    # "upper_BBANDS",
    
    # "10_ROC",
    "div_ROC",
    
    "%K_STOCH",
    "%D_STOCH",
    
    # "Open",
    # "High",
    # "Low",
    "Close",
    "Volume",
    "Symbol",
    "rolling_return",
    # "rolling_ret%"
    
    "trend_7_21_SMA",
    "div_SMA",
    "conv_SMA",

    "conv_EMA",
    "trend_721_EMA",
    "div_EMA",

    "conv_DEMA",
    "trend721_DEMA",
    "div_DEMA",

    "div_RSI",
    "overbought_RSI",
    "conv_RSI",
    "oversold_RSI",

    "crossOverAbove_MFI",
    "overbought_MFI",
    "oversold_MFI",

    "conv_MACD",
    "div_MACD",

    "div_CCI",
    "conv_CCI",

    "neg_momentum_BBANDS",
    "pos_momentum_BBANDS",

    "%KCross_STOCH",
    "%DOversold_STOCH",
    "%KOversold_STOCH",
    "%DOverbought_STOCH",
    "%KOverbought_STOCH",
]

# NIFTY500_SYMBOLS = pd.read_csv(os.path.join("../", "../", "Code", "Dataset", "NIFTY500_SYMBOLS.csv"))
# df = get_nifty_500(NIFTY500_SYMBOLS["Symbol"])

# df.to_csv(os.path.join("../", "Data", "1_NIFTY500_5y.csv"), index=False)
# df = pd.read_csv(os.path.join("../", "Data", "1_NIFTY500_5y.csv"))
# df_grpby = df.groupby('Symbol')
# targets = pd.DataFrame()

# # adding returns based on number of trading days
# for grp in df_grpby:
#     # swing, positional, business cylic
#     # _, grp[1]['1w'] = getPriceDifference(grp[1], 5)
#     # _, grp[1]['2w'] = getPriceDifference(grp[1], 10)
#     # _, grp[1]['3w'] = getPriceDifference(grp[1], 15)
#     # _, grp[1]['1m'] = getPriceDifference(grp[1], 20)
#     # _, grp[1]['5w'] = getPriceDifference(grp[1], 25)
#     # _, grp[1]['6w'] = getPriceDifference(grp[1], 30)
#     # _, grp[1]['7w'] = getPriceDifference(grp[1], 35)
#     # _, grp[1]['2m'] = getPriceDifference(grp[1], 40)
#     # _, grp[1]['9w'] = getPriceDifference(grp[1], 45)
#     # _, grp[1]['10w'] = getPriceDifference(grp[1], 50)
#     # _, grp[1]['11w'] = getPriceDifference(grp[1], 55)
#     # _, grp[1]['3m'] = getPriceDifference(grp[1], 60)
    
#     # # half-yearly
#     # _, grp[1]['6m'] = getPriceDifference(grp[1], 125)
    
#     # # annual rotators
#     # _, grp[1]['1y'] = getPriceDifference(grp[1], 252)
    
#     # # long term
#     # _, grp[1]['3y'] = getPriceDifference(grp[1], 756)
#     # _, grp[1]['5y'] = getPriceDifference(grp[1], 1260)

#     grp[1]["calc_fr_1_rw"], grp[1]["calc_fr_5_rw"] = getPriceDifference(grp[1], 5)
    
#     targets = pd.concat([targets, grp[1]])

# targets['rolling_return'] = np.where(targets['calc_fr_5_rw'] >= PRICE_CHANGE, 1, 0)
# print(f"Event rate for {PRICE_CHANGE}% price change: {(targets[targets['rolling_return'] == 1].shape[0] / targets.shape[0]) * 100}")

# grouped = targets.groupby('Symbol')
# final_df = pd.DataFrame()
# # traversing all groups in grouped dataframe
# start = time.time()
# for group in grouped:
#     # passing values at 1 index from tuple named group
#     addAllIndicators(group[1])
#     final_df = pd.concat([final_df, group[1]])
    
# print(time.time() - start)    

# final_df.to_csv(os.path.join('../', 'Data', '2_NIFTY500_5y_WithIndicator.csv'), index=False)

final_df = pd.read_csv(os.path.join('../', 'Data', '2_NIFTY500_5y_WithIndicator.csv'), index_col='Date', parse_dates=True)
# df = final_df.loc[final_df.index <= str(pd.to_datetime(PREDICT_PERIOD) - pd.tseries.offsets.BusinessDay(n = 1))]
# X, y = pre_processing(df)
# model = build_model(X, y)

to_pred_df = final_df.loc[final_df.index == PREDICT_PERIOD]
to_pred_df = to_pred_df[FEATURES]
to_pred_df = pre_processing(to_pred_df, prediction_set=True)
predict(to_pred_df)
