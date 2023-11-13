import os
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt

from globals import TRAIN_PERIOD, VALIDATION_PERIOD, TEST_PERIOD

pd.set_option("display.max_columns", None)

final_df = pd.read_csv(os.path.join('Data', '2_NIFTY500_5y_WithIndicator.csv'), parse_dates=True, index_col=[0])


def process_train():
    pass


def process_validate():
    pass


def process_test():
    pass


def split_datasets():

    train_df = final_df.loc[final_df.index <= TRAIN_PERIOD]
    # train_df = process_train(final_df)
    train_df.to_csv(os.path.join("Data", "3_Train_5y.csv"))

    hold_out_df = final_df.loc[(final_df.index <= VALIDATION_PERIOD) & (final_df.index > TRAIN_PERIOD)]
    # hold_out_df = process_validate(final_df)
    hold_out_df.to_csv(os.path.join("Data", "3_HoldOut_5y.csv"))

    test_df = final_df.loc[(final_df.index <= TEST_PERIOD) & (final_df.index > VALIDATION_PERIOD)]
    # test_df = process_test(final_df)
    test_df.to_csv(os.path.join("Data", "3_Test_5y.csv"))