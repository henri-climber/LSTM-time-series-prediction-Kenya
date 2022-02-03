import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 500)


def create_df_of_file(filename, measure, directory="Data", interval=None):
    dataframe = pd.read_csv(os.path.join(directory, filename))

    # set start of dataframe to time where it was in Kenya
    time = dataframe["date"].tolist()
    start = time.index("2015-04-21 11:00:00")
    dataframe = dataframe[start:]

    # linear interpolation to fill NA values
    dataframe = dataframe.interpolate("linear")
    # remove all non valid measure times (only measurements in 10minute steps are valid)
    dataframe = dataframe[dataframe["date"].str[-4:] == "0:00"]
    time = dataframe["date"].tolist()

    dataframe['date'] = pd.to_datetime(dataframe.pop('date'), format='%Y-%m-%d %H:%M:%S')
    dataframe.index = dataframe["date"]

    dataframe = dataframe.resample("3h").mean()

    # rename value column to actual measure category
    dataframe.rename(columns={"value": measure}, inplace=True)

    dataframe.reset_index(inplace=True, drop=True)

    return dataframe, time


def create_prec_df(time_df, station, interval=None):
    """
    This function is needed because the prec data had way more measurements than all the other data.
    """

    dataframe = pd.read_csv(os.path.join("Data", f"{station}-prec.csv"))
    time = dataframe["date"].tolist()
    start = time.index("2015-04-21 11:00:00")
    time = time[start:]
    dataframe = dataframe[start:]

    # linear interpolation to fill NA values
    dataframe = dataframe.interpolate("linear")

    dataframe.rename(columns={"value": "prec"}, inplace=True)

    dataframe.set_index('date', inplace=True)

    # remove all measurement times that aren't in the data of the time_df parameter
    to_remove = list(set(time).difference(set(time_df)))
    dataframe.drop(labels=to_remove, inplace=True, errors="ignore")

    dataframe.reset_index(inplace=True, drop=False)

    dataframe['date'] = pd.to_datetime(dataframe.pop('date'), format='%Y-%m-%d %H:%M:%S')
    dataframe.index = dataframe["date"]
    dataframe = dataframe.resample("3h").sum()

    dataframe.reset_index(inplace=True, drop=True)

    return dataframe


def create_filenames(station):
    categories = ["nit", "doc", "disch", "elc", "tcd", "toc", "tsp", "tur", "wl"]
    filenames = []

    for category in categories:
        filenames.append(f"{station}-{category}.csv")
    return filenames


def load_data(station, interval=None):
    filenames = create_filenames(station)

    frames = [create_df_of_file(file, file[4:7], interval=interval)[0] for file in filenames]

    _, df_time = create_df_of_file(f"{station}-nit.csv", "nit", interval=interval)
    prec_df = create_prec_df(df_time, station, interval=interval)

    frames.append(prec_df)

    df = pd.concat(frames, axis=1)

    return df


def create_tf_dataset(data, label, seq_length=3, batch_size=32):
    data = data[:-seq_length]
    label = label[seq_length:]

    ds = tf.keras.utils.timeseries_dataset_from_array(data, label, sequence_length=seq_length,
                                                      sequence_stride=1,
                                                      batch_size=batch_size, )

    return ds


def create_final_ds(station, label, interval=None, batch_size=32, seq_length=3):
    if os.path.exists(f"{station}-dataframe.pkl") and interval is None:
        print("loading normal pickle")
        df = pd.read_pickle(f"{station}-dataframe.pkl")

    elif interval is not None:
        if os.path.exists(f"{station}-dataframe-interval{interval}.pkl"):
            print("loading pickle")
            df = pd.read_pickle(f"{station}-dataframe-interval{interval}.pkl")
        else:
            print("creating Data")
            df = load_data(station, interval=interval)
            df.to_pickle(f"{station}-dataframe-interval{interval}.pkl")

    else:
        df = load_data(station)
        df.to_pickle(f"{station}-dataframe.pkl")

    # split data into train (70%), val (20%) and test (10%) data
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    feature_train = train_df.drop(["nit"], axis=1)
    feature_val = val_df.drop(["nit"], axis=1)
    feature_test = test_df.drop(["nit"], axis=1)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(feature_train.to_numpy())
    feature_train_scaled = feature_scaler.transform(feature_train)
    feature_val_scaled = feature_scaler.transform(feature_val)
    feature_test_scaled = feature_scaler.transform(feature_test)

    target_train = np.array(train_df["nit"], ndmin=2).T
    target_val = np.array(val_df["nit"], ndmin=2).T
    target_test = np.array(test_df["nit"], ndmin=2).T

    # creating tensorflow time series datasets
    train_ds_norm = create_tf_dataset(feature_train_scaled, target_train, batch_size=batch_size, seq_length=seq_length)
    val_ds_norm = create_tf_dataset(feature_val_scaled, target_val, batch_size=batch_size, seq_length=seq_length)
    test_ds_norm = create_tf_dataset(feature_test_scaled, target_test, batch_size=batch_size, seq_length=seq_length)

    # only the first three return values are needed for training
    return train_ds_norm, val_ds_norm, test_ds_norm, train_df
