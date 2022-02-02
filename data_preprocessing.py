import os

import numpy as np
import pandas as pd
import tensorflow as tf

pd.set_option('display.max_columns', 500)


def create_df_of_file(filename, measure, directory="Data", interval=None):
    dataframe = pd.read_csv(os.path.join(directory, filename))

    # linear interpolation to fill NA values
    dataframe = dataframe.interpolate("linear")

    # set start of dataframe to time where it was in Kenya
    time = dataframe["date"].tolist()
    start = time.index("2015-04-21 11:00:00")
    dataframe = dataframe[start:]

    # rename value column to actual measure category
    dataframe.rename(columns={"value": measure}, inplace=True)

    # remove all non valid measure times (only measurements in 10minute steps are valid)
    dataframe = dataframe[dataframe["date"].str[-4:] == "0:00"]
    time = dataframe["date"].tolist()

    # extract time values and replace with index
    date_time = pd.to_datetime(dataframe.pop('date'), format='%Y-%m-%d %H:%M:%S')
    date_time.reset_index(inplace=True, drop=True)

    dataframe.reset_index(inplace=True, drop=True)

    if interval is not None:
        dataframe = dataframe.rolling(interval, min_periods=1).mean()
        dataframe = dataframe[interval::interval]
        dataframe.reset_index(inplace=True, drop=True)

    return dataframe, date_time, time


def create_prec_df(time_df, station, interval=None):
    """
    This function is needed because the prec data had way more measurements than all the other data.
    """

    dataframe = pd.read_csv(os.path.join("Data", f"{station}-prec.csv"))

    # linear interpolation to fill NA values
    dataframe = dataframe.interpolate("linear")

    time = dataframe["date"].tolist()
    start = time.index("2015-04-21 11:00:00")
    time = time[start:]
    dataframe = dataframe[start:]

    dataframe.rename(columns={"value": "prec"}, inplace=True)

    dataframe.set_index('date', inplace=True)

    # remove all measurement times that aren't in the data of the time_df parameter
    to_remove = list(set(time).difference(set(time_df)))
    dataframe.drop(labels=to_remove, inplace=True, errors="ignore")

    dataframe.reset_index(inplace=True, drop=True)

    if interval is not None:
        dataframe = dataframe.rolling(interval, min_periods=1).sum()
        dataframe = dataframe[interval::interval]
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

    _, date_time, df_time = create_df_of_file(f"{station}-nit.csv", "nit", interval=interval)
    prec_df = create_prec_df(df_time, station, interval=interval)

    frames.append(prec_df)

    df = pd.concat(frames, axis=1)

    # transforming time input
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = 365.2425 * day

    if interval is not None:
        day_df = np.sin(timestamp_s * (2 * np.pi / day))
        year_df = np.sin(timestamp_s * (2 * np.pi / year))

        day_df = day_df.rolling(interval, min_periods=1).mean()[interval::interval]
        day_df.reset_index(inplace=True, drop=True)

        year_df = year_df.rolling(interval, min_periods=1).mean()[interval::interval]
        year_df.reset_index(inplace=True, drop=True)

        df["Day Sin"] = day_df
        df["Year Sin"] = year_df

    else:
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))

    return df


def create_tf_dataset(data_in, label, seq_length=72, batch_size=32):
    data = data_in.drop([label], axis=1)
    # data = data_in.copy()
    data = data[:-seq_length]
    data = np.array(data, dtype=np.float32)

    targets = data_in[label]
    targets = targets[seq_length - 1:]
    targets = np.array(targets, dtype=np.float32)

    ds = tf.keras.utils.timeseries_dataset_from_array(data, targets, sequence_length=seq_length,
                                                      sequence_stride=1,
                                                      batch_size=batch_size, )

    return ds


def create_final_ds(station, label, interval=None):
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

    # min max normalization
    train_min = train_df.min()
    train_max = train_df.max()

    train_df_norm = (train_df - train_min) / (train_max - train_min)
    val_df_norm = (val_df - train_min) / (train_max - train_min)
    test_df_norm = (test_df - train_min) / (train_max - train_min)

    # creating tensorflow time series datasets
    train_ds_norm = create_tf_dataset(train_df_norm, label)
    val_ds_norm = create_tf_dataset(val_df_norm, label)
    test_ds_norm = create_tf_dataset(test_df_norm, label)

    train_ds = create_tf_dataset(train_df, label)

    # only the first three return values are needed for training
    return train_ds_norm, val_ds_norm, test_ds_norm, train_df, train_ds, train_df_norm, test_df_norm, val_df_norm, train_min, train_max
