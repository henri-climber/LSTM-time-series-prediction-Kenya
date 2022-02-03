import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import seaborn as sns

from data_preprocessing import create_final_ds, create_tf_dataset

# uses Early Stopping
EPOCHS = 50

train_ds, val_ds, test_ds, train_df, = create_final_ds(
    "SHA", "nit", batch_size=32, seq_length=8, interval=18)


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.GlobalMaxPooling1D(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation="linear")
])

model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(learning_rate=0.0001),
              metrics=[tf.metrics.RootMeanSquaredError()])

# tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))
# tf.metrics.RootMeanSquaredError()
# tf.metrics.MeanAbsoluteError()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_root_mean_squared_error",
    patience=20,
    min_delta=0.001,
)


def train_model():
    history = model.fit(train_ds, epochs=EPOCHS,
                        validation_data=val_ds, callbacks=[early_stopping])

    model.save("MAE_interval=3h.h5")

    # list all data in history
    print(history.history.keys())
    # visualize history for accuracy
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model MSE')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # visualize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


train_model()


def visualize_model():
    model.evaluate(train_ds.take(1))
    model.load_weights("nit_models/RMSE_interval=18.h5")
    print(model.evaluate(test_ds))
    print(model.evaluate(train_ds))
    print(model.evaluate(val_ds))
    # visualize specific length of data
    """predictions = []
    labels = []
    for batch in test_ds.take(2000):
        prediction = model.predict(batch)

        prediction = list(prediction)
        for ind, p in enumerate(prediction):
            predictions.append(np.max(p))

        y = [arr.numpy() for arr in batch][1]
        labels.append(y[0])

    x = range(0, len(predictions))
    print(predictions)
    print(labels)"""

    """# visualize whole dataset
    predictions = model.predict(test_ds).flatten()
    #labels = np.array(test_df_norm["nit"])

    x = range(0, len(predictions))
    x1 = range(0, len(labels))

    # convert normalized values to actual previous values
    #reconstruct_norm = lambda d: d * (max_v["nit"] - min_v["nit"]) + min_v["nit"]
    predictions = reconstruct_norm(np.array(predictions))
    labels = reconstruct_norm(np.array(labels))

    print(predictions)
    print(labels)

    sns.set()
    fig, ax1 = plt.subplots()

    ax1.plot(x1, labels, color="blue", label="actual", linewidth=1)
    ax1.plot(x, predictions, color="orange", label="Predictions", linewidth=1.5)

    plt.legend()

    plt.show()


def extract_feature_importance():
    model.evaluate(test_ds.take(1))
    model.load_weights("nit_models/MAE_interval=6.h5")

    feature_df = pd.DataFrame(columns=["Feature removed", "loss", "RMSE"])

    loss, metric = model.evaluate(train_ds)
    feature_df.loc[0] = ["Normal"] + [loss] + [metric]

    for ind, feature in enumerate(train_df_norm.columns[1:]):
        # shuffle the feature
        n_df = train_df_norm.copy()
        np.random.shuffle(n_df[feature])

        feature_dataset = create_tf_dataset(n_df, "nit")

        loss, metric = model.evaluate(feature_dataset)
        feature_df.loc[ind + 1] = [feature] + [loss] + [metric]

    feature_df.to_pickle("features.pkl")
    print(feature_df)


def calculate_important_features():
    feature_df = pd.read_pickle("features.pkl")

    norm = feature_df["RMSE"][0]
    differences = [norm]
    divided = [norm]

    for rmse in feature_df["RMSE"][1:]:
        differences.append(rmse - norm)
        divided.append(rmse / norm)

    feature_df["Difference"] = differences
    feature_df["Divided"] = divided
    feature_df.sort_values("Difference", inplace=True, ignore_index=True, ascending=False)
    feature_df.to_pickle("feature_importance.pkl")


def show_important_features():
    feature_df = pd.read_pickle("feature_importance.pkl")
    print(feature_df)"""
