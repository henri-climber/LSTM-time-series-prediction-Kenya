# Model

**All models with interval=6**
- 
    LSTM(64)
    Dropout(0.2)
    LSTM(128)
    Dropout(0.2)
    GlobalMaxPooling1D()
    Dense(64, activation="relu")

    Dense(1, activation="linear")

- loss: MeanSquaredError
- optimizer: Adam (learning rate: 0.001)
- batch_size: 32
- sequence_length: 24


**RMSE_interval=1.h5**