import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from bustle_properties import *
from utils import *

if __name__ == "__main__":

    tensorflow.random.set_seed(1234)

    dataFrame = pd.read_csv(data_directory+"bustle_encoded_training_data.csv", dtype=np.byte)
    input_layer_size = 4 * (85 + 4 * len(InputStringOutputStringProperties))

    dataFrame = dataFrame.drop_duplicates(subset=dataFrame.columns.difference(['label']))
    print("shape after duplication removal: ")
    print(dataFrame.shape)

    positive_samples_dataframe = dataFrame.loc[dataFrame['label'] == 1]
    print("positive samples count:")
    print(positive_samples_dataframe.shape)

    negative_samples_dataframe = dataFrame.loc[dataFrame['label'] == 0]
    print("negative samples count: ")
    print(negative_samples_dataframe.shape)

    minimum_sample_count = min(len(positive_samples_dataframe.index),
                               len(negative_samples_dataframe.index))

    positive_samples_dataframe = positive_samples_dataframe.iloc[:minimum_sample_count, :]
    negative_samples_dataframe = negative_samples_dataframe.iloc[:minimum_sample_count, :]
    dataframe_list = [positive_samples_dataframe, negative_samples_dataframe]
    dataFrame = pd.concat(dataframe_list, axis=0, ignore_index=True)

    X = dataFrame.iloc[:, :input_layer_size]
    y = dataFrame.iloc[:, input_layer_size:]
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    BustleModel = Sequential()
    BustleModel.add(
        Dense((input_layer_size * 3) / 2, input_dim=X.shape[1], activation="relu"))
    BustleModel.add(Dense(y.shape[1], activation="sigmoid"))
    BustleModel.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    BustleModel.summary()
    BustleModel.fit(X_train, y_train, verbose=1, epochs=20, batch_size=8, validation_split=0.2)
    BustleModel.evaluate(X_test, y_test)
    BustleModel.save(models_directory+"EncodedBustleModelForPS.hdf5")
