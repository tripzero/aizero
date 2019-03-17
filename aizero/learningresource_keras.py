#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

import asyncio
import os
from aizero import ResourceNotFoundException
from aizero import get_resource as rsrc


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def normalize(x, mean, std):
    return (x - mean / std)


def get_minimum_layer_length(layers):
    lengths = []
    for layer in layers:
        lengths.append(len(layer.values))

    return np.min(lengths)


def to_dataframe(layers, last=False):
    d = {}
    min_features = get_minimum_layer_length(layers)

    for layer in layers:
        d[layer.layer_name] = layer.values[-min_features:]

    raw_dataset = pd.DataFrame(data=d, dtype=np.float)

    dataset = raw_dataset.copy()

    dataset = dataset.dropna()

    if not last:
        return dataset

    return dataset[-1:]


def timestamped_layers_to_dataframe(layers, last=False):
    d = {}
    min_features = get_minimum_layer_length(layers)

    for layer in layers:
        try:
            col_name = "{}_timestamp".format(layer.layer_name)
            data = layer.resource.data
            d[col_name] = data[-min_features:, 0]
            d[layer.layer_name] = data[-min_features:, 1]
        except Exception as ex:
            print(ex)
            print("layer {} does not appear to have timestamp. skipping".format(
                layer.layer_name))

    raw_dataset = pd.DataFrame(data=d, dtype=np.float)

    dataset = raw_dataset.copy()

    dataset = dataset.dropna()

    if not last:
        return dataset

    return dataset


class Learning:

    def __init__(self, model_subdir, features,
                 prediction_feature, **kwargs):

        self.all_layers = features
        self.prediction_feature = prediction_feature
        self.stats = None

        try:
            db_root = rsrc("ConfigurationResource").config["db_root"]
        except ResourceNotFoundException:
            db_root = "{}/.cache".format(os.environ["HOME"])

        self.model_dir = "{}/{}".format(db_root, model_subdir)
        self.values_cache = "{}/values.db".format(self.model_dir)
        self.model_path = "{}/cp.ckpt".format(self.model_dir)

        self.model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu,
                               input_shape=[len(features) - 1]),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)

        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=['mean_absolute_error',
                                    'mean_squared_error'])

        try:
            self.model.load_weights(self.model_path)

            for layer in self.all_layers:
                print("restoring layer {} from {}".format(
                    layer.layer_name, self.values_cache))
                layer.restore(self.values_cache)
                print("layer {} has ({}) values".format(
                    layer.layer_name, len(layer.values)))

            self.get_stats(self.to_dataframe(self.all_layers))

        except Exception as ex:
            print("WARNING: Failed to load model from {}".format(
                self.model_path))
            print("you will need to call train() to create model")

    def get_stats(self, dataset=None):

        if dataset is not None:

            p_feature = self.prediction_feature
            stats = dataset.describe()
            stats.pop(p_feature)
            stats = stats.transpose()
            self.stats = stats

        return self.stats

    def split_train_test(self, dataset):
        train_data = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_data.index)

        return train_data, test_dataset

    def train(self, and_test=False, tensorboard=False):

        p_feature = self.prediction_feature

        dataset = self.to_dataframe(self.all_layers)

        stats = self.get_stats(dataset)

        train_data = dataset
        if and_test:
            train_data, test_dataset = self.split_train_test(dataset)
            test_labels = test_dataset.pop(p_feature)
            normed_test_data = normalize(
                test_dataset, stats['mean'], stats['std'])

        train_labels = train_data.pop(p_feature)
        normed_train_data = normalize(train_data, stats['mean'], stats['std'])

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=100)

        cp_callback = keras.callbacks.ModelCheckpoint(self.model_path,
                                                      save_weights_only=True,
                                                      verbose=0)

        callbacks = [early_stop, cp_callback, PrintDot()]

        if tensorboard:
            from datetime import datetime
            t = datetime.now().time()
            tb_callback = keras.callbacks.TensorBoard(
                log_dir="logs/{}".format(t))
            callbacks.append(tb_callback)

        history = self.model.fit(
            normed_train_data, train_labels,
            epochs=1000, validation_split=0.2, verbose=0,
            callbacks=callbacks)

        if and_test:
            loss, mae, mse = self.model.evaluate(
                normed_test_data, test_labels, verbose=0)

            print("\ntesting set mean abs Error: {:5.2f}".format(mae))
            self.mean_absolute_error = mae

        return history

    def predict(self, replace_layers=None):
        p_feature = self.prediction_feature

        layers = []
        replace_layer_names = self.layer_names(replace_layers)

        for layer in self.all_layers:
            if layer.layer_name not in replace_layer_names:
                layers.append(layer)
            else:
                layers.append(self.get_layer(layer.layer_name, replace_layers))

        dataset = self.to_dataframe(layers, last=True)

        stats = self.get_stats()

        dataset.pop(p_feature)

        dataset_norm = normalize(dataset, stats['mean'], stats['std'])

        print("prediction set:")
        print(dataset)

        return self.model.predict(dataset_norm).flatten()[-1]

    def evaluate(self):
        p_feature = self.prediction_feature
        stats = self.get_stats()

        dataset = self.to_dataframe(self.all_layers)[-10:]

        test_dataset = dataset
        test_labels = test_dataset.pop(p_feature)
        normed_test_data = normalize(
            test_dataset, stats['mean'], stats['std'])

        loss, mae, mse = self.model.evaluate(
            normed_test_data, test_labels, verbose=0)

        return mae

    def get_layer(self, layer_name, layers):
        for layer in layers:
            if layer.layer_name == layer_name:
                return layer

    def to_dataframe(self, layers, last=False):
        return to_dataframe(layers, last)

    def plot_history(self, history):
        import matplotlib.pyplot as plt

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label='Val Error')
        plt.ylim([0, 5])
        plt.legend()

        # plt.figure()
        # plt.xlabel('Epoch')
        # plt.ylabel('Mean Square Error [$MPG^2$]')
        # plt.plot(hist['epoch'], hist['mean_squared_error'],
        #         label='Train Error')
        # plt.plot(hist['epoch'], hist['val_mean_squared_error'],
        #         label='Val Error')
        # plt.ylim([0, 20])
        # plt.legend()
        plt.show()

    def plot(self):
        import matplotlib.pyplot as plt
        dataframe = self.to_dataframe(self.all_layers)
        self.plot_dataframe(dataframe)

        plt.show()

    def plot_dataframe(self, dataframe, filter=None):
        import seaborn as sns

        if filter is None:
            return sns.pairplot(dataframe, diag_kind="kde")
        else:
            return sns.pairplot(dataframe[filter], diag_kind="kde")

    def layer_names(self, layers):
        names = []

        if layers is not None:
            for layer in layers:
                names.append(layer.layer_name)

        return names


def layers_from_csv(csv_path, layer_names):
    from aizero.generic_predictor import FakeLayer

    raw_dataset = pd.read_csv(csv_path, names=layer_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()

    layers = []

    for key in layer_names:
        layer = FakeLayer(key, None)

        data = dataset.get(key).values

        layer.reset_data()

        for value in data:
            layer.append_value(value)

        layers.append(layer)

    return layers


def layers_to_csv(csv_path, layers):
    with open(csv_path, 'w') as f:

        data = timestamped_layers_to_dataframe(layers)
        with open(csv_path, 'w') as f:
            data.to_csv(path_or_buf=f, header=True)


def test_create_cp_learning():

    dataset_path = keras.utils.get_file(
        "auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    layer_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                   'Acceleration', 'Model Year', 'Origin']

    layers = layers_from_csv(dataset_path, layer_names)

    learner = Learning("test_keras", layers, "MPG")

    # learner.plot()

    history = learner.train(and_test=True)

    # learner.plot_history(history)

    val = learner.predict()

    print("prediction: {}".format(val))

    mpg_layer = learner.get_layer("MPG", learner.all_layers)
    print("actual: {}".format(mpg_layer.values[-1]))

    return val


def test_model_from_checkpoint():
    dataset_path = keras.utils.get_file(
        "auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    layer_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                   'Acceleration', 'Model Year', 'Origin']

    layers = layers_from_csv(dataset_path, layer_names)

    learner = Learning("test_keras", layers, "MPG")

    # Don't train. should load weights automatically from test_create_cp_learning

    val = learner.predict()

    print("prediction: {}".format(val))

    mpg_layer = learner.get_layer("MPG", learner.all_layers)
    print("actual: {}".format(mpg_layer.values[-1]))

    return val


if __name__ == "__main__":
    p1 = test_create_cp_learning()
    p2 = test_model_from_checkpoint()

    assert p1 == p2
