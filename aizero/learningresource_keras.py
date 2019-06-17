#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.models import model_from_json

import asyncio
import os
from aizero import ResourceNotFoundException
from aizero import get_resource as rsrc
from aizero import Resource


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


def to_dataframe(features, last=False):
    f = features[0]

    if last:
        raw_dataset = f.dataframe.tail(1).reset_index(drop=True)
    else:
        raw_dataset = f.dataframe

    for feature in features:
        if feature is f:
            continue

        if last:
            df = feature.dataframe.tail(1).reset_index(drop=True)
        else:
            df = feature.dataframe

        try:
            raw_dataset = pd.merge_asof(raw_dataset, df,
                                        direction="nearest",
                                        left_index=True,
                                        right_index=True)
        except Exception as ex:
            print(ex)
            print(df.tail())
            print("columns: {}".format(df.columns))
            print('index: {}'.format(df.index))
            print("types: {}".format(df.dtypes))
            print("merge with:")
            print(raw_dataset.tail())
            print("columns: {}".format(raw_dataset.columns))
            print('index: {}'.format(raw_dataset.index))
            print("types: {}".format(raw_dataset.dtypes))
            raise ex

    if "timestamp" in raw_dataset.columns:
        raw_dataset = raw_dataset.set_index("timestamp")

    dataset = raw_dataset.copy()

    dataset = dataset.dropna()

    # print("final data frame:")
    # print(dataset.head())

    return dataset


def features_shape(features):
    cols = 0

    for feature in features:
        cols += len(feature.columns())

    return cols


def get_feature(feature_name, features):
    for feature in features:
        if (feature_name == feature.feature_name or
                feature_name in feature.dataframe.columns):
            return feature


class FakeFeatureColumn:

    def __init__(self, name, value):
        self.feature_name = name
        self.value = value

        self._dataframe = pd.DataFrame({name: [value]})

    def columns(self):
        return [self.feature_name]

    @property
    def dataframe(self):
        return self._dataframe

    def value(self, pn=None):
        return self.value


class FeatureColumn:

    def __init__(self, name, resource, property_names):
        self.feature_name = name
        self.resource = resource
        self.property_names = property_names
        self.restore = resource.restore
        self.persist = resource.persist

    def columns(self):

        try:
            return self.dataframe.columns
        except ValueError:
            pass
        """except KeyError as ke:
            print(self.resource.dataframe.columns)
            raise ke
        """

        if isinstance(self.property_names, list):
            return self.property_names

        return [self.property_names]

    @property
    def dataframe(self):

        if "timestamp" in self.resource.dataframe.columns:
            ps = ["timestamp"]
        else:
            ps = []

        if isinstance(self.property_names, list):
            ps.extend(self.property_names)
        else:
            ps.append(self.feature_name)

        df = self.resource.dataframe[ps]

        if ("timestamp" in self.resource.variables and
                "timestamp" in self.resource.dataframe.columns):
            df = df.set_index("timestamp")

        return df

    def value(self, pn=None):
        if pn is None:
            pn = self.feature_name

        return self.dataframe[pn].to_numpy()[-1]


class Learning:

    def __init__(self, model_subdir, features,
                 prediction_feature, persist=False,
                 model_json=None, model=None, layers=None,
                 **kwargs):

        self.all_features = features
        self.prediction_feature = prediction_feature
        self.stats = None
        self.persist = persist

        try:
            db_root = rsrc("ConfigurationResource").config["db_root"]
        except ResourceNotFoundException:
            db_root = "{}/.cache".format(os.environ["HOME"])

        self.model_dir = "{}/{}".format(db_root, model_subdir)
        self.values_cache = "{}".format(self.model_dir)
        self.model_path = "{}/cp.ckpt".format(self.model_dir)

        shape = features_shape(features)

        if model is None:

            if layers is None:
                layers = [
                    keras.layers.Dense(64, activation=tf.nn.relu,
                                       input_shape=[shape - 1]),
                    keras.layers.Dense(64, activation=tf.nn.relu),
                    keras.layers.Dense(1)
                ]

            self.model = keras.Sequential(layers)

            if model_json is not None:
                self.model = model_from_json(model_json)

            # optimizer = tf.train.RMSPropOptimizer(0.001, decay=0.0)
            optimizer = tf.keras.optimizers.RMSprop(0.001)

            self.model.compile(loss='mean_squared_error',
                               optimizer=optimizer,
                               metrics=['mean_absolute_error',
                                        'mean_squared_error'])
        else:
            self.model = model

        if self.persist:
            try:
                self.model.load_weights(self.model_path)

                for feature in self.all_features:
                    cache_dir = "{}/{}.xz".format(
                        self.values_cache, feature.feature_name)
                    print("restoring layer {} from {}".format(
                        feature.feature_name, cache_dir))
                    feature.restore(cache_dir)
                    print("layer {} has ({}) values".format(
                        feature.feature_name, len(feature.dataframe)))

                self.get_stats(self.to_dataframe(self.all_features))

            except Exception as ex:
                print(ex)
                print("WARNING: Failed to load model from {}".format(
                    self.model_path))
                print("you will need to call train() to create model")

    def get_stats(self, dataset=None):

        if dataset is not None and self.stats is None:

            p_feature = self.prediction_feature
            stats = dataset.describe()
            stats.pop(p_feature)
            stats = stats.transpose()
            self.stats = stats

        # else:
        #   self.get_stats(self.to_dataframe(self.all_features))

        return self.stats

    def split_train_test(self, dataset):
        train_data = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_data.index)

        return train_data, test_dataset

    def train(self, and_test=False, tensorboard=False):

        p_feature = self.prediction_feature

        dataset = self.to_dataframe(self.all_features)

        # print("train():")
        # print(dataset.head())

        stats = self.get_stats(dataset)

        train_data = dataset

        if and_test:
            train_data, test_dataset = self.split_train_test(dataset)
            test_labels = test_dataset.pop(p_feature)
            normed_test_data = normalize(
                test_dataset, stats['mean'], stats['std'])

        train_labels = train_data.pop(p_feature)
        normed_train_data = normalize(train_data, stats['mean'], stats['std'])

        # print("normalized data:")
        # print(normed_test_data.tail())

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

        # persist all data
        if self.persist:
            for feature in self.all_features:
                cache_dir = "{}/{}.xz".format(
                    self.values_cache, feature.feature_name)
                feature.persist(cache_dir)

        return history

    def predict(self, replace_features=None):
        p_feature = self.prediction_feature

        features = []
        replace_feature_names = self.feature_names(replace_features)
        all_feature_names = self.feature_names(self.all_features)

        if not np.all(np.isin(replace_feature_names, all_feature_names)):
            raise ValueError(
                "replace_feature name must match a model feature")

        for feature in self.all_features:
            if feature.feature_name not in replace_feature_names:
                features.append(feature)
            else:
                features.append(self.get_feature(feature.feature_name,
                                                 replace_features))

        dataset = self.to_dataframe(features, last=True)

        stats = self.get_stats()

        assert stats is not None, "no statistics. probably haven't trained"

        dataset.pop(p_feature)

        dataset_norm = normalize(dataset, stats['mean'], stats['std'])

        print("prediction set:")
        print(dataset)

        return self.model.predict(dataset_norm).flatten()[-1]

    def evaluate(self):
        p_feature = self.prediction_feature
        stats = self.get_stats()

        dataset = self.to_dataframe(self.all_features)[-10:]

        test_dataset = dataset
        test_labels = test_dataset.pop(p_feature)
        normed_test_data = normalize(
            test_dataset, stats['mean'], stats['std'])

        loss, mae, mse = self.model.evaluate(
            normed_test_data, test_labels, verbose=0)

        return mae

    def get_feature(self, feature_name, features):
        return get_feature(feature_name, features)

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
        dataframe = self.to_dataframe(self.all_features)
        self.plot_dataframe(dataframe)

        plt.show()

    def plot_dataframe(self, dataframe, filter=None):
        import seaborn as sns

        if filter is None:
            return sns.pairplot(dataframe, diag_kind="kde")
        else:
            return sns.pairplot(dataframe[filter], diag_kind="kde")

    def feature_names(self, features):
        names = []

        if features is not None:
            for feature in features:
                names.append(feature.feature_name)

        return names

    def model_json(self):
        return self.model.to_json()


def features_from_csv(csv_path, feature_names):

    dataset = pd.read_csv(csv_path, names=feature_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

    import uuid
    feature_rsrc = Resource(uuid.uuid4().hex, variables=feature_names)
    feature_rsrc.data_frame = dataset

    features = []

    for feature in feature_names:

        feature_col = FeatureColumn(feature,
                                    feature_rsrc,
                                    feature)

        features.append(feature_col)

    return features


def features_to_csv(csv_path, features):
    with open(csv_path, 'w') as f:
        data = to_dataframe(features)

        with open(csv_path, 'w') as f:
            data.to_csv(path_or_buf=f, header=True)


def test_create_cp_learning(persist=False):
    tf.logging.set_verbosity(tf.logging.ERROR)
    dataset_path = keras.utils.get_file(
        "auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    feature_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                     'Weight', 'Acceleration', 'Model Year', 'Origin']

    features = features_from_csv(dataset_path, feature_names)

    print(features[0].dataframe.tail())

    learner = Learning("test_keras", features, "MPG", persist=persist)

    # learner.plot()

    history = learner.train(and_test=True)

    # learner.plot_history(history)

    print("prediction with Cylinders=8:{}".format(
        learner.predict([FakeFeatureColumn("Cylinders", 8)])))

    val = learner.predict()

    print("prediction: {}".format(val))

    mpg_layer = learner.get_feature("MPG", learner.all_features)
    print("actual: {}".format(mpg_layer.value("MPG")))

    val2 = learner.predict()

    assert abs(val - val2) <= 10

    return val


def model_from_checkpoint():
    tf.logging.set_verbosity(tf.logging.ERROR)
    dataset_path = keras.utils.get_file(
        "auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    layer_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                   'Acceleration', 'Model Year', 'Origin']

    features = features_from_csv(dataset_path, layer_names)

    learner = Learning("test_keras", features, "MPG", persist=True)

    # Don't train. should load weights automatically from test_create_cp_learning

    val = learner.predict()

    print("prediction: {}".format(val))

    mpg_layer = learner.get_feature("MPG", learner.all_features)
    print("actual: {}".format(mpg_layer.value("MPG")))

    return val


def test_to_dataframe():
    tf.logging.set_verbosity(tf.logging.ERROR)
    import uuid
    import time

    r1 = Resource(uuid.uuid4().hex, variables=["a"])

    r1.set_value("a", 1)
    r1.set_value("a", 2)
    r1.set_value("a", 3)
    r1.set_value("a", 4)

    print(r1.dataframe.head())

    time.sleep(1.0)

    r2 = Resource(uuid.uuid4().hex, variables=["b"])

    r2.set_value("b", 1)
    r2.set_value("b", 2)
    r2.set_value("b", 3)
    r2.set_value("b", 4)

    print(r2.dataframe.head())

    fc1 = FeatureColumn('a', r1, r1.variables)
    fc2 = FeatureColumn('b', r2, r2.variables)

    combined_frame = to_dataframe([fc1, fc2])

    print(combined_frame.head())

    assert "a" in combined_frame.columns
    assert "b" in combined_frame.columns

    assert len(combined_frame) == 4

    df_last = to_dataframe([fc1, fc2], True)

    print(df_last)

    assert "a" in df_last.columns
    assert "b" in df_last.columns
    assert len(df_last) == 1

    assert df_last["a"].to_numpy()[0] == 4


def test_restore_model():
    tf.logging.set_verbosity(tf.logging.ERROR)
    p1 = test_create_cp_learning(True)

    p2 = model_from_checkpoint()

    assert abs(p2 - p1) < 10


def test_model_to_json():
    import json

    l = Learning("/tmp/test_model_subdir", [
        FeatureColumn("feature1", Resource("Feature1", ["a"]), ['a']),
        FeatureColumn("feature2", Resource("Feature2", ["ab"]), ['ab'])],
        "feature2")

    print("model json: {}".format(json.dumps(l.model_json())))

    assert l.model_json() is not None


def test_model_from_json():
    import uuid
    model_json = """
    {
       "class_name":"Sequential",
       "config":{
          "name":"sequential_3",
          "layers":[
             {
                "class_name":"Dense",
                "config":{
                   "name":"dense_9",
                   "trainable":true,
                   "batch_input_shape":[
                      null,
                      1
                   ],
                   "dtype":"float32",
                   "units":64,
                   "activation":"relu",
                   "use_bias":true,
                   "kernel_initializer":{
                      "class_name":"GlorotUniform",
                      "config":{
                         "seed":null,
                         "dtype":"float32"
                      }
                   },
                   "bias_initializer":{
                      "class_name":"Zeros",
                      "config":{
                         "dtype":"float32"
                      }
                   },
                   "kernel_regularizer":null,
                   "bias_regularizer":null,
                   "activity_regularizer":null,
                   "kernel_constraint":null,
                   "bias_constraint":null
                }
             },
             {
                "class_name":"Dense",
                "config":{
                   "name":"dense_10",
                   "trainable":true,
                   "dtype":"float32",
                   "units":64,
                   "activation":"relu",
                   "use_bias":true,
                   "kernel_initializer":{
                      "class_name":"GlorotUniform",
                      "config":{
                         "seed":null,
                         "dtype":"float32"
                      }
                   },
                   "bias_initializer":{
                      "class_name":"Zeros",
                      "config":{
                         "dtype":"float32"
                      }
                   },
                   "kernel_regularizer":null,
                   "bias_regularizer":null,
                   "activity_regularizer":null,
                   "kernel_constraint":null,
                   "bias_constraint":null
                }
             },
             {
                "class_name":"Dense",
                "config":{
                   "name":"dense_11",
                   "trainable":true,
                   "dtype":"float32",
                   "units":1,
                   "activation":"linear",
                   "use_bias":true,
                   "kernel_initializer":{
                      "class_name":"GlorotUniform",
                      "config":{
                         "seed":null,
                         "dtype":"float32"
                      }
                   },
                   "bias_initializer":{
                      "class_name":"Zeros",
                      "config":{
                         "dtype":"float32"
                      }
                   },
                   "kernel_regularizer":null,
                   "bias_regularizer":null,
                   "activity_regularizer":null,
                   "kernel_constraint":null,
                   "bias_constraint":null
                }
             }
          ]
       },
       "keras_version":"2.2.4-tf",
       "backend":"tensorflow"
    }
    """

    l = Learning("/tmp/test_model_subdir", [
        FeatureColumn("feature1", Resource(uuid.uuid4().hex, ["a"]), ['a']),
        FeatureColumn("feature2", Resource(uuid.uuid4().hex, ["ab"]), ['ab'])],
        "feature2", model_json=model_json)

    assert l.model


def test_predict_features_will_replace_features():

    model = Learning("/tmp/test_model_subdir", [
        FakeFeatureColumn("feature1", 1),
        FakeFeatureColumn("feature2", 2)],
        "feature2")

    try:
        model.predict(replace_features=[
            FakeFeatureColumn("feature1", 1.1)])
    except AssertionError:
        # we expect an assertion error because this model is not trained
        pass

    ex_hit = False
    try:
        model.predict(replace_features=[
            FakeFeatureColumn("not_feature1", 1.1)])
    except ValueError:
        ex_hit = True

    assert ex_hit


def test_get_feature():
    import uuid

    assert get_feature("feature1", [FakeFeatureColumn("feature1", 1)])
    assert get_feature(
        "not_feature1", [FakeFeatureColumn("feature1", 1)]) is None

    rsrc = Resource(uuid.uuid4().hex, ["a"])
    rsrc.set_value("a", 1)

    assert get_feature(
        "a", [FeatureColumn("feature1", rsrc,
                            property_names=["a"])])


def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    test_to_dataframe()
    test_model_to_json()


if __name__ == "__main__":
    main()
