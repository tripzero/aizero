import json
import os
import shelve
import tempfile
import time
from collections import deque
from datetime import datetime
from aizero import Resource

import numpy as np
import tensorflow as tf


class NoneValue(Exception):
    pass


def normalize(v, mean, std):
    return (v - mean) / std


class Layer:

    def __init__(self, layer_name, resource, property_name, data_max_size=50000000, no_persist=False):
        self.resource = resource
        self.layer_name = layer_name
        self.property_name = property_name
        self.data_max_size = data_max_size
        self.db_file = None
        self.value_changed_cb = None
        self.no_persist = no_persist
        self.data_wait_queue = []

        self.reset_data()

        try:
            for item in resource.values:
                self._values.append(item)
        except:
            print("resource {} has no existing data values".format(
                self.resource.name))

        self.resource.subscribe(property_name, self.value_changed)

    def reset_data(self):
        self._values = deque([], maxlen=self.data_max_size)

    def subscribe(self, cb):
        self.value_changed_cb = cb

    def value_changed(self, value):
        # This variable should be overridden by GenericPredictor:
        if self.value_changed_cb:
            self.value_changed_cb(value)

    def cache_current_value(self):
        if self.value is None:
            raise NoneValue()

        self.append_value(self.value)

    def restore(self, db_file):
        if self.no_persist:
            return

        self.db_file = db_file

        if db_file:
            db = shelve.open(db_file, protocol=2)

            if self.layer_name in db:
                self._values = db[self.layer_name]

            db.close()

    def persist(self):
        if self.no_persist:
            return

        if not self.db_file:
            print("layer ({}) persist failed. no db_file specified".format(
                self.layer_name))
            return

        db = shelve.open(self.db_file, protocol=2)
        db[self.layer_name] = self._values
        db.close()

    @property
    def value(self):
        return self.resource.getValue(self.property_name)

    def get_constant(self, values):
        return tf.constant(values)

    def append_value(self, value):
        self._values.append(float(value))

    @property
    def values(self):
        try:
            return self.resource.values
        except:
            pass

        # if len(self._values) == 0 and self.value != None:
        #    self._values.append(self.value)

        # return a copy, not the deque itself
        # This avoids an error in tensorflow
        return np.copy(self._values)

        # return self._values

    def layer(self):
        return tf.feature_column.numeric_column(key=self.layer_name)


class HashLayer(Layer):

    def __init__(self, layer_name, resource, property_name, bucket_size=1000):
        Layer.__init__(self, layer_name, resource, property_name)
        self.bucket_size = bucket_size

    def get_constant(self, values):
        return tf.SparseTensor(indices=[[i, 0] for i in range(len(values))],
                               values=values,
                               dense_shape=[len(values), 1])

    def layer(self):
        return tf.feature_column.categorical_column_with_hash_bucket(self.layer_name, self.bucket_size)

    def append_value(self, value):
        self._values.append(value)

    def normalize(self):
        new_values = []
        for v in self._values:
            new_values.append(str(v))

        self._values = new_values


def to_dnn_features(feature_cols):
    features_w_emb_cols = [x if not isinstance(x, tf.feature_column.categorical_column_with_hash_bucket("foo", 100).__class__)
                           else tf.feature_column.embedding_column(x, dimension=16) for x in feature_cols]
    return features_w_emb_cols


class GenericPredictor:

    def __init__(self, layers, actual, model_dir="generic_predictor",
                 estimator=None, estimator_class=None, use_dnn_regressor=True):

        if estimator_class == None:
            estimator_class = tf.estimator.LinearRegressor

        features_col = []

        self.layers = layers

        for layer in layers:
            features_col.append(layer.layer())
            layer.subscribe(self.layer_has_updated_value)

        checkpoint = None

        if os.path.exists(model_dir) and os.path.isfile("{}/checkpoint".format(model_dir)):
            checkpoint = model_dir

        if estimator:
            self.model = estimator

        else:
            try:
                self.model = estimator_class(
                    feature_columns=features_col, model_dir=model_dir, warm_start_from=checkpoint)
            except:
                self.model = estimator_class(
                    feature_columns=features_col, model_dir=model_dir)  # tf 1.5 compatibility

            try:
                if use_dnn_regressor:
                    self.model = tf.estimator.DNNRegressor(hidden_units=[100, 50, 25],
                                                           feature_columns=to_dnn_features(
                                                               features_col),
                                                           model_dir=model_dir,
                                                           warm_start_from=checkpoint)
            except Exception as ex:
                print("ERROR: {}".format(ex))

        self.actual = actual
        self.all_layers = np.append(layers, actual)

    def make_data_unique(self):

        length = self._get_minimum_layer_length()

        if length == 0:
            return

        _data = []

        for i in range(length):

            _col = []

            for layer in self.all_layers:
                _col.append(layer.values[i])

            _data.append(_col)

        print("non-unique data length: {}".format(length))

        _data = np.unique(_data, axis=0)

        length = len(_data)

        print("unique data length: {}".format(length))

        for layer in self.all_layers:
            layer.reset_data()

        for i in range(length):

            _col = _data[i]

            for n in range(len(self.all_layers)):
                # print("appending value {}({}) to layer {}".format(_col[n],
                #       type(_col[n]), self.all_layers[n].layer_name))
                self.all_layers[n].append_value(_col[n])

    def layer_has_updated_value(self, value):
        if not self._has_values():
            return

        self.update_layers()

    def update_layers(self):
        for layer in self.all_layers:
            try:
                layer.cache_current_value()
            except NoneValue:
                print("layer {} has None value".format(layer.layer_name))

    def _layer_name_in(self, layer_name, layers):

        if layers == None:
            return False

        for layer in layers:
            if layer_name == layer.layer_name:
                return True

        return False

    def _has_values(self, skip_layers=None):

        for layer in self.layers:
            if self._layer_name_in(layer.layer_name, skip_layers):
                continue

            value = layer.value

            if value == None:
                print("{} does not have a value".format(layer.layer_name))
                return False

        return True

    def _get_minimum_layer_length(self):
        lengths = []
        for layer in self.all_layers:
            lengths.append(len(layer.values))

        return np.min(lengths)

    def _get_features(self, replace_layers=None, only_last=False):
        features = {}
        minimum_length_data = self._get_minimum_layer_length()

        for layer in self.layers:
            if self._layer_name_in(layer.layer_name, replace_layers):
                continue

            if only_last:
                features[layer.layer_name] = layer.get_constant(
                    [layer.value])
            else:
                features[layer.layer_name] = layer.get_constant(
                    layer.values[-minimum_length_data:])

        if replace_layers is not None:
            for layer in replace_layers:
                features[layer.layer_name] = layer.get_constant(
                    [layer.value])

        return features

    def train(self, steps=2000):
        """if not self._has_values():
            print("we don't have valid data to train yet.")
            return
        """

        for layer in self.all_layers:
            print("layer {} has {} values".format(
                layer.layer_name, len(layer.values)))

        for layer in self.all_layers:
            if len(layer.values) == 0:
                print("layer {} has values array of 0. cannot train.".format(
                    layer.layer_name))
                return

        def input_fn():
            min_features = self._get_minimum_layer_length()

            print("training with {} values...".format(min_features))
            features = self._get_features()

            print(features)

            label = tf.constant(self.actual.values[-min_features:])

            return features, label

        self.model.train(input_fn=input_fn, steps=steps)

        eval_results = self.model.evaluate(input_fn=input_fn, steps=1)
        print("training results: {}".format(eval_results))

        for layer in self.all_layers:
            print("persisting {} ({} values) to {}".format(
                layer.layer_name, len(layer._values), layer.db_file))
            layer.persist()

    def predict(self, replace_layers=None):
        ps = self.predict_raw(replace_layers)

        if ps == None:
            return

        for p in ps:
            if "predictions" in p:
                return p["predictions"][0]

            elif "class_ids" in p:
                return p["class_ids"][0]

    def predict_raw(self, replace_layers=None):
        if not self._has_values(replace_layers):
            print("we don't have valid data to predict yet.")
            return

        def input_fn():
            features = self._get_features(replace_layers, True)

            return features

        # print("predicting features: {}".format(input_fn()))

        ps = self.model.predict(input_fn=input_fn)

        return ps


class Em7Resource:

    def __init__(self, device, resource, property_name,
                 start_date, end_date, credentials, limit=None):

        # credentials is tuple of (user, pass, em7_server_url)

        self.device = device
        self.name = resource.replace(" ", "_")
        self.property_name = property_name
        self.credentials = credentials
        self.resource = resource
        self.data = None
        self.limit = limit

        self.load_data(start_date, end_date)

    def load_data(self, start_date, end_date):
        import aizero.em7 as em7
        data = em7.get_data(
            self.device,
            self.resource,
            self.property_name,
            credentials=self.credentials,
            begin_timestamp=start_date,
            end_timestamp=end_date)

        if data is None:
            raise Exception(
                "No data for resource {}. aborting".format(self.device))

        self.corrected_data = np.zeros((0, 2), np.float32)

        for line in data:
            value = line[1]
            if not np.isnan(value):
                self.corrected_data = np.append(
                    self.corrected_data, [line], axis=0)

        if self.data is None:
            self.data = self.corrected_data

        else:
            self.data = np.append(self.data, self.corrected_data, axis=0)

        if self.limit:
            self.data = self.data[:self.limit]

        print("num of values for '{}'/'{}'/'{}': {}".format(self.device,
                                                            self.resource,
                                                            self.property_name,
                                                            len(self.data)))

    @property
    def values(self):
        return self.data[:, 1]

    def subscribe(self, foo, callback):
        pass

    def getValue(self, property_name):
        return self.values[-1]

    def align(self, other, wiggle=7 * 60):

        def fuzzy_date_compare(timestamp1, timestamp2, wiggle):
            return (abs(timestamp1 - timestamp2) <= wiggle)

        values = np.zeros((0, 2), np.float32)
        other_values = np.zeros((0, 2), np.float32)

        for i in self.data:
            for n in other.data:
                if fuzzy_date_compare(i[0], n[0], wiggle):
                    # print("{} and {} have a match: {} vs {}".format(
                    #    self.name, other.name, i[0], n[0]))
                    values = np.append(values, [i], axis=0)
                    other_values = np.append(other_values, [n], axis=0)
                    break

        return values, other_values

    def align_many2(self, others, wiggle=7 * 60):

        def fuzzy_date_compare(timestamp1, timestamp2, wiggle):
            return (abs(timestamp1 - timestamp2) <= wiggle)

        # first find the array with least elements:
        all_of_them = [self]
        for i in others:
            all_of_them.append(i)

        smallest = None
        mins_array = []

        for arry in all_of_them:
            mins_array.append(len(arry.data))

        min_index = np.argwhere(mins_array == np.min(mins_array))[-1][0]
        smallest = all_of_them[min_index]

        # remove the smalled from the 'all' list
        all_of_them.pop(min_index)

        values = np.zeros((0, 2), np.float32)

        for i in smallest.data:
            has_it = False
            for arry in all_of_them:
                for n in arry.data:
                    if fuzzy_date_compare(i[0], n[0], wiggle):
                        has_it = True
                        break

                if not has_it:
                    print("no match for {}".format(i[0]))
                    break

            if has_it:
                # print("adding {} to {} timeseries".format(
                #   i[0], smallest.name))
                values = np.append(values, [i], axis=0)

        smallest.data = values

        for arry in all_of_them:
            v, arry.data = arry.align(smallest, wiggle)

    def align_many(self, others, wiggle=7 * 60):

        # first find the array with least elements:
        all_of_them = [self]
        for i in others:
            all_of_them.append(i)

        smallest = None
        mins_array = []

        for arry in all_of_them:
            mins_array.append(len(arry.data))

        min_index = np.argwhere(mins_array == np.min(mins_array))[-1][0]
        smallest = all_of_them[min_index]

        # remove the smalled from the 'all' list
        all_of_them.pop(min_index)

        # do the alignment of everything else to the smallest data set
        for arry in all_of_them:
            smallest.data, arry.data = arry.align(smallest, wiggle)


class FakeResource:

    def __init__(self, value, name="fakeresource"):
        self.value = value
        self.name = name
        # self.values = [value]

    def getValue(self, pn):
        return self.value

    def subscribe(self, pn, cb):
        pass


class FakeLayer(Layer):

    def __init__(self, name, value):
        resource = FakeResource(value)
        Layer.__init__(self, name, resource, "fake_property_name")
        self.layer_name = name


class FakeHashLayer(HashLayer):

    def __init__(self, name, value):
        resource = FakeResource(value)
        HashLayer.__init__(self, name, resource, "fake_property_name")
        self.layer_name = name


def main():
    month = 7
    start_day = 1
    end_day = 20

    day1 = datetime(2018, month, start_day)
    day2 = datetime(2018, month, end_day)

    air_temp_resource = Em7Resource(
        "redStrip", "AirTemperature", "airTemp", day1, day2)
    soil_temp_resource = Em7Resource(
        "redhouse", "SoilTemperature", "soilTemperature", day1, day2)
    ext_air_temp_resource = Em7Resource(
        "weather", "WeatherUnderground", "Exterior Temperature", day1, day2)
    hvac_reservoir_temp_resource = Em7Resource(
        "geo_heat_exchanger", "HVAC Reservoir", "inlet", day1, day2)

    air_temp_resource.align_many(
        [soil_temp_resource, ext_air_temp_resource, hvac_reservoir_temp_resource])

    air_temp_layer = Layer("air_temp", air_temp_resource, "airTemp")
    soil_temp_layer = Layer("soil_temp", soil_temp_resource, "soilTemp")
    ext_air_temp_layer = Layer(
        "ext_air_temp", ext_air_temp_resource, "Exterior Temperature")
    hvac_reservoir_temp_layer = Layer(
        "reservoir_temp", hvac_reservoir_temp_resource, "inlet")

    predictor = GenericPredictor([soil_temp_layer,
                                  ext_air_temp_layer,
                                  hvac_reservoir_temp_layer],
                                 air_temp_layer, "test_predictor")

    predictor.train()

    result = predictor.predict([FakeLayer("soil_temp", 14.33),
                                FakeLayer("ext_air_temp", 11.8),
                                FakeLayer("reservoir_temp", 21.41)])  # actual 30C

    print("predicted value: {}".format(result))

    print("making data unique...")

    data_length_before_unique = predictor._get_minimum_layer_length()
    predictor.make_data_unique()

    data_length_after_unique = predictor._get_minimum_layer_length()
    predictor.train()

    result = predictor.predict([FakeLayer("soil_temp", 14.33),
                                FakeLayer("ext_air_temp", 11.8),
                                FakeLayer("reservoir_temp", 21.41)])  # actual 30C

    print("predicted value: {}".format(result))


if __name__ == "__main__":
    main()
