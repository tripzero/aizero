"""
    device_predictor

    predicts whether or not a device will be running or not based on
    time of day and day of week.
"""

from resource import HOURS, MINS, Resource, ResourceRequires

import tensorflow as tf
from tensorflow import keras

from aizero.learningresource_keras import Learning, FeatureColumn
from aizero.learningresource_keras import FakeFeatureColumn, features_shape
from aizero.time_of_day_resource import HourOfDayResource
from aizero.resource_py3 import Py3Resource as resource_poll
from aizero.sys_time import get_current_datetime

import sys
import traceback


class DevicePredictor(Resource):

    def __init__(self, device_name):
        super().__init__("{}_device_predictor".format(
            device_name), ["running"])

        self.device_name = device_name
        self.device = None
        self.running = False

        self.rsrcs = ResourceRequires([
            device_name,
            "HourOfDayResource",
            "DayOfWeekResource"
        ], self.init_predictor)

    def init_predictor(self, rsrcs):

        self.device = rsrcs(self.device_name)

        features = [
            FeatureColumn("hour_of_day", self.rsrcs(
                "HourOfDayResource"), "hour_of_day"),
            FeatureColumn("day_of_week", self.rsrcs(
                "DayOfWeekResource"), "day_of_week"),
            FeatureColumn("device_running", self.device, "running"),
        ]

        self.subdir = self.name

        shape = [features_shape(features) - 1]

        layers = [keras.layers.Dense(4, activation=tf.nn.relu,
                                     input_shape=shape,
                                     kernel_initializer='random_normal'),
                  keras.layers.Dense(4, activation=tf.nn.relu,
                                     kernel_initializer='random_normal'),
                  keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                     kernel_initializer='random_normal')]

        self.predictor = Learning(self.subdir, features=features,
                                  prediction_feature="device_running",
                                  persist=True,
                                  loss="binary_crossentropy",
                                  layers=layers)

        self.poller = resource_poll(self.poll_func, MINS(10))
        self.poller2 = resource_poll(self.predictor.train, HOURS(1))

    def poll_func(self):
        self.running = self.predict()

        self.setValue("running", self.running)

        print("{} is predicted running: {}".format(self.name, self.running))

    def predict(self, date_to_predict=None):
        """ Return likelyhood that home has at least one human in it
            between 0-100 """

        running = None

        try:
            if date_to_predict is None:
                date_to_predict = get_current_datetime()

            hour = HourOfDayResource.hour(date_to_predict)
            weekday = date_to_predict.date().weekday()

            running = self.predictor.predict("device_running",
                                             [FakeFeatureColumn("day_of_week",
                                                                weekday),
                                              FakeFeatureColumn("hour_of_day",
                                                                hour)])

        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

        finally:
            return running
