import asyncio
import sys
import traceback

from numpy import float64

from aizero.device_resource import DeviceResource, RunIfCanPolicy
from aizero.resource import Resource, MINS, HOURS, get_resource
from aizero.learningresource_keras import *
from aizero.time_of_day_resource import HourOfDayResource, DayOfWeekResource
from aizero.sys_time import get_current_datetime
from aizero.resource_py3 import Py3Resource as resource_poll


def lstm_convert_dataset(dataset):

    rows = len(dataset)
    cols = len(dataset.columns)

    dataset = dataset.to_numpy()

    dataset = np.expand_dims(dataset, axis=0)

    dataset = dataset.reshape((rows, cols, 1))

    return dataset


class OccupancyPredictorResource(DeviceResource):

    def __init__(self, name=None,
                 occupancy_resource=None, prediction_threshold=0.70):

        if occupancy_resource is None:
            occupancy_resource = "GlobalOccupancy"

        if name is None:
            self.name = "{}_OccuppancyPredictorResource".format(
                occupancy_resource)

        super().__init__(self.name,
                         power_usage=100,
                         variables=["occupancy_prediction",
                                    "occupancy_prediction_raw"])

        self.did_train = False

        self.runtime_policy = RunIfCanPolicy(
            conditions=[lambda: not self.did_train])

        self.prediction_threshold = prediction_threshold

        self.occupancy_resource = occupancy_resource

        self.model_dir = "{}_occupancy_predictor".format(occupancy_resource)

        Resource.waitResource(
            [occupancy_resource], self.init_predictor)

        asyncio.get_event_loop().create_task(self.reset_train_counter())

    @asyncio.coroutine
    def reset_train_counter(self):
        while True:
            yield from asyncio.sleep(HOURS(12))

            self.did_train = False

    def init_predictor(self):
        print("initializing occupancy predictor")

        occupancy_feature = FeatureColumn("occupancy", get_resource(
            self.occupancy_resource), "occupancy")

        day_of_week_feature = FeatureColumn("day_of_week",
                                            DayOfWeekResource(),
                                            "day_of_week")
        hour_feature = FeatureColumn("hour_of_day",
                                     HourOfDayResource(),
                                     "hour_of_day")

        all_features = [occupancy_feature, day_of_week_feature, hour_feature]

        shape = features_shape(all_features)

        layers = [
            keras.layers.LSTM(64, activation="relu",
                              input_shape=(shape - 1, 1)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ]

        self.predictors = Learning(model_subdir=self.model_dir,
                                   features=all_features,
                                   prediction_feature="occupancy",
                                   persist=True,
                                   layers=layers,
                                   loss="binary_crossentropy")

        self.poller = resource_poll(self.poll_func, MINS(10))
        self.poller = resource_poll(self.wait_can_run, HOURS(1))

    def train(self, and_test=True):
        self.predictors.train(and_test=and_test,
                              convert_func=lstm_convert_dataset)

    def predict_occupancy(self, date_to_predict=None):
        """ Return likelyhood that home has at least one human in it
            between 0-100"""
        try:
            if date_to_predict is None:
                date_to_predict = get_current_datetime()

            hour = HourOfDayResource.hour(date_to_predict)

            hour_of_week_fl = FakeFeatureColumn(
                "day_of_week", float64(date_to_predict.date().weekday()))

            hour_of_day_fl = FakeFeatureColumn("hour_of_day", hour)

            occupancy = self.predictors.predict(
                replace_features=[hour_of_day_fl,
                                  hour_of_week_fl],
                convert_func=lstm_convert_dataset)

            print("predicted occupancy: {}".format(round(occupancy, 2)))

            return occupancy

        except ValueError:
            print("Failed to predict. Model probably not trained yet")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

        except Exception:
            print("failed to predict occupancy")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

    def poll_func(self):
        predict_val = self.predict_occupancy()
        if predict_val is not None:
            self.setValue("occupancy_prediction", bool(
                predict_val > self.prediction_threshold))
            self.setValue('occupancy_prediction_raw',
                          float(round(predict_val, 2)))

    def run(self):
        DeviceResource.run(self)

        self.did_train = True

        self.train()
        """except Exception as ex:
            print("Failed to train occupancy_predictor {} - {}".format(
                self.name, ex))
            raise(ex)
            self.did_train = False
            """

        self.stop()

    def stop(self):
        DeviceResource.stop(self)


def main():
    # TODO: do some basic testing
    pass


if __name__ == "__main__":
    main()
