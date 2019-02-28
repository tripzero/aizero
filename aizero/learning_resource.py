import os
import sys
import traceback

from aizero import get_resource as rsrc
from aizero import GenericPredictor, HashLayer


class LearningResource:

    def __init__(self, model_subdir, layers=[], create_predictors_for_layers=None,
                 estimator_class=None,
                 training_steps=2000):
        """
            :param create_predictors_for_layers - list of predictors to create.
                                                  by default predictors will be
                                                  created for every layer.
        """

        self.training_steps = training_steps

        try:
            db_root = rsrc("ConfigurationResource").config["db_root"]
        except:
            db_root = "{}/.cache".format(os.environ["HOME"])

        self.model_dir = "{}/{}".format(db_root, model_subdir)
        self.values_cache = "{}/values.db".format(self.model_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if len(layers):
            self.create_alternate_predictors(
                layers, create_predictors_for_layers, estimator_class)

    def create_alternate_predictors(self, layers,
                                    create_predictors_for_layers=None,
                                    estimator_class=None):
        predictors = {}

        self.all_layers = layers

        for actual in layers:
            try:
                if (create_predictors_for_layers is not None
                        and actual.layer_name not in create_predictors_for_layers):
                    continue

                feature_layers = list(filter(lambda x: x != actual, layers))
                alt_model = GenericPredictor(layers=feature_layers,
                                             actual=actual,
                                             model_dir="{}/{}".format(
                                                 self.model_dir, actual.layer_name),
                                             estimator_class=estimator_class)

                predictors[actual.layer_name] = alt_model
            except:
                print("Failed to create model for {}".format(actual.layer_name))

        for predictor_key in predictors:
            predictor = predictors[predictor_key]

            for layer in predictor.all_layers:
                print("restoring layer {} from {}".format(
                    layer.layer_name, self.values_cache))
                layer.restore(self.values_cache)
                print("layer {} has ({}) values".format(
                    layer.layer_name, len(layer.values)))

        self.predictors = predictors

        return self.predictors

    def predict(self, feature_name, replace_layers):
        if feature_name not in self.predictors.keys():
            print("no predictor for {}".format(feature_name))

        prediction_record = {}

        for layer in self.all_layers:
            prediction_record[layer.layer_name] = layer.value

        for layer in replace_layers:
            prediction_record[layer.layer_name] = layer.value

        return self.predictors[feature_name].predict(replace_layers)

    def train(self):
        return self._train(self.predictors)

    def _train(self, predictors):
        predictor = None

        try:
            for predictor_key in predictors:
                print("training {}".format(predictor_key))
                predictor = predictors[predictor_key]

                if isinstance(predictor.actual, HashLayer):
                    # We don't know how to solve for hash layers yet... skip
                    continue

                # for layer in predictor.all_layers:
                #     print("layer {} values: {}".format(layer.layer_name, layer.values))

                predictor.train(steps=self.training_steps)

        except TypeError as te:

            print("failed to train model: {}".format(predictor_key))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

        except:
            print("failed to train model: {}".format(predictor_key))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

    @property
    def num_values(self):
        return len(self.all_layers[0].values)

    def to_csv(self, filename):

        with open(filename, 'w') as f:

            # write column headers
            layer_names = []
            for layer in self.all_layers:
                layer_names.append(layer.layer_name)

            f.write(",".join(layer_names))
            f.write("\n")

            num_values = self.num_values

            for i in range(num_values):
                row = []

                for layer in self.all_layers:
                    value = str(layer.values[i])
                    row.append(value)

                f.write(",".join(row))
                f.write("\n")

    def from_csv(self, filename):
        import pandas

        dataframe = pandas.read_csv(csv_file, header=0)

        for layer in self.all_layers:

            data = dataframe.get(layer.layer_name).values

            layer.reset_data()

            for value in data:
                layer.append_value(value)

        layer.persist()
