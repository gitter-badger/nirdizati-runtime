"""
Copyright (c) 2016-2017 The Nirdizati Project.
This file is part of "Nirdizati".

"Nirdizati" is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.

"Nirdizati" is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this program.
If not, see <http://www.gnu.org/licenses/lgpl.html>.
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import frequency_encoder


class PredictiveModel():
    hardcoded_prediction = None

    def __init__(self, case_id_col, cls_kwargs, cls_method="rf"):

        self.case_id_col = case_id_col
        self.encoder = frequency_encoder.FrequencyEncoder()

        if cls_method == "gbm":
            self.cls = GradientBoostingRegressor(**cls_kwargs)
        elif cls_method == "rf":
            self.cls = RandomForestRegressor(**cls_kwargs)
        else:
            print("Classifier method not known")

    def fit(self, dt_train):
        train_encoded = self.encoder.encode_trace(dt_train)

        remtime_cols = "remaining_time"

        train_x = train_encoded.drop([self.case_id_col], axis=1)
        train_X = train_x.drop(remtime_cols, axis=1)

        train_y = train_encoded[remtime_cols]

        self.train_X = train_X

        if len(train_y.unique()) < 2:  # less than 2 classes are present
            self.hardcoded_prediction = train_y.iloc[0]
            self.cls.classes_ = train_y.unique()
        else:
            self.cls.fit(train_X, train_y)

    def predict_proba(self, dt_test):
        test_encoded = self.encoder.encode_trace(dt_test)

        remtime_cols = "remaining_time"

        test_x = test_encoded.drop([self.case_id_col], axis=1)
        test_X = test_x.drop(remtime_cols, axis=1)
        test_X = test_X[-1:]

        missing_columns = self.train_X.columns.difference(test_X.columns) # events that are absent from partial trace but occurred in the training set
        for col in missing_columns:
            test_X[col] = 0
            print("Column %s added" % col)

        self.test_case_names = test_encoded[self.case_id_col]
        self.test_X = test_X
        self.test_y = test_encoded[remtime_cols]

        if self.hardcoded_prediction is not None:  # e.g. model was trained with one class only
            predictions_proba = np.array([1.0, 0.0] * test_X.shape[0]).reshape(test_X.shape[0], 2)
        else:
            predictions_proba = self.cls.predict(test_X)

        return predictions_proba
