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

from PredictiveModel import PredictiveModel
import numpy as np


class PredictiveMonitor():
    models = {}

    def __init__(self, event_nr_col, case_id_col, encoder_kwargs, cls_kwargs, cls_method="rf"):

        self.event_nr_col = event_nr_col
        self.case_id_col = case_id_col

        self.cls_method = cls_method

        self.encoder_kwargs = encoder_kwargs
        self.cls_kwargs = cls_kwargs

    def train(self, dt_train, max_events=None):

        max_events = np.max(dt_train[self.event_nr_col]) if max_events is None else max_events
        self.max_events = max_events

        pred_model = PredictiveModel(case_id_col=self.case_id_col,
                                         cls_method=self.cls_method,
                                         encoder_kwargs=self.encoder_kwargs, cls_kwargs=self.cls_kwargs)
        pred_model.fit(dt_train)
        self.models = pred_model

    def test(self, dt_test):

        # select relevant model
        pred_model = self.models

        # predict
        predictions_proba = pred_model.predict_proba(dt_test)
        #predictions_proba = np.rint(predictions_proba)
        return predictions_proba.item()
