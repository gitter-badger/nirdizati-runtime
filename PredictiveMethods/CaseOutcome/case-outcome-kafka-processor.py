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

from PredictiveMonitor import PredictiveMonitor
import pandas as pd
import sys
import batch.dataset_params as dataset_params
import cPickle
import json
from kafka import KafkaProducer, KafkaConsumer
from StringIO import StringIO

if len(sys.argv) != 7:
    sys.exit("Usage: python {} bootstrap-server:port events-topic predictions-topic dataset-name label-column-id json-column-id".format(sys.argv[0]))

bootstrap_server, events_topic, predictions_topic, dataset, label_col, json_col = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

group_id = 'caseOutcome({},{})'.format(dataset, json_col)
consumer = KafkaConsumer(events_topic, group_id='caseOutcome({},{})'.format(dataset, json_col), bootstrap_servers=bootstrap_server, auto_offset_reset='earliest')
producer = KafkaProducer(bootstrap_servers=bootstrap_server)

""" Read from the Kafka topic """
for message in consumer:
    jsonValue = json.loads(message.value)

    s = StringIO(json.dumps(jsonValue))
    test = pd.read_json(s, orient='records')
    s.close()

    case_id_col = dataset_params.case_id_col[dataset]
    event_nr_col = dataset_params.event_nr_col[dataset]
    pos_label = dataset_params.pos_label[dataset]

    static_cols = dataset_params.static_cols[dataset]
    dynamic_cols = dataset_params.dynamic_cols[dataset]
    cat_cols = dataset_params.cat_cols[dataset]

    encoder_kwargs = {"event_nr_col": event_nr_col, "static_cols": static_cols, "dynamic_cols": dynamic_cols,
                      "cat_cols": cat_cols, "oversample_fit": False, "minority_label": "true", "fillna": True,
                      "random_state": 22}

    cls_method = dataset_params.cls_method[dataset]

    if cls_method == "rf":
        cls_kwargs = {"n_estimators": dataset_params.n_estimators[dataset],
                      "random_state": 22}
    elif cls_method == "gbm":
        cls_kwargs = {"n_estimators": dataset_params.n_estimators[dataset],
                      "learning_rate": dataset_params.learning_rate[dataset],
                      "random_state": 22}
    else:
        print("Classifier method not known")

    predictive_monitor = PredictiveMonitor(event_nr_col=event_nr_col, case_id_col=case_id_col,
                                           label_col=label_col, pos_label=pos_label,
                                           cls_method=cls_method, encoder_kwargs=encoder_kwargs, cls_kwargs=cls_kwargs)

    with open('PredictiveMethods/CaseOutcome/predictive_monitor_%s_%s.pkl' % (dataset,label_col), 'rb') as f:
        predictive_monitor.models = cPickle.load(f)

    outcome = predictive_monitor.test(test);
    output = {
        "log":         jsonValue[-1]["log"],
        "case_id": jsonValue[-1]["case_id"],
        "event_nr":    jsonValue[-1]["event_nr"],
        "predictions": {
            json_col: outcome
        }
    }
    print(json.dumps(output))
    producer.send(predictions_topic, json.dumps(output))
