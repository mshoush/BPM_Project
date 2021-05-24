import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

import os

import pickle
import joblib
import xgboost as xgb


dataset_ref = "DahnaLoanApplication_prepared"
params_dir = "optimizer_log"
results_dir = "experiment_log"
bucket_method = "cluster"
cls_encoding = "index"
cls_method = "xgboost"
ngram_size = 4

bucket_encoding = "agg"

method_name = "%s_%s" % (bucket_method, cls_encoding)



encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]
}


methods = encoding_dict[cls_encoding]

train_ratio = 0.8
random_state = 22

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))

dataset_name=dataset_ref


# read the data
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'fillna': True}


# split into training and test
train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")


dt_test_prefixes = dataset_manager.generate_prefix_data(test, ngram_size)


# create prefix logs

dt_train_prefixes = dataset_manager.generate_prefix_data(train, ngram_size)


# Bucketing prefixes based on control flow
bucketer_args = {'encoding_method': bucket_encoding,
                 'case_id_col': dataset_manager.case_id_col,
                 'cat_cols': [dataset_manager.activity_col],
                 'num_cols': [],
                 'random_state': random_state}


# load optimal params
optimal_params_filename = os.path.join(params_dir,
                                       "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))

with open(optimal_params_filename, "rb") as fin:
    args = pickle.load(fin)

if bucket_method == "cluster":
    bucketer_args["n_clusters"] = int(args["n_clusters"])
bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)


bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)

bucket_assignments_test = bucketer.predict(dt_test_prefixes)

preds_all = []
test_y_all = []
train_y_all=[]
nr_events_all = []

for bucket in set(bucket_assignments_test):
    if bucket_method == "prefix":
        current_args = args[bucket]
    else:
        current_args = args
    relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
        bucket_assignments_train == bucket]
    relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[
        bucket_assignments_test == bucket]
    dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)

    nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))
    if len(relevant_train_cases_bucket) == 0:
        preds = [dataset_manager.get_class_ratio(train)] * len(relevant_test_cases_bucket)

    else:
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                       relevant_train_cases_bucket)  # one row per event
        train_y = dataset_manager.get_label_numeric(dt_train_bucket)
        train_y_all.extend(train_y)
        if len(set(train_y)) < 2:
            preds = [train_y[0]] * len(relevant_test_cases_bucket)

            test_y_all.extend(dataset_manager.get_label_numeric(dt_test_bucket))
        else:

            feature_combiner = FeatureUnion(
                [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])


            cls = xgb.XGBClassifier(objective='binary:logistic',
                                    n_estimators=500,
                                    learning_rate=current_args['learning_rate'],
                                    subsample=current_args['subsample'],
                                    max_depth=int(current_args['max_depth']),
                                    colsample_bytree=current_args['colsample_bytree'],
                                    min_child_weight=int(current_args['min_child_weight']),
                                    seed=random_state)

            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

            pipeline.fit(dt_train_bucket, train_y)


            # predict separately for each prefix case
            preds = []
            test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
            for _, group in test_all_grouped:

                test_y_all.extend(dataset_manager.get_label_numeric(group))


                _ = bucketer.predict(group)

                preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                pred = pipeline.predict_proba(group)[:, preds_pos_label_idx]


                preds.extend(pred)

    preds_all.extend(preds)


dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all})

dt_results.to_csv('dt_results.csv',index=False)
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)
dt_train_prefixes.to_csv('dt_train_prefixes.csv',index=False)
dt_test_prefixes.to_csv('dt_test_prefixes.csv',index=False)

print("The AUC is: %s\n" % (roc_auc_score(dt_results.actual, dt_results.predicted)))

joblib.dump(pipeline, 'transform_predict.joblib')
