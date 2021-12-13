import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=1)

# Average CV score on the training set was: -1.0665493145737346
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            make_union(
                FunctionTransformer(copy),
                StackingEstimator(estimator=LassoLarsCV(normalize=True))
            ),
            make_union(
                make_union(
                    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
                    StackingEstimator(estimator=XGBRegressor(learning_rate=0.5, max_depth=5, min_child_weight=15, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.4, verbosity=0))
                ),
                Binarizer(threshold=0.4)
            )
        ),
        StackingEstimator(estimator=XGBRegressor(learning_rate=0.5, max_depth=5, min_child_weight=17, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.4, verbosity=0))
    ),
    ElasticNetCV(l1_ratio=1.0, tol=0.01)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
