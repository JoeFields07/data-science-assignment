from analysis_helper import AnalysisHelper
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

TEST_CHANNEL_KEYS = ["Channel1", "Channel2"]
TEST_FEATURE_KEYS = ["Feature1", "Feature2"]
TEST_FEATURE_DATA = {"Channel1" : {"Feature1" : np.arange(1.0, 11.0), "Feature2" : np.arange(1.0, 11.0)}, 
                     "Channel2" : {"Feature1" : np.arange(1.0, 11.0), "Feature2" : np.arange(1.0, 11.0)}}
TEST_LABELS = ["File1", "File2"]
TEST_DATA_LIST = [TEST_FEATURE_DATA, TEST_FEATURE_DATA, TEST_FEATURE_DATA]

TEST_OUTLIER_DATA_X = np.array([[1.0, 1.1, 1.2, 1.3, 1.4],
                     [1.0, 1.1, 1.2, 1.3, 1.4],
                     [1.0,'NaN',1.2, 1.3, 1.4],
                     [1.0, 1000.0, 1.2, 1.3, 1.4]])

TEST_OUTLIER_DATA_Y = np.ones(5)



def test_remove_outliers_aligned():
    filtered_X, filtered_y, removed = AnalysisHelper.remove_outliers_aligned(TEST_OUTLIER_DATA_X, TEST_OUTLIER_DATA_Y, threshold_limit = 6.0)
    
    assert type(filtered_X) is np.ndarray
    assert np.issubdtype(filtered_X.dtype, np.floating)
    assert type(filtered_y) is np.ndarray
    assert np.issubdtype(filtered_y.dtype, np.int64)
    assert removed == 2


def test_import_feature_data():
    obj = AnalysisHelper()
    obj.import_feature_data(TEST_DATA_LIST, TEST_CHANNEL_KEYS, TEST_FEATURE_KEYS, threshold_limit=6.0)
    
    assert type(obj.ml_matrix_X) is np.ndarray
    assert np.issubdtype(obj.ml_matrix_X.dtype, np.floating)
    assert type(obj.ml_matrix_y) is np.ndarray
    assert np.issubdtype(obj.ml_matrix_y.dtype, np.int64)


def test_preprocess_data():
    obj = AnalysisHelper()
    obj.import_feature_data(TEST_DATA_LIST, TEST_CHANNEL_KEYS, TEST_FEATURE_KEYS, threshold_limit=6.0)
    obj.preprocess_data(test_size=0.3, random_state=42)

    assert len(obj.X_train) == 7
    assert len(obj.y_train) == 7
    assert len(obj.X_test) == 3
    assert len(obj.y_test) == 3
    assert np.mean(obj.X_train) == 0.0
    assert np.std(obj.X_train) == 1.0


def test_train_PCA():
    obj = AnalysisHelper()
    obj.import_feature_data(TEST_DATA_LIST, TEST_CHANNEL_KEYS, TEST_FEATURE_KEYS, threshold_limit=6.0)
    obj.preprocess_data(test_size=0.3, random_state=42)
    obj.train_PCA()
    
    assert type(obj.X_train_PCA) is np.ndarray
    assert np.issubdtype(obj.X_train_PCA.dtype, np.floating)


def test_apply_PCA():
    obj = AnalysisHelper()
    obj.import_feature_data(TEST_DATA_LIST, TEST_CHANNEL_KEYS, TEST_FEATURE_KEYS, threshold_limit=6.0)
    obj.preprocess_data(test_size=0.3, random_state=42)
    obj.train_PCA()
    obj.apply_PCA()
    
    assert type(obj.X_test_PCA) is np.ndarray
    assert np.issubdtype(obj.X_test_PCA.dtype, np.floating)


def test_train_classifier():
    obj = AnalysisHelper()
    obj.import_feature_data(TEST_DATA_LIST, TEST_CHANNEL_KEYS, TEST_FEATURE_KEYS, threshold_limit=6.0)
    obj.preprocess_data(test_size=0.3, random_state=42)
    obj.train_PCA()
    obj.apply_PCA()

    obj.train_classifier(classifier='SVM', C=1.0, n_estimators=100, random_state=42)
    assert isinstance(obj.clf, SVC)

    obj.train_classifier(classifier='RF', C=1.0, n_estimators=100, random_state=42)
    assert isinstance(obj.clf, RandomForestClassifier)


def test_predict_classifier():
    obj = AnalysisHelper()
    obj.import_feature_data(TEST_DATA_LIST, TEST_CHANNEL_KEYS, TEST_FEATURE_KEYS, threshold_limit=6.0)
    obj.preprocess_data(test_size=0.3, random_state=42)
    obj.train_PCA()
    obj.apply_PCA()
    obj.train_classifier(classifier='RF', C=1.0, n_estimators=100, random_state=42)
    obj.predict_classifier(1, TEST_LABELS)

    assert type(obj.y_pred) is np.ndarray
    assert np.issubdtype(obj.y_pred.dtype, np.int64)