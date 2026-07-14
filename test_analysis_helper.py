from analysis_helper import AnalysisHelper
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

MAD_FILTER_THRESHOLD = 10.0     # threshold for MAD filtering (default 10.0)
PCA_COMPONENTS = 10             # number of PCA components to calculate (default 10)
CLASSIFIER = 'RF'               # classifier ('SVM' or 'RF')
RF_N_ESTIMATORS = 100           # Random Forest classifier 'n_estimators' parameter (default 100)
SVM_C = 1.0                     # SVM classifier 'C' parameter (default 1.0)
RANDOM_SEED = 42                # random seed used
TRAIN_TEST_SPLIT = 0.2          # proportion of the test set (CHANGING THIS WILL BREAK THE TEST)

# 3 files, 2 channels with 2 features, which are 10 rows long. Makes 3 classes, X matrix length 20
TEST_CHANNEL_KEYS = ["Channel1", "Channel2"]
TEST_FEATURE_KEYS = ["Feature1", "Feature2"]
TEST_FEATURE_DATA = {"Channel1" : {"Feature1" : np.arange(1.0, 11.0), "Feature2" : np.arange(1.0, 11.0)}, 
                     "Channel2" : {"Feature1" : np.arange(1.0, 11.0), "Feature2" : np.arange(1.0, 11.0)}}
TEST_LABELS = ["File1", "File2", "File3"]
TEST_DATA_LIST = [TEST_FEATURE_DATA, TEST_FEATURE_DATA, TEST_FEATURE_DATA]

TEST_OUTLIER_DATA_X = np.array([[1.0, 1.1, 1.2, 1.3, 1.4],
                     [1.0, 1.1, 1.2, 1.3, 1.4],
                     [1.0, np.nan, 1.2, 1.3, 1.4],
                     [1000.0, 1.0, 1.2, 1.3, 1.4]])

TEST_OUTLIER_DATA_Y = np.ones(4, dtype=int)


def test_remove_outliers_aligned():
    filtered_X, filtered_y, removed = AnalysisHelper.remove_outliers_aligned(None, 
                                                                             TEST_OUTLIER_DATA_X, 
                                                                             TEST_OUTLIER_DATA_Y, 
                                                                             MAD_FILTER_THRESHOLD)
    
    assert type(filtered_X) is np.ndarray
    assert np.issubdtype(filtered_X.dtype, np.floating)
    assert type(filtered_y) is np.ndarray
    assert np.issubdtype(filtered_y.dtype, np.int64)
    assert removed == 1     #should only remove the 1000, not the nan


def test_import_feature_data():
    obj = AnalysisHelper()
    obj.import_feature_data(data_list=TEST_DATA_LIST, 
                            channel_keys=TEST_CHANNEL_KEYS, 
                            feature_keys=TEST_FEATURE_KEYS, 
                            threshold_limit=MAD_FILTER_THRESHOLD)
    
    assert type(obj.ml_matrix_X) is np.ndarray
    assert np.issubdtype(obj.ml_matrix_X.dtype, np.floating)
    assert type(obj.ml_matrix_y) is np.ndarray
    assert np.issubdtype(obj.ml_matrix_y.dtype, np.int64)


def test_preprocess_data():
    obj = AnalysisHelper()
    obj.import_feature_data(data_list=TEST_DATA_LIST, 
                            channel_keys=TEST_CHANNEL_KEYS, 
                            feature_keys=TEST_FEATURE_KEYS, 
                            threshold_limit=MAD_FILTER_THRESHOLD)
    obj.preprocess_data(test_size=TRAIN_TEST_SPLIT, 
                        random_state=RANDOM_SEED)
    #makes 30 rows

    assert obj.X_train.shape[0] == 24
    assert obj.y_train.shape[0] == 24
    assert obj.X_test.shape[0] == 6
    assert obj.y_test.shape[0] == 6


def test_train_PCA():
    obj = AnalysisHelper()
    obj.import_feature_data(data_list=TEST_DATA_LIST, 
                            channel_keys=TEST_CHANNEL_KEYS, 
                            feature_keys=TEST_FEATURE_KEYS, 
                            threshold_limit=MAD_FILTER_THRESHOLD)
    obj.preprocess_data(test_size=TRAIN_TEST_SPLIT, 
                        random_state=RANDOM_SEED)
    obj.train_PCA(n_components=PCA_COMPONENTS, 
                  random_state=RANDOM_SEED)
    
    assert type(obj.X_train_PCA) is np.ndarray
    assert np.issubdtype(obj.X_train_PCA.dtype, np.floating)


def test_apply_PCA():
    obj = AnalysisHelper()
    obj.import_feature_data(data_list=TEST_DATA_LIST, 
                            channel_keys=TEST_CHANNEL_KEYS, 
                            feature_keys=TEST_FEATURE_KEYS, 
                            threshold_limit=MAD_FILTER_THRESHOLD)
    obj.preprocess_data(test_size=TRAIN_TEST_SPLIT, 
                        random_state=RANDOM_SEED)
    obj.train_PCA(n_components=PCA_COMPONENTS, 
                  random_state=RANDOM_SEED)
    obj.apply_PCA()
    
    assert type(obj.X_test_PCA) is np.ndarray
    assert np.issubdtype(obj.X_test_PCA.dtype, np.floating)


def test_train_classifier():
    obj = AnalysisHelper()
    obj.import_feature_data(data_list=TEST_DATA_LIST, 
                            channel_keys=TEST_CHANNEL_KEYS, 
                            feature_keys=TEST_FEATURE_KEYS, 
                            threshold_limit=MAD_FILTER_THRESHOLD)
    obj.preprocess_data(test_size=TRAIN_TEST_SPLIT, 
                        random_state=RANDOM_SEED)
    obj.train_PCA(n_components=PCA_COMPONENTS, 
                  random_state=RANDOM_SEED)
    obj.apply_PCA()

    obj.train_classifier(classifier='SVM', 
                         C=SVM_C, 
                         n_estimators=RF_N_ESTIMATORS, 
                         random_state=RANDOM_SEED)
    assert isinstance(obj.clf, SVC)

    obj.train_classifier(classifier='RF', 
                         C=SVM_C, 
                         n_estimators=RF_N_ESTIMATORS, 
                         random_state=RANDOM_SEED)
    assert isinstance(obj.clf, RandomForestClassifier)


def test_predict_classifier():
    obj = AnalysisHelper()
    obj.import_feature_data(data_list=TEST_DATA_LIST, 
                            channel_keys=TEST_CHANNEL_KEYS, 
                            feature_keys=TEST_FEATURE_KEYS, 
                            threshold_limit=MAD_FILTER_THRESHOLD)
    obj.preprocess_data(test_size=TRAIN_TEST_SPLIT, 
                        random_state=RANDOM_SEED)
    obj.train_PCA(n_components=PCA_COMPONENTS, 
                  random_state=RANDOM_SEED)
    obj.apply_PCA()
    obj.train_classifier(classifier='RF', 
                         C=SVM_C, 
                         n_estimators=RF_N_ESTIMATORS, 
                         random_state=RANDOM_SEED)
    obj.predict_classifier()

    assert type(obj.y_pred) is np.ndarray
    assert np.issubdtype(obj.y_pred.dtype, np.int64)