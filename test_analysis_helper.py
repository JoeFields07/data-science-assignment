from analysis_helper import AnalysisHelper
import numpy as np

TEST_CHANNEL_KEYS = ["Channel1", "Channel2"]
TEST_FEATURE_KEYS = ["Feature1", "Feature2"]
TEST_FEATURE_DATA = {"Channel1" : {"Feature1" : np.array([1, 2, 3, 4, 5]), "Feature2" : np.array([1, 2, 3, 4, 5])}, 
                     "Channel2" : {"Feature1" : np.array([1, 2, 3, 4, 5]), "Feature2" : np.array([1, 2, 3, 4, 5])}}

TEST_OUTLIER_DATA_X = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 1.0, 1.0],
                     [1.0,'NaN',1.0, 1.0, 1.0],
                     [1.0, 1000.0, 1.0, 1.0, 1.0]])

TEST_OUTLIER_DATA_Y = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])

TEST_DATA_LIST = [TEST_FEATURE_DATA, TEST_FEATURE_DATA, TEST_FEATURE_DATA]


def test_remove_outliers_aligned():
    filtered_X, filtered_y, removed = AnalysisHelper.remove_outliers_aligned(TEST_OUTLIER_DATA_X, TEST_OUTLIER_DATA_Y, threshold_limit = 4.0)
    assert type(filtered_X) is np.ndarray
    assert np.issubdtype(filtered_X.dtype, np.floating)

    assert type(filtered_X) is np.ndarray
    assert np.issubdtype(filtered_X.dtype, np.floating)


def test_import_feature_data()
    AnalysisHelper.import_feature_data(TEST_DATA_LIST, TEST_CHANNEL_KEYS, TEST_FEATURE_KEYS, threshold_limit=4.0)

def test_preprocess_data()
    preprocess_data(test_size=0.2, random_state=42)
def test_train_PCA()
    train_PCA()

def test_apply_PCA()
    apply_PCA()

def test_train_classifier()
    train_classifier(classifier='SVM', C=1.0, n_estimators=100, random_state=42)

def test_predict_classifier()
    predict_classifier(figure_num, labels)