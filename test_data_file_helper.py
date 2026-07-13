from data_file_helper import FileHelper
import numpy as np
import os

TEST_DATA = "./data/Segmented_Linear_Baseline.mat"
TEST_FEATURE_FILE = "./data_features/test.pkl"

# make mock dataset for raw data?

obj = FileHelper(TEST_DATA)
obj.feature_filepath = TEST_FEATURE_FILE
obj.data = obj.load_data_file()
obj.data_stats = obj.extract_features()


def test_load_file_data():
    #should a test file be made that is uploaded to Git
    assert type(obj.data["PlateLFAccX"]) is np.ndarray
    assert np.issubdtype(obj.data["PlateLFAccX"].dtype, np.floating)


def test_load_feature_file():
    if not TEST_FEATURE_FILE.is_file():
        obj.export_features()
    data_stats = obj.load_data_file_features()

    assert type(data_stats["PlateLFAccX"]) is np.ndarray
    assert np.issubdtype(data_stats["PlateLFAccX"].dtype, np.floating)
    os.remove(TEST_FEATURE_FILE)
    
    
def test_export_features():
    if TEST_FEATURE_FILE.is_file():
        os.remove(TEST_FEATURE_FILE)
    obj.export_features()

    assert TEST_FEATURE_FILE.is_file()
    os.remove(TEST_FEATURE_FILE)


def test_remove_data():
    test_obj = obj
    test_obj.remove_data()
    assert test_obj.data == {}


def test_extract_features():
    data_stats = obj.export_features()
    assert type(data_stats["PlateLFAccX"]["mean"]) is np.ndarray
    assert np.issubdtype(data_stats["PlateLFAccX"]["mean"].dtype, np.floating)

    assert type(data_stats["PlateLFAccX"]["kurtosis"]) is np.ndarray
    assert np.issubdtype(data_stats["PlateLFAccX"]["kurtosis"].dtype, np.floating)

