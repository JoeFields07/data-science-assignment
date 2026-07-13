from data_file_helper import FileHelper
import numpy as np
import os
from pathlib import Path

TEST_DATA = Path("./data/Segmented_Linear_Baseline.mat")
TEST_FEATURE_FILE = Path("./data_features/test.pkl")

# make mock dataset for raw data?

obj = FileHelper(TEST_DATA)
obj.feature_filepath = TEST_FEATURE_FILE
obj.data = obj.load_data_file()
obj.data_stats = obj.extract_features()


def test_load_data_file():
    assert type(obj.data["PlateLFAccX"]) is list
    assert np.issubdtype(obj.data["PlateLFAccX"].dtype, np.floating)


def test_load_feature_file():
    if not TEST_FEATURE_FILE.is_file():
        obj.export_features()
    data_stats = obj.load_feature_file()

    assert type(data_stats["PlateLFAccX"]["mean"]) is np.ndarray
    assert np.issubdtype(data_stats["PlateLFAccX"]["mean"].dtype, np.floating)

    assert type(data_stats["PlateLFAccX"]["kurtosis"]) is np.ndarray
    assert np.issubdtype(data_stats["PlateLFAccX"]["kurtosis"].dtype, np.floating)
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
    assert type(obj.data_stats["PlateLFAccX"]["mean"]) is np.ndarray
    assert np.issubdtype(obj.data_stats["PlateLFAccX"]["mean"].dtype, np.floating)

    assert type(obj.data_stats["PlateLFAccX"]["kurtosis"]) is np.ndarray
    assert np.issubdtype(obj.data_stats["PlateLFAccX"]["kurtosis"].dtype, np.floating)