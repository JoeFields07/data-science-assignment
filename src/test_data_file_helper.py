from data_file_helper import FileHelper
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent          #change if file structure is different
TEST_DATA = Path("data/Segmented_Linear_Baseline.mat")
TEST_FEATURE_FILE = PROJECT_ROOT / Path("data_features") / Path("test.pkl")


obj = FileHelper(project_root=PROJECT_ROOT, filepath=TEST_DATA)
obj.feature_file_path = TEST_FEATURE_FILE
obj.data = obj.load_data_file()
obj.data_stats = obj.extract_features()


def test_load_data_file():
    assert type(obj.data["PlateLFAccX"]) is list
    assert not len(obj.data["PlateLFAccX"]) == 0


def test_load_feature_file():
    if not obj.feature_file_path.is_file():
        obj.export_features()
    data_stats = obj.load_feature_file()

    assert type(data_stats["PlateLFAccX"]["mean"]) is np.ndarray
    assert np.issubdtype(data_stats["PlateLFAccX"]["mean"].dtype, np.floating)

    assert type(data_stats["PlateLFAccX"]["kurtosis"]) is np.ndarray
    assert np.issubdtype(data_stats["PlateLFAccX"]["kurtosis"].dtype, np.floating)
    os.remove(obj.feature_file_path)
    
    
def test_export_features():
    if obj.feature_file_path.is_file():
        os.remove(obj.feature_file_path)
    obj.export_features()

    assert obj.feature_file_path.is_file()
    os.remove(obj.feature_file_path)


def test_remove_data():
    test_obj = obj
    test_obj.remove_data()
    assert test_obj.data == {}


def test_extract_features():
    assert type(obj.data_stats["PlateLFAccX"]["mean"]) is np.ndarray
    assert np.issubdtype(obj.data_stats["PlateLFAccX"]["mean"].dtype, np.floating)

    assert type(obj.data_stats["PlateLFAccX"]["kurtosis"]) is np.ndarray
    assert np.issubdtype(obj.data_stats["PlateLFAccX"]["kurtosis"].dtype, np.floating)