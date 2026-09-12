# coding: utf-8
"""Tests for the Metal GPU training backend (``device_type="metal"``).

These tests run only when the loaded ``lib_lightgbm`` was built with
``-DUSE_METAL=1`` on a machine with a Metal device; otherwise they skip.
"""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from .utils import load_breast_cancer


def _metal_usable() -> bool:
    """Return True if this build can train (even briefly) on Metal."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.9, random_state=0)
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {"objective": "binary", "verbose": -1, "num_threads": 1, "device_type": "metal"}
    try:
        lgb.train(params, train_data, num_boost_round=2)
    except lgb.basic.LightGBMError:
        return False
    return True


needs_metal = pytest.mark.skipif(not _metal_usable(), reason="Metal backend not available in this build")


@needs_metal
def test_metal_matches_cpu_on_small_data():
    """Below the GPU workload threshold, Metal must equal CPU bit-for-bit."""
    X_train, X_test, y_train, _ = train_test_split(
        *load_breast_cancer(return_X_y=True), test_size=0.2, random_state=2
    )
    train_data = lgb.Dataset(X_train, label=y_train)
    base_params = {
        "objective": "binary",
        "verbose": -1,
        "num_threads": 1,
        "num_leaves": 15,
        "max_bin": 63,
    }
    bst_cpu = lgb.train({**base_params, "device_type": "cpu"}, train_data, num_boost_round=10)
    # small data stays on the CPU fallback path at the default threshold
    bst_metal = lgb.train({**base_params, "device_type": "metal"}, train_data, num_boost_round=10)
    np.testing.assert_array_equal(bst_cpu.predict(X_test), bst_metal.predict(X_test))


@needs_metal
def test_metal_gpu_path_matches_cpu_quality():
    """Forced onto the GPU, Metal must learn an equivalent model.

    GPU float accumulation is nondeterministic at the bit level (as documented
    for the OpenCL backend), so this asserts quality agreement, not equality.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        *load_breast_cancer(return_X_y=True), test_size=0.2, random_state=2
    )
    train_data = lgb.Dataset(X_train, label=y_train)
    base_params = {
        "objective": "binary",
        "verbose": -1,
        "num_threads": 1,
        "num_leaves": 15,
        "max_bin": 63,
    }
    bst_cpu = lgb.train({**base_params, "device_type": "cpu"}, train_data, num_boost_round=20)
    # metal_min_hist_workload=0 routes even this small data through GPU kernels
    bst_metal = lgb.train(
        {**base_params, "device_type": "metal", "metal_min_hist_workload": 0},
        train_data,
        num_boost_round=20,
    )
    pred_cpu = bst_cpu.predict(X_test)
    pred_metal = bst_metal.predict(X_test)
    assert np.corrcoef(pred_cpu, pred_metal)[0, 1] > 0.999
    assert float(np.abs(pred_cpu - pred_metal).mean()) < 5e-3
    # quality parity: same ranking ability on held-out data
    from sklearn.metrics import roc_auc_score

    assert abs(roc_auc_score(y_test, pred_cpu) - roc_auc_score(y_test, pred_metal)) < 1e-3


@needs_metal
def test_metal_model_loads_and_predicts(tmp_path):
    """A Metal-trained model must save/load/predict like any other model."""
    X_train, X_test, y_train, _ = train_test_split(
        *load_breast_cancer(return_X_y=True), test_size=0.2, random_state=2
    )
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "verbose": -1,
        "num_threads": 1,
        "num_leaves": 15,
        "max_bin": 63,
        "device_type": "metal",
        "metal_min_hist_workload": 0,
    }
    bst = lgb.train(params, train_data, num_boost_round=10)
    expected = bst.predict(X_test)
    model_path = tmp_path / "metal_model.txt"
    bst.save_model(model_path)
    reloaded = lgb.Booster(model_file=model_path)
    np.testing.assert_array_equal(reloaded.predict(X_test), expected)
