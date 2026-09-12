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
    bst_cpu = lgb.train({**base_params, "device_type": "cpu"}, train_data, num_boost_round=30)
    # small data stays on the CPU fallback path at the default threshold
    bst_metal = lgb.train({**base_params, "device_type": "metal"}, train_data, num_boost_round=30)
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


def _row_wise_crash_data(seed=7, n=20000):
    """Sparse + NaN-heavy data whose layout triggers row-wise sharing."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 10), dtype=np.float32)
    X[:, 0] = rng.random(n)
    X[:, 1] = rng.random(n)
    X[rng.random(n) < 0.7, 1] = 0.0  # sparse zeros -> multi-val group
    X[:, 2] = rng.random(n)
    X[rng.random(n) < 0.4, 2] = np.nan  # NaN-heavy
    X[:, 3] = rng.integers(0, 5, n)
    X[:, 4] = rng.random(n) * 100
    X[:, 5] = rng.normal(0, 1, n)
    X[rng.random(n) < 0.5, 5] = 0.0
    X[:, 6] = rng.random(n)
    X[:, 7] = rng.integers(0, 250, n).astype(np.float32)
    X[:, 8] = rng.random(n)
    X[rng.random(n) < 0.2, 8] = 0.0
    X[:, 9] = rng.random(n) * 1000
    y = X[:, 0] * 50 + np.nan_to_num(X[:, 2], nan=0.5) * 30 + rng.normal(0, 1, n)
    return X, y


@needs_metal
def test_metal_row_wise_sparse_completes():
    """Row-wise data with sparse groups must not crash GPU training.

    Regression test: the GPU->pool copy used raw group-bin boundaries
    instead of the share-state packed feature offsets, so with row-wise
    sharing every dense histogram landed shifted and training died in
    CHECK_GT(left/right_count, 0) a few splits in.
    """
    X, y = _row_wise_crash_data()
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "quantile",
        "verbose": -1,
        "num_threads": 8,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "seed": 0,
        "device_type": "metal",
        "metal_min_hist_workload": 0,
    }
    bst = lgb.train(params, train_data, num_boost_round=5)
    assert np.isfinite(bst.predict(X)).all()


@needs_metal
def test_metal_row_wise_matches_cpu_quality():
    """Forced row-wise, Metal must learn an equivalent model to CPU.

    Same quality bar as test_metal_gpu_path_matches_cpu_quality but with
    row-wise histogram sharing forced on, covering the packed-offset copy
    path deterministically.
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
        "force_row_wise": True,
    }
    bst_cpu = lgb.train({**base_params, "device_type": "cpu"}, train_data, num_boost_round=20)
    bst_metal = lgb.train(
        {**base_params, "device_type": "metal", "metal_min_hist_workload": 0},
        train_data,
        num_boost_round=20,
    )
    pred_cpu = bst_cpu.predict(X_test)
    pred_metal = bst_metal.predict(X_test)
    assert np.corrcoef(pred_cpu, pred_metal)[0, 1] > 0.999
    assert float(np.abs(pred_cpu - pred_metal).mean()) < 5e-3
    from sklearn.metrics import roc_auc_score

    assert abs(roc_auc_score(y_test, pred_cpu) - roc_auc_score(y_test, pred_metal)) < 1e-3


def _bundled_data(seed=11, n=21000):
    """Three mutually-exclusive sparse features EFB bundles into one single-val group.

    f0/f1/f2 are nonzero on disjoint row thirds, so their conflict counts are
    0 and they share one DenseBin with offset sub-ranges. All three drive the
    label hard so trees must split on the bundled group (otherwise a bad
    GPU->pool copy for bundled groups would go unnoticed).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 7), dtype=np.float32)
    X[0::3, 0] = rng.random((n + 2) // 3)
    X[1::3, 1] = rng.random((n + 1) // 3)
    X[2::3, 2] = rng.random(n // 3)
    X[:, 3] = rng.normal(0, 1, n)
    X[:, 4] = rng.random(n) * 10
    X[:, 5] = rng.normal(5, 2, n)
    X[:, 6] = rng.integers(0, 20, n).astype(np.float32)
    y = 50 * X[:, 0] + 50 * X[:, 1] + 50 * X[:, 2] + 2 * X[:, 3] + rng.normal(0, 1, n)
    return X, y


@needs_metal
@pytest.mark.parametrize("force_row_wise", [False, True])
def test_metal_bundled_group_matches_cpu(force_row_wise):
    """Bundled single-val groups must copy to the pool with correct offsets.

    Regression test: the GPU->pool copy plan must use each group's cumulative
    stored offsets per sub-feature. Using per-feature flags instead silently
    shifts every bundled sub-feature's histogram (src ranges [0, 1) + rest
    instead of the true stored sub-ranges).
    """
    X, y = _bundled_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    train_data = lgb.Dataset(X_train, label=y_train)
    base_params = {
        "objective": "regression",
        "verbose": -1,
        "num_threads": 1,
        "num_leaves": 31,
        "max_bin": 63,
        "min_data_in_leaf": 20,
        "force_row_wise": force_row_wise,
    }
    bst_cpu = lgb.train({**base_params, "device_type": "cpu"}, train_data, num_boost_round=20)
    bst_metal = lgb.train(
        {**base_params, "device_type": "metal", "metal_min_hist_workload": 0},
        train_data,
        num_boost_round=20,
    )
    pred_cpu = bst_cpu.predict(X_test)
    pred_metal = bst_metal.predict(X_test)
    assert np.corrcoef(pred_cpu, pred_metal)[0, 1] > 0.999
    assert float(np.abs(pred_cpu - pred_metal).mean()) < 0.5
    rmse = lambda p: float(np.sqrt(np.mean((p - y_test) ** 2)))
    assert abs(rmse(pred_cpu) - rmse(pred_metal)) / rmse(pred_cpu) < 0.05


def _multival_data(seed=11, n=21000):
    """Sparse features EFB must place in a multival group.

    f0/f1 are binary with 6% ones (true most-frequent-bin-0 sub-features);
    f2/f3 are continuous but nonzero on the same 15% of rows, so their
    conflict count forces a multival (rather than single-val bundled) group.
    All four drive the label hard so trees must split on the GPU-packed
    multival columns (otherwise a bad pack/copy for them would go unnoticed).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 7), dtype=np.float32)
    X[:, 0] = (rng.random(n) < 0.06).astype(np.float32)
    X[:, 1] = (rng.random(n) < 0.06).astype(np.float32)
    support = rng.random(n) < 0.15
    X[support, 2] = rng.random(support.sum())
    X[support, 3] = rng.random(support.sum()) * 10
    X[:, 4] = rng.normal(0, 1, n)
    X[:, 5] = rng.random(n) * 10
    X[:, 6] = rng.normal(5, 2, n)
    y = 50 * X[:, 0] + 50 * X[:, 1] + 50 * X[:, 2] + 5 * X[:, 3] + 2 * X[:, 4] + rng.normal(0, 1, n)
    return X, y


@needs_metal
@pytest.mark.parametrize("force_row_wise", [False, True])
def test_metal_multival_columns_match_cpu(force_row_wise):
    """Multival sub-features packed as GPU columns must match CPU histograms.

    Regression test: multival per-sub iterators return natural bins (PushData
    stores most-frequent rows as absent and shifts the rest so Get() decodes
    to natural). Remapping packed values (e.g. subtracting 1 for
    most-frequent-bin-0 subs) silently shifts binary sub-feature histograms
    and diverges training.
    """
    X, y = _multival_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    train_data = lgb.Dataset(X_train, label=y_train)
    base_params = {
        "objective": "regression",
        "verbose": -1,
        "num_threads": 1,
        "num_leaves": 31,
        "max_bin": 63,
        "min_data_in_leaf": 20,
        "force_row_wise": force_row_wise,
    }
    bst_cpu = lgb.train({**base_params, "device_type": "cpu"}, train_data, num_boost_round=20)
    bst_metal = lgb.train(
        {**base_params, "device_type": "metal", "metal_min_hist_workload": 0},
        train_data,
        num_boost_round=20,
    )
    pred_cpu = bst_cpu.predict(X_test)
    pred_metal = bst_metal.predict(X_test)
    assert np.corrcoef(pred_cpu, pred_metal)[0, 1] > 0.999
    assert float(np.abs(pred_cpu - pred_metal).mean()) < 0.5
    rmse = lambda p: float(np.sqrt(np.mean((p - y_test) ** 2)))
    assert abs(rmse(pred_cpu) - rmse(pred_metal)) / rmse(pred_cpu) < 0.05


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
    bst = lgb.train(params, train_data, num_boost_round=30)
    expected = bst.predict(X_test)
    model_path = tmp_path / "metal_model.txt"
    bst.save_model(model_path)
    reloaded = lgb.Booster(model_file=model_path)
    np.testing.assert_array_equal(reloaded.predict(X_test), expected)
