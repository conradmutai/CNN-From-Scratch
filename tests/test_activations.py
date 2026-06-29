"""
test_activations.py
-------------------
Unit tests for the plain activation FUNCTIONS in activations.py:
    relu, softmax, sigmoid
"""

import numpy as np

from src.activations import relu, softmax, sigmoid


# relu
def test_relu_known_values():
    x = np.array([-3.0, -0.0001, 0.0, 0.0001, 5.0])
    out = relu(x)
    expected = np.array([0.0, 0.0, 0.0, 0.0001, 5.0])
    assert np.array_equal(out, expected)


def test_relu_preserves_shape():
    x = np.random.randn(2, 3, 4)
    assert relu(x).shape == x.shape


# softmax
def test_softmax_uniform_input():
    # equal logits -> equal probabilities
    out = softmax(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(out, [1 / 3, 1 / 3, 1 / 3])


def test_softmax_sums_to_one_per_row():
    x = np.random.randn(5, 7)
    out = softmax(x, axis=-1)
    row_sums = np.sum(out, axis=-1)
    assert np.allclose(row_sums, np.ones(5))


def test_softmax_is_numerically_stable():
    # Without the max-subtraction trick, exp(1002) overflows to inf -> NaNs.
    # This test would FAIL if that subtraction were removed.
    x = np.array([1000.0, 1001.0, 1002.0])
    out = softmax(x)
    assert not np.any(np.isnan(out)), "softmax overflowed to NaN on large inputs"
    assert np.isclose(np.sum(out), 1.0)
    # largest logit should get the largest probability
    assert np.argmax(out) == 2


def test_softmax_monotonic():
    # larger logit -> larger probability
    out = softmax(np.array([1.0, 2.0, 3.0]))
    assert out[0] < out[1] < out[2]


def test_softmax_axis_argument():
    # softmax down columns (axis=0): each column should sum to 1
    x = np.random.randn(4, 3)
    out = softmax(x, axis=0)
    col_sums = np.sum(out, axis=0)
    assert np.allclose(col_sums, np.ones(3))


# sigmoid
def test_sigmoid_midpoint():
    assert np.isclose(sigmoid(0.0), 0.5)


def test_sigmoid_bounded_and_saturates():
    x = np.array([-50.0, -1.0, 0.0, 1.0, 50.0])
    out = sigmoid(x)
    assert not np.any(np.isnan(out)), "sigmoid produced NaN"
    assert np.all(out > 0.0) and np.all(out < 1.0 + 1e-12)
    # saturation at the extremes
    assert out[0] < 1e-9          # sigmoid(-50) ~ 0
    assert out[-1] > 1 - 1e-9     # sigmoid(+50) ~ 1


def test_sigmoid_symmetry():
    # sigmoid(-x) == 1 - sigmoid(x)
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    assert np.allclose(sigmoid(-x), 1 - sigmoid(x))


# Plain runner so `python test_activations.py` works without pytest.
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")