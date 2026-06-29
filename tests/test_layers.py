"""
test_layers.py
--------------
Unit tests for the layer classes in layers.py.
"""

import numpy as np

from src.layers import (
    Conv2D,
    MaxPool,
    Flatten,
    FullyConnected,
    BatchNorm2D,
    ReLU,
)


# ---------------------------------------------------------------------------
# Shared helper: numerical gradient via central differences.
#
# Perturbs each entry of `param` (IN PLACE, then restores it) and measures how
# loss_fn() changes. loss_fn must be a no-arg closure that recomputes a scalar
# loss using the *current* value of param. Returns an array shaped like param.
#
# Keep test magnitudes small so layer.clip_gradient (max_norm=5.0) does NOT fire,
# otherwise the analytical (clipped) value won't match the numerical (unclipped) one.
# ---------------------------------------------------------------------------
def numerical_grad(param, loss_fn, eps=1e-4):
    grad = np.zeros_like(param, dtype=np.float64)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = float(param[idx])

        param[idx] = orig + eps
        plus = loss_fn()

        param[idx] = orig - eps
        minus = loss_fn()

        param[idx] = orig  # restore
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


# ===========================================================================
# Flatten  — easiest: pure shape round-trip, no math.
# ===========================================================================
def test_flatten_forward_backward():
    f = Flatten()
    x = np.random.randn(2, 3, 4, 4)

    out = f.forward(x)
    assert out.shape == (2, 3 * 4 * 4), "flatten should collapse all dims except batch"

    back = f.backward(out)
    assert back.shape == x.shape, "backward must restore the original input shape"
    assert np.array_equal(back, x), "flatten round-trip must preserve values exactly"


# ===========================================================================
# ReLU layer  — known input/output, and the strict (> 0) gradient gate.
# This is the test that would have caught the dead-ReLU collapse.
# ===========================================================================
def test_relu_forward_backward():
    relu = ReLU()
    x = np.array([[-1.0, 0.0, 2.0],
                  [3.0, -4.0, 5.0]])

    out = relu.forward(x)
    expected_out = np.array([[0.0, 0.0, 2.0],
                             [3.0, 0.0, 5.0]])
    assert np.array_equal(out, expected_out)

    grad_output = np.ones_like(x)
    grad_input = relu.backward(grad_output)
    # derivative is 1 where input > 0, else 0. NOTE: input == 0 -> 0 (strict >).
    expected_grad = np.array([[0.0, 0.0, 1.0],
                              [1.0, 0.0, 1.0]])
    assert np.array_equal(grad_input, expected_grad)


# ===========================================================================
# MaxPool  — 2x2, stride 2. Known max per quadrant; gradient routes ONLY to
# the winning position. Distinct values used so there are no ties in the mask.
# ===========================================================================
def test_maxpool_forward_backward():
    mp = MaxPool()
    x = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=float)  # (1, 1, 4, 4)

    out = mp.forward(x)
    expected_out = np.array([[[[6, 8],
                               [14, 16]]]], dtype=float)
    assert np.array_equal(out, expected_out)

    grad = np.array([[[[1, 2],
                       [3, 4]]]], dtype=float)
    grad_in = mp.backward(grad)
    # each upstream grad lands on the position that was the max in its 2x2 patch
    expected_grad_in = np.array([[[[0, 0, 0, 0],
                                   [0, 1, 0, 2],
                                   [0, 0, 0, 0],
                                   [0, 3, 0, 4]]]], dtype=float)
    assert np.array_equal(grad_in, expected_grad_in)


# ===========================================================================
# FullyConnected  — forward is a plain matmul, so backward is exact calculus.
# Hand-computed assertions; no numerical gradient check needed.
# ===========================================================================
def test_fully_connected_forward():
    inp = np.array([[1.0, 2.0, 3.0]])          # (1, 3)
    weight = np.array([[1.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0]])            # (3, 2)
    bias = np.array([0.5, -0.5])               # (2,)

    fc = FullyConnected(weight, bias, None, None)
    out = fc.forward(inp)
    expected = inp @ weight + bias             # [[1+0+3+0.5, 0+2+3-0.5]] = [[4.5, 4.5]]
    assert np.allclose(out, expected)


def test_fully_connected_backward():
    inp = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])          # (2, 3)
    weight = np.array([[0.1, 0.2],
                       [0.3, 0.4],
                       [0.5, 0.6]])            # (3, 2)
    bias = np.array([0.0, 0.0])
    grad_output = np.array([[1.0, 0.0],
                            [0.0, 1.0]])       # (2, 2)

    fc = FullyConnected(weight, bias, None, None)
    fc.forward(inp)
    grad_input = fc.backward(grad_output)

    expected_grad_weight = inp.T @ grad_output
    expected_grad_bias = np.sum(grad_output, axis=0)
    expected_grad_input = grad_output @ weight.T

    assert np.allclose(fc.grad_weight, expected_grad_weight)
    assert np.allclose(fc.grad_bias, expected_grad_bias)
    assert np.allclose(grad_input, expected_grad_input)

    # shape checks (catch transposed-dim / wrong-axis bugs)
    assert fc.grad_weight.shape == weight.shape
    assert fc.grad_bias.shape == bias.shape
    assert grad_input.shape == inp.shape


# ===========================================================================
# Conv2D forward  — trivial known value: 3x3 input, 3x3 kernel, pad=0, stride 1
# gives a 1x1 output equal to the elementwise sum (bias is NOT added in forward).
# ===========================================================================
def test_conv2d_forward_known_value():
    x = np.arange(9, dtype=float).reshape(1, 1, 3, 3)   # values 0..8, sum = 36
    w = np.ones((1, 1, 3, 3), dtype=float)
    conv = Conv2D(w, np.zeros(1), None, None, None, pad=0)

    out = conv.forward(x)
    assert out.shape == (1, 1, 1, 1)
    assert np.isclose(out[0, 0, 0, 0], np.sum(x * w))   # 36.0


# ===========================================================================
# Conv2D backward  — gradient check for weight grad and input grad.
# (Bias is intentionally skipped; see header note #1.)
# Tolerances are loose because the forward output is float32.
# ===========================================================================
def test_conv2d_backward_gradient_check():
    np.random.seed(0)
    x = np.random.randn(1, 1, 4, 4) * 0.5
    w = np.random.randn(1, 1, 3, 3) * 0.5
    conv = Conv2D(w, np.zeros(1), None, None, None, pad=0)

    out = conv.forward(x)                                # (1, 1, 2, 2)
    grad_output = np.random.randn(*out.shape) * 0.3      # small -> avoids clip

    input_grad = conv.backward(grad_output)
    analytical_w = conv.weight_grad.copy()
    analytical_x = input_grad.copy()

    def loss():
        return float(np.sum(grad_output * conv.forward(x)))

    num_w = numerical_grad(conv.weight, loss)
    num_x = numerical_grad(x, loss)

    # If float32 noise ever trips these, bump eps in numerical_grad to 1e-3.
    assert np.allclose(num_w, analytical_w, rtol=1e-2, atol=1e-3), \
        f"conv weight grad mismatch:\nnum={num_w}\nana={analytical_w}"
    assert np.allclose(num_x, analytical_x, rtol=1e-2, atol=1e-3), \
        f"conv input grad mismatch:\nnum={num_x}\nana={analytical_x}"


# ===========================================================================
# BatchNorm2D forward  — with gamma=1, beta=0 the stored x_hat should have
# per-channel mean ~0 and std ~1 (std is very slightly under 1 due to +epsilon).
# ===========================================================================
def test_batchnorm_forward_normalizes():
    np.random.seed(1)
    x = np.random.randn(4, 3, 5, 5) * 2.0 + 1.0          # arbitrary mean/scale
    gamma = np.ones(3, dtype=np.float32)
    beta = np.zeros(3, dtype=np.float32)
    bn = BatchNorm2D(epsilon=1e-5, beta=beta, gamma=gamma, sigma=None,
                     gamma_grad=None, beta_grad=None,
                     gamma_optimizer=None, beta_optimizer=None)

    bn.forward(x)
    xh = bn.x_hat
    for c in range(3):
        assert abs(xh[:, c].mean()) < 1e-2, f"channel {c} mean not ~0"
        assert abs(xh[:, c].std() - 1.0) < 1e-2, f"channel {c} std not ~1"


# ===========================================================================
# BatchNorm2D backward  — gradient check on the returned input gradient.
# This is the MOST likely test to need a looser tolerance, because forward uses
# (s_h + epsilon) while backward uses s_h (see header note #2), plus float32.
# If it's flaky, loosen atol/rtol or comment it out — it's a known approximation.
# ===========================================================================
def test_batchnorm_backward_gradient_check():
    np.random.seed(2)
    x = np.random.randn(2, 2, 3, 3) * 1.0
    gamma = np.ones(2, dtype=np.float32)
    beta = np.zeros(2, dtype=np.float32)
    bn = BatchNorm2D(epsilon=1e-5, beta=beta, gamma=gamma, sigma=None,
                     gamma_grad=None, beta_grad=None,
                     gamma_optimizer=None, beta_optimizer=None)

    out = bn.forward(x)
    grad_output = np.random.randn(*out.shape) * 0.3

    input_grad = bn.backward(grad_output)  # x_out (NOT clipped, unlike gamma/beta grads)
    analytical_x = input_grad.copy()

    def loss():
        return float(np.sum(grad_output * bn.forward(x)))

    num_x = numerical_grad(x, loss)

    assert np.allclose(num_x, analytical_x, rtol=2e-2, atol=2e-2), \
        f"batchnorm input grad mismatch:\nnum={num_x}\nana={analytical_x}"


# ---------------------------------------------------------------------------
# Plain runner so `python test_layers.py` works without pytest.
# ---------------------------------------------------------------------------
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