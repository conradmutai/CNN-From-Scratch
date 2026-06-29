"""
test_model.py
-------------
Tests for the Model container in model.py.
"""

import numpy as np

from src.model import Model


# Fake layers for wiring tests.
class RecordingLayer:
    """Records the order in which forward/backward/update are called."""
    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.updated = False

    def forward(self, x):
        self.log.append(("forward", self.name))
        return x

    def backward(self, grad):
        self.log.append(("backward", self.name))
        return grad

    def update(self):
        self.updated = True
        self.log.append(("update", self.name))


class AddLayer:
    """forward adds k; used to prove output of one layer feeds into the next."""
    def __init__(self, k):
        self.k = k

    def forward(self, x):
        return x + self.k

    def backward(self, grad):
        return grad

    def update(self):
        pass


class ScaleGradLayer:
    """backward multiplies the gradient; used to prove reverse-order chaining."""
    def __init__(self, factor):
        self.factor = factor

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad * self.factor

    def update(self):
        pass


# Forward runs layers in order.
def test_forward_runs_in_order():
    log = []
    layers = [RecordingLayer("A", log), RecordingLayer("B", log), RecordingLayer("C", log)]
    Model(layers).forward(np.zeros(1))

    forward_order = [name for kind, name in log if kind == "forward"]
    assert forward_order == ["A", "B", "C"]


# Forward feeds each layer's output into the next.
def test_forward_chains_values():
    layers = [AddLayer(1), AddLayer(10), AddLayer(100)]
    out = Model(layers).forward(np.array([0.0]))
    assert out[0] == 111.0  # 0 -> 1 -> 11 -> 111


# Backward runs layers in REVERSE order.
def test_backward_runs_in_reverse_order():
    log = []
    layers = [RecordingLayer("A", log), RecordingLayer("B", log), RecordingLayer("C", log)]
    Model(layers).backward(np.zeros(1))

    backward_order = [name for kind, name in log if kind == "backward"]
    assert backward_order == ["C", "B", "A"]


# Backward chains gradients through layers in reverse.
def test_backward_chains_gradients():
    # backward visits C, then B, then A. grad starts at 1, gets *2, *3, *4.
    layers = [ScaleGradLayer(2), ScaleGradLayer(3), ScaleGradLayer(4)]
    grad = Model(layers).backward(np.array([1.0]))
    assert grad[0] == 1.0 * 4 * 3 * 2  # 24


# update() is called on every layer, including no-op layers.
def test_update_calls_every_layer():
    log = []
    recorders = [RecordingLayer("A", log), RecordingLayer("B", log), RecordingLayer("C", log)]
    Model(recorders).update()
    assert all(layer.updated for layer in recorders)


def test_update_does_not_crash_on_noop_layers():
    # MaxPool/Flatten/ReLU have empty update() methods; make sure Model.update
    # tolerates layers whose update does nothing.
    from layers import MaxPool, Flatten, ReLU
    Model([MaxPool(), Flatten(), ReLU()]).update()  # should simply not raise


# Integration: a real forward -> backward -> update cycle reduces loss.
# This is a loose, directional test, not an exact-value one.
def test_integration_training_reduces_loss():
    from src.layers import Flatten, FullyConnected
    from src.optimizers import Adam
    from src.activations import softmax
    from src.losses import cross_entropy_loss

    np.random.seed(0)

    # toy data: 20 samples, 4 features, 2 classes
    n, in_features, n_classes = 20, 4, 2
    x = np.random.randn(n, 1, 2, 2)          # (n, 1, 2, 2) -> flattens to (n, 4)
    y_idx = np.random.randint(0, n_classes, size=n)
    y = np.eye(n_classes)[y_idx]

    weight = np.random.randn(in_features, n_classes).astype(np.float32) * 0.1
    bias = np.zeros(n_classes, dtype=np.float32)
    fc = FullyConnected(weight, bias, Adam(weight), Adam(bias))

    model = Model([Flatten(), fc])

    def loss_now():
        probs = softmax(model.forward(x))
        return cross_entropy_loss(y, probs, n)

    initial_loss = loss_now()

    for _ in range(100):
        probs = softmax(model.forward(x))
        grad = probs - y          # softmax + cross-entropy combined gradient
        model.backward(grad)
        model.update()

    final_loss = loss_now()
    assert final_loss < initial_loss, \
        f"training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"


# Plain runner so `python test_model.py` works without pytest.
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