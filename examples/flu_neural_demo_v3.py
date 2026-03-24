"""
FLU Neural Demo  v3
===================
1. Repo smoke-test      — verifies the GitHub upload is intact
2. FLUOwenInitializer   — new initialiser using the DN2-proven APN-Owen
                          scrambled field (FractalNet.generate_scrambled)
3. Variance audit       — compares raw (pre-norm) scale of all three inits
4. Standard MLP         — Kaiming vs FLU (integer) vs FLU-Owen on moons data
5. FLU-Sparse-Lazy      — decoupled microtask net, now using FLU-Owen init

FLUOwenInitializer design
-------------------------
The FLU integer initialiser (FLUInitializer) generates balanced Latin
hyperprism coordinates in Z_n and normalises to unit variance.  The raw
digits span [-n//2, n//2], so for n=29 the pre-normalisation std ~ 8.4.

FLU-Owen takes the proven DN2 pipeline:
    index -> FM-Dance coords -> APN-Owen scramble -> [0,1)^d
and then centres to [-0.5, 0.5)^d.  The pre-normalisation std is always
bounded by 1/(2*sqrt(3)) ~ 0.289, regardless of n.  That is ~3-4% of the
FLU integer scale.

This much tighter raw scale means:
  - The normalisation divisor is very stable and close to 0.289
  - The initialisation variance is predictable across different layer shapes
  - Each layer can use a distinct seed_rank -> independent APN permutations
    (DN2-P1/P2 PROVEN: Latin + net-t-value preserved under APN scrambling)

The implementation:
  - Finds the smallest n in GOLDEN_SEEDS s.t. n^2 >= rows*cols
    (guarantees APN seeds are available for the scrambling)
  - Generates rows*cols points from FractalNet(n, d=2).generate_scrambled()
  - Takes column 0, centres, reshapes to (rows, cols), normalises to unit var

A seed_rank counter is kept so each layer call gets a distinct APN scramble.
"""

import sys, time, math, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 0.  FLUOwenInitializer ────────────────────────────────────────────────────

from flu.core.factoradic import GOLDEN_SEEDS
from flu.core.fractal_net import FractalNet

_SUPPORTED_ODD_N = sorted(GOLDEN_SEEDS.keys())   # [3,5,7,11,13,17,19,23,29,31,…]


def _nearest_supported_n(min_n: int) -> int:
    """Smallest n in GOLDEN_SEEDS >= min_n (so APN scrambling is available)."""
    for n in _SUPPORTED_ODD_N:
        if n >= min_n:
            return n
    return _SUPPORTED_ODD_N[-1]


class FLUOwenInitializer:
    """
    Weight initialiser using the DN2-proven APN-Owen scrambled FractalNet field.

    For each layer call:
      1. Choose n = smallest supported odd >= ceil(sqrt(rows*cols))
         so that n^2 >= rows*cols and GOLDEN_SEEDS[n] is available.
      2. Generate rows*cols points from FractalNet(n, d=2).generate_scrambled(
             mode='owen', seed_rank=self._counter)
      3. Centre to [-0.5, 0.5), take column 0, reshape to (rows, cols).
      4. Normalise to unit variance.

    Properties (STATUS from theorem registry):
      - Latin property preserved under APN-Owen scrambling  — DN2-P1 PROVEN
      - Net t-value preserved                               — DN2-P2 PROVEN
      - FFT spectral-artefact reduction confirmed           — DN2-P3 CONFIRMED
      - Raw std tightly bounded in [0, 1/(2*sqrt(3))] ~ 0.289
        (all coordinates live in [0,1), centered gives [-0.5, 0.5))
      - Each call uses an independent APN scramble via seed_rank counter

    Parameters
    ----------
    auto_advance : bool
        If True (default), increment seed_rank counter after each weights()
        call so that successive layers get independent scrambles.
    """

    def __init__(self, auto_advance: bool = True):
        self.auto_advance = auto_advance
        self._counter = 0

    def weights(
        self,
        shape,
        seed_rank: int | None = None,
    ) -> np.ndarray:
        """
        Generate an Owen-scrambled FLU weight tensor.

        Parameters
        ----------
        shape     : (rows, cols)
        seed_rank : int | None  — override the auto-advance counter

        Returns
        -------
        np.ndarray  float64, unit variance, ~zero mean
        """
        shape = tuple(int(s) for s in shape)
        if len(shape) != 2:
            raise ValueError(f"FLUOwenInitializer.weights() requires a 2-D shape, got {shape}")

        rows, cols = shape
        needed     = rows * cols
        n          = _nearest_supported_n(math.ceil(math.sqrt(needed)))
        sr         = seed_rank if seed_rank is not None else self._counter

        net = FractalNet(n=n, d=2)
        pts = net.generate_scrambled(needed, mode="owen", seed_rank=sr) - 0.5
        W   = pts[:, 0].reshape(rows, cols)          # column 0 → weight matrix

        std = W.std()
        if std > 1e-12:
            W = W / std                              # unit variance

        if self.auto_advance and seed_rank is None:
            self._counter += 1

        return W

    def to_torch_parameter(
        self, weights: np.ndarray, requires_grad: bool = True
    ) -> "torch.nn.Parameter":
        t = torch.tensor(weights, dtype=torch.float32)
        return torch.nn.Parameter(t, requires_grad=requires_grad)

    def raw_stats(self, shape) -> dict:
        """Return pre-normalisation statistics for a given shape."""
        rows, cols = shape
        needed = rows * cols
        n      = _nearest_supported_n(math.ceil(math.sqrt(needed)))
        net    = FractalNet(n=n, d=2)
        pts    = net.generate_scrambled(needed, mode="owen") - 0.5
        flat   = pts[:, 0]
        return {
            "n": n, "raw_std": float(flat.std()),
            "raw_mean": float(flat.mean()),
            "raw_min": float(flat.min()), "raw_max": float(flat.max()),
        }


# ── helpers shared by all three inits ─────────────────────────────────────────

_owen_init_global = FLUOwenInitializer()


def kaiming_init_linear(m: nn.Linear):
    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def flu_init_linear(m: nn.Linear):
    from flu.applications import FLUInitializer
    W = FLUInitializer().weights(m.weight.shape)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(W.astype(np.float32)))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def flu_owen_init_linear(m: nn.Linear, seed_rank: int | None = None):
    W = _owen_init_global.weights(m.weight.shape, seed_rank=seed_rank)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(W.astype(np.float32)))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ── 1.  Repo smoke-test ───────────────────────────────────────────────────────

def smoke_test():
    print("=" * 62)
    print("  SMOKE TEST — FLU repository")
    print("=" * 62)
    from flu.applications import FLUInitializer
    from flu.applications.neural import DynamicFLUNetwork
    from flu.theory.theorem_registry import status_report

    W = FLUInitializer().weights((9, 9))
    print(f"  FLUInitializer(9,9)        mean={W.mean():.2e}  std={W.std():.4f}  OK")

    net = DynamicFLUNetwork(n=7, in_features=7)
    net.add_layer("fc1", 7).add_layer("fc2", 7).add_layer("fc3", 7)
    rep = net.expansion_report()
    print(f"  DynamicFLUNetwork          layers={rep['layers_registered']}  "
          f"latin={rep['latin_property'][:6]}  OK")

    # FLUOwenInitializer smoke
    oi = FLUOwenInitializer()
    Wo = oi.weights((9, 9))
    rs = oi.raw_stats((27, 27))
    print(f"  FLUOwenInitializer(9,9)    mean={Wo.mean():.4f}  std={Wo.std():.4f}  "
          f"n_used={rs['n']}  raw_std={rs['raw_std']:.4f}  OK")

    lines = status_report().splitlines()
    proven = next(l for l in lines if "PROVEN" in l and "theorems" in l)
    print(f"  TheoremRegistry            {proven.strip()}  OK")
    print()


# ── 2.  Variance audit ────────────────────────────────────────────────────────

def variance_audit():
    print("=" * 62)
    print("  VARIANCE AUDIT — raw (pre-norm) scale comparison")
    print("=" * 62)
    from flu.applications import FLUInitializer
    from flu.core.parity_switcher import generate as ps_gen

    shapes = [(2, 27), (27, 27), (27, 2)]
    oi = FLUOwenInitializer()

    print(f"  {'Shape':<10}  {'Kaiming std':>11}  {'FLU int std':>11}  {'Owen raw std':>12}  {'Owen/FLU':>8}")
    print(f"  {'-'*10}  {'-'*11}  {'-'*11}  {'-'*12}  {'-'*8}")
    for shape in shapes:
        rows, cols = shape
        # Kaiming: std = sqrt(2 / fan_in)
        kai_std = math.sqrt(2.0 / cols)

        # FLU integer raw std
        n_flu = max(rows, cols); n_flu = n_flu if n_flu % 2 == 1 else n_flu + 1
        flu_raw = ps_gen(n_flu, 2, signed=True).astype(float).flatten()[:rows*cols]
        flu_std = flu_raw.std()

        # Owen raw std (pre-norm, bounded in [-0.5, 0.5))
        rs = oi.raw_stats(shape)
        owen_std = rs["raw_std"]

        ratio = owen_std / flu_std
        print(f"  {str(shape):<10}  {kai_std:>11.4f}  {flu_std:>11.4f}  {owen_std:>12.4f}  {ratio:>7.1%}")

    print()
    print("  Owen raw std is bounded in [0, 0.289] for ALL shapes/n values.")
    print("  This comes directly from the unit-interval [0,1)^d structure.")
    print("  After unit-variance normalisation all three converge, but Owen")
    print("  has the tightest and most predictable pre-norm scale.")
    print()


# ── 3.  Synthetic dataset ─────────────────────────────────────────────────────

def make_data(n=800, seed=42):
    rng = np.random.RandomState(seed)
    n_h = n // 2
    t   = np.linspace(0, np.pi, n_h)
    X0  = np.c_[np.cos(t),     np.sin(t)]     + rng.randn(n_h, 2) * 0.15
    X1  = np.c_[1-np.cos(t), 1-np.sin(t)-0.5] + rng.randn(n_h, 2) * 0.15
    X   = np.vstack([X0, X1]).astype(np.float32)
    y   = np.array([0]*n_h + [1]*n_h, dtype=np.int64)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    s = int(0.75 * n)
    return (torch.from_numpy(X[:s]), torch.from_numpy(y[:s]),
            torch.from_numpy(X[s:]), torch.from_numpy(y[s:]))


# ── 4.  Standard MLP — three inits ───────────────────────────────────────────

class MLP(nn.Module):
    HIDDEN = 27

    def __init__(self, init: str = "kaiming"):
        super().__init__()
        H = self.HIDDEN
        linears = [nn.Linear(2, H), nn.Linear(H, H), nn.Linear(H, 2)]
        self.layers = nn.ModuleList(linears)

        oi = FLUOwenInitializer()
        for i, m in enumerate(linears):
            if init == "kaiming":
                kaiming_init_linear(m)
            elif init == "flu":
                flu_init_linear(m)
            elif init == "flu_owen":
                flu_owen_init_linear(m, seed_rank=i)   # layer-distinct seeds
            else:
                raise ValueError(init)

    def forward(self, x):
        x = torch.tanh(self.layers[0](x))
        x = torch.tanh(self.layers[1](x))
        return self.layers[2](x)


def train_mlp(model, Xtr, ytr, Xte, yte, epochs=150, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, accs = [], []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward(); opt.step()
        losses.append(loss.detach().item())
        with torch.no_grad():
            model.eval()
            accs.append((model(Xte).argmax(1)==yte).float().mean().item())
    return losses, accs


def run_mlp_comparison(Xtr, ytr, Xte, yte):
    print("=" * 62)
    print("  STANDARD MLP — Kaiming vs FLU-int vs FLU-Owen  (epochs=150)")
    print("=" * 62)
    results = {}
    for init_name in ("kaiming", "flu", "flu_owen"):
        torch.manual_seed(0)
        m = MLP(init=init_name)
        t0 = time.time()
        losses, accs = train_mlp(m, Xtr, ytr, Xte, yte)
        elapsed = time.time() - t0
        results[init_name] = {"losses": losses, "accs": accs}
        label = {"kaiming": "Kaiming ", "flu": "FLU-int ", "flu_owen": "FLU-Owen"}[init_name]
        print(f"  {label}  loss={losses[-1]:.4f}  acc={accs[-1]:.3f}  "
              f"loss@ep1={losses[0]:.4f}  t={elapsed:.2f}s")
    print()
    return results


# ── 5.  Microtask queue + decoupled local layer (now with Owen init) ──────────

class MicrotaskQueue:
    def __init__(self, capacity=4):
        self.capacity = capacity
        self._q       = collections.deque()
        self.n_pushed = self.n_flushed = 0

    def push(self, fn):
        self._q.append(fn); self.n_pushed += 1
        if len(self._q) >= self.capacity: self.flush()

    def flush(self):
        while self._q: self._q.popleft()(); self.n_flushed += 1

    def stats(self):
        return {"pushed": self.n_pushed, "flushed": self.n_flushed, "pending": len(self._q)}


class FLULocalLayer(nn.Module):
    def __init__(self, in_f, out_f, n_classes=2, threshold=0.15,
                 queue_capacity=4, lr=5e-3, layer_idx=0):
        super().__init__()
        self.fc         = nn.Linear(in_f, out_f)
        self.act        = nn.Tanh()
        self.local_head = nn.Linear(out_f, n_classes)
        self.threshold  = threshold
        self.queue      = MicrotaskQueue(capacity=queue_capacity)
        self.opt        = torch.optim.SGD(
            list(self.fc.parameters()) + list(self.local_head.parameters()),
            lr=lr, momentum=0.85, weight_decay=1e-4)
        self._cached = None

        # FLU-Owen init — distinct seed per layer
        flu_owen_init_linear(self.fc, seed_rank=layer_idx)
        nn.init.kaiming_uniform_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)

    def forward(self, x):
        out = self.act(self.fc(x))
        self._cached = out.detach().clone()
        return out

    def local_step(self, labels):
        if self._cached is None: return 0.0, 0.0
        with torch.no_grad():
            mask    = (self._cached.abs() > self.threshold).float()
            density = mask.mean().item()
        x_sparse   = (self._cached * mask).detach()
        logits     = self.local_head(x_sparse)
        local_loss = F.cross_entropy(logits, labels)
        loss_val   = local_loss.detach().item()

        _loss   = local_loss
        _active = torch.tensor(max(density, 0.01))
        _opt    = self.opt
        _fw     = self.fc.weight

        def _task():
            _opt.zero_grad(); _loss.backward()
            with torch.no_grad():
                if _fw.grad is not None: _fw.grad.mul_(_active)
            _opt.step()

        self.queue.push(_task)
        return loss_val, density

    def flush(self): self.queue.flush()


class FLUSparseLazyNet(nn.Module):
    def __init__(self, in_f=2, hidden=27, n_classes=2, threshold=0.15, queue_capacity=4):
        super().__init__()
        kw = dict(n_classes=n_classes, threshold=threshold, queue_capacity=queue_capacity)
        self.l1   = FLULocalLayer(in_f,   hidden, layer_idx=0, **kw)
        self.l2   = FLULocalLayer(hidden, hidden, layer_idx=1, **kw)
        self.l3   = FLULocalLayer(hidden, hidden, layer_idx=2, **kw)
        self.head = nn.Linear(hidden, n_classes)
        flu_owen_init_linear(self.head, seed_rank=3)

    def forward(self, x):
        return self.head(self.l3(self.l2(self.l1(x))))

    def local_steps(self, labels):
        return [l.local_step(labels) for l in [self.l1, self.l2, self.l3]]

    def flush_all(self):
        for l in [self.l1, self.l2, self.l3]: l.flush()

    def queue_stats(self):
        return {f"l{i+1}": [self.l1, self.l2, self.l3][i].queue.stats() for i in range(3)}


def train_sparse_lazy(model, Xtr, ytr, Xte, yte, epochs=150, lr=5e-3):
    global_opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    losses, accs, local_log, density_log = [], [], [], []
    for _ in range(epochs):
        model.train()
        logits      = model(Xtr)
        global_loss = F.cross_entropy(logits, ytr)
        global_opt.zero_grad(); global_loss.backward(); global_opt.step()
        stats = model.local_steps(ytr)
        model.flush_all()
        losses.append(global_loss.detach().item())
        local_log.append([s[0] for s in stats])
        density_log.append([s[1] for s in stats])
        with torch.no_grad():
            model.eval()
            accs.append((model(Xte).argmax(1)==yte).float().mean().item())
    return losses, accs, local_log, density_log


def run_sparse_lazy(Xtr, ytr, Xte, yte):
    print("=" * 62)
    print("  FLU-SPARSE-LAZY NET — now with FLU-Owen init (layer_idx seeds)")
    print("=" * 62)
    torch.manual_seed(2)
    model = FLUSparseLazyNet(hidden=27, threshold=0.15, queue_capacity=4)

    t0 = time.time()
    losses, accs, local_log, density_log = train_sparse_lazy(
        model, Xtr, ytr, Xte, yte, epochs=150)
    elapsed = time.time() - t0

    print(f"  Training time  : {elapsed:.2f}s")
    print(f"  Global loss @ep1   : {losses[0]:.4f}")
    print(f"  Global loss @ep75  : {losses[74]:.4f}")
    print(f"  Global loss @ep150 : {losses[-1]:.4f}")
    print(f"  Final accuracy     : {accs[-1]:.3f}")
    print()
    print("  Local layer CE losses @ ep150:")
    for i, (ll, d) in enumerate(zip(local_log[-1], density_log[-1])):
        print(f"    l{i+1}  local_loss={ll:.4f}  density={d:.3f}  sparsity={1-d:.3f}")
    print()
    print("  Microtask queue stats:")
    for name, s in model.queue_stats().items():
        print(f"    {name}  pushed={s['pushed']}  flushed={s['flushed']}")
    print()
    return {"losses": losses, "accs": accs}


# ── 6.  Summary ───────────────────────────────────────────────────────────────

def print_summary(mlp_res, lazy_res):
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    rows = [
        ("MLP  Kaiming  (full backprop)",      mlp_res["kaiming"]["losses"][-1],  mlp_res["kaiming"]["accs"][-1]),
        ("MLP  FLU-int  (full backprop)",       mlp_res["flu"]["losses"][-1],      mlp_res["flu"]["accs"][-1]),
        ("MLP  FLU-Owen (full backprop)",       mlp_res["flu_owen"]["losses"][-1], mlp_res["flu_owen"]["accs"][-1]),
        ("FLU-Owen + Sparse-Lazy (decoupled)",  lazy_res["losses"][-1],            lazy_res["accs"][-1]),
    ]
    print(f"  {'Method':<42}  {'Loss':>7}  {'Acc':>6}")
    print(f"  {'-'*42}  {'-'*7}  {'-'*6}")
    for name, loss, acc in rows:
        print(f"  {name:<42}  {loss:>7.4f}  {acc:>6.3f}")
    print()
    print("  FLUOwenInitializer key properties (vs FLU integer init):")
    print("  - Raw std bounded in [0, 0.289] — ~3-4% of FLU integer raw std")
    print("  - Source: APN-Owen scrambled [0,1)^d field, centred to [-0.5,0.5)")
    print("  - Guarantee basis: DN2-P1 (Latin), DN2-P2 (net-t-value) PROVEN")
    print("  - Each layer uses a distinct seed_rank → independent APN scrambles")
    print("  - Normalised to unit variance for drop-in compatibility")
    print("=" * 62)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    smoke_test()
    variance_audit()
    Xtr, ytr, Xte, yte = make_data(n=800)
    mlp_res  = run_mlp_comparison(Xtr, ytr, Xte, yte)
    lazy_res = run_sparse_lazy(Xtr, ytr, Xte, yte)
    print_summary(mlp_res, lazy_res)
