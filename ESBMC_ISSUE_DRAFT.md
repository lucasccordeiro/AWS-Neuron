# ESBMC upstream issue (filed)

> Filed as https://github.com/esbmc/esbmc/issues/4509 (labels: `python`, `bug`).
> This file is kept for reference; track status on the issue itself.

---

## Title

`[python] Transitive imports through an intermediate module fail to resolve symbols`

## Labels

`python`, `bug`

## Affected version

`ESBMC version 8.2.0 64-bit aarch64 macos` (also reproduces against a
recent main; specify exact commit when filing).

## Summary

The Python frontend resolves direct imports from the script's directory
(`harness/main.py` -> `stubs.py` works), but does **not** resolve
*transitive* imports: when `harness/main.py` imports a kernel from
`kernels/kernel.py`, and `kernels/kernel.py` in turn imports symbols from
`stubs.py`, calls to those symbols inside the kernel fail at conversion or
symex time. Several import patterns produce several distinct errors; all
of them block the natural three-layer
`stubs.py + kernels/<kernel>.py + harness/<harness>.py` decomposition.

## Reproducer

Layout:

```
spike/
├── stubs.py
├── kernels/
│   └── kernel.py
└── harness/
    ├── stubs.py        # copy of ../stubs.py, sibling to harness entry
    ├── kernels/
    │   └── kernel.py   # copy of ../kernels/kernel.py
    └── main.py
```

The "sibling" copies under `harness/` are needed because ESBMC searches only
the entry script's directory for modules. The bug is then in *transitive*
resolution through the (now-co-located) `kernels/kernel.py`.

### Reproducer files

`harness/stubs.py`:

```python
class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1

def make_tile(d0: int, d1: int) -> Tile:
    assert d0 > 0
    assert d1 > 0
    return Tile(d0, d1)

def check_shape(a: Tile, b: Tile) -> None:
    assert a.d0 == b.d0
    assert a.d1 == b.d1
```

`harness/kernels/kernel.py`:

```python
from stubs import Tile, make_tile, check_shape

def add_kernel(a: Tile, b: Tile) -> Tile:
    check_shape(a, b)
    return make_tile(a.d0, a.d1)
```

`harness/main.py`:

```python
from stubs import make_tile
from kernels.kernel import add_kernel

a = make_tile(4, 8)
b = make_tile(4, 8)
c = add_kernel(a, b)
assert c.d0 == 4
assert c.d1 == 8
```

### Observed

`esbmc harness/main.py` reports:

```
[Counterexample]
State 1 file harness/kernels/kernel.py line 4 column 4 function add_kernel
  Violated property:
    Unsupported function 'check_shape' is reached
    0

VERIFICATION FAILED
```

`check_shape` is defined in `harness/stubs.py` and is imported into
`harness/kernels/kernel.py`; calling it inside `add_kernel` produces
*Unsupported function 'check_shape' is reached*. `make_tile` (also from
`stubs.py`, also called by the kernel) is silently treated as
unsupported as well — but only the first such call shows up in the
counterexample.

### Other patterns tried, and their failures

1. `from stubs import *` in both `kernel.py` and `main.py` →
   `ERROR: Function "make_tile" not found (harness/main.py line 4)`
2. `import stubs` + qualified `stubs.check_shape(...)` calls →
   `ERROR: Variable 'kernels' is not defined at line 6.` (when main does
   `import kernels.kernel` and later writes `kernels.kernel.add_kernel(...)`)
3. Same as (2) but with `from kernels.kernel import add_kernel` in
   `main.py` and `import stubs` in `kernel.py` →
   `ERROR: function call: argument "...stubs.py@F@check_shape@a" type
   mismatch: got pointer, expected struct` — class identity (`Tile`)
   appears not to propagate across the import boundary, so `check_shape`'s
   parameter type does not unify with the caller's `Tile` instance.

### Workaround

Inline every dependency into the entry module. The version that works:

```python
# harness/kernels/kernel.py — contains BOTH the stub class and the kernel
class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1

def make_tile(d0, d1) -> Tile: ...
def add_kernel(a, b) -> Tile: ...
```

```python
# harness/main.py — imports everything from the bundled kernel module
from kernels.kernel import Tile, make_tile, add_kernel
...
```

This pattern verifies cleanly (27 VCCs, `VERIFICATION SUCCESSFUL`),
demonstrating that single-hop imports of class types do work; the failure
is specific to multi-hop chains.

## Impact

Without transitive imports, any non-trivial verification artefact that
wants to share a stub / model library across multiple kernels has to
either:

- inline the stubs into every kernel file (defeats the "single source of
  truth" property of the stub library, which is itself part of the
  trusted base), or

- use a build-time concatenation step that bundles stub + kernel + harness
  into one ESBMC-ready file per target.

The latter is the workaround currently used in the NKI/ESBMC PoC at
https://github.com/lucasccordeiro/AWS-Neuron (see `build.py` /
`Makefile`).

## Suggested fix area

`src/python-frontend/module_manager.cpp` (or the equivalent that resolves
`from X import Y` symbol bindings): when processing a function body, the
symbol resolver should consult the *importer's* import table when
encountering an unbound name, not only the entry module's.

I'm happy to provide a more reduced repro or test on a specific commit if
helpful.
