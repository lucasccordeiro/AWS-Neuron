# DRAFT: third ESBMC upstream issue

> Status: not yet filed. Review before posting to
> https://github.com/esbmc/esbmc/issues/

## Title

`[python] Segfault when entry script and an imported submodule share an unqualified module name`

## Labels

`python`, `bug`

## Affected version

`ESBMC version 8.2.0 64-bit aarch64 macos` (post-#4509/#4510 fixes; reproduces against the binary built from those merged PRs).

## Summary

When the script passed to ESBMC has the same unqualified name as a module it imports via `from kernels.<name> import ...`, the Python frontend segfaults during conversion. The crash is silent — no source location, no traceback — so anyone hitting it sees only "Parsing X" followed by SIGSEGV.

## Minimal reproducer

Layout:

```
spike/
├── stubs.py
├── kernels/
│   └── tensor_add.py
└── tensor_add.py          # entry script, same basename as kernels/tensor_add.py
```

### Files

`spike/stubs.py`:

```python
class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1

def make_tile(d0: int, d1: int) -> Tile:
    assert d0 > 0
    assert d1 > 0
    return Tile(d0, d1)
```

`spike/kernels/tensor_add.py`:

```python
from stubs import Tile, make_tile

def nki_tensor_add(a: Tile, b: Tile) -> Tile:
    assert a.d0 == b.d0
    return make_tile(a.d0, a.d1)
```

`spike/tensor_add.py` (the entry script with the conflicting basename):

```python
from stubs import make_tile
from kernels.tensor_add import nki_tensor_add

a = make_tile(4, 8)
c = nki_tensor_add(a, a)
assert c.d0 == 4
```

### Observed

```
$ esbmc tensor_add.py
Target: 64-bit little-endian aarch64-unknown-macos with esbmclibc
Parsing tensor_add.py
Segmentation fault: 11
```

Exit status 139.

### Workaround

Rename the entry script so its unqualified name differs from the imported submodule. With `spike/tensor_add.py` renamed to `spike/run_tensor_add.py` (and the kernel module unchanged), verification succeeds cleanly:

```
$ esbmc run_tensor_add.py
...
VERIFICATION SUCCESSFUL
```

## Impact

For library-style verification artefacts that pair an entry script with a same-named kernel/source module, the only way to verify is to invent a disambiguating name. We hit this in the NKI/ESBMC PoC at https://github.com/lucasccordeiro/AWS-Neuron — every entry script carries a `verify_` prefix to keep its unqualified name distinct from `kernels.<same_name>`.

## Suggested fix area

The conversion stage that records modules in the module table probably collides the entry-script symbol with `kernels.<name>` because both have the same final path component. A separate canonical name for the entry script (e.g. `__main__` or the full path) would avoid the collision. Whatever the cause, a clean error message ("entry-script basename collides with imported submodule") would be a strict improvement over the silent segfault.

Happy to provide a reduced repro under `regression/python/` if useful.
