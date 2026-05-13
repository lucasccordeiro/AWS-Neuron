# ESBMC upstream issue (filed and resolved)

> Filed as https://github.com/esbmc/esbmc/issues/4510 (labels: `python`, `bug`).
> Fixed by https://github.com/esbmc/esbmc/pull/4511 (merged). The
> initialised-variable workarounds in the `interpolate_*` kernels have been
> removed. This file is kept for the historical reproducer.

## Title

`[python] Bare variable annotation inside a while-loop body crashes the Python frontend with a JSON type_error`

## Labels

`python`, `bug`

## Affected version

`ESBMC version 8.2.0 64-bit aarch64 macos`. Likely reproduces on a recent main.

## Summary

A standalone variable type annotation without an initializer
(`name: int`) placed inside a `while`-loop body causes the Python
frontend to crash at the *Converting* stage with an uncaught
`nlohmann::json_abi_v3_11_3::detail::type_error`. The same annotation
works fine at module scope or at the top of a function body; only
placement inside the loop triggers the crash.

## Minimal reproducer

```python
def f(n: int) -> int:
    h: int = 0
    while h < 3:
        x: int         # ← bare annotation, no initializer
        if 5 < n:
            x = 5
        else:
            x = n
        h = h + 1
    return 0

f(10)
```

### Observed

```
$ esbmc repro.py
Target: 64-bit little-endian aarch64-unknown-macos with esbmclibc
Parsing repro.py
Converting
libc++abi: terminating due to uncaught exception of type nlohmann::json_abi_v3_11_3::detail::type_error
```

(exit status: SIGABRT)

### Workaround

Initialise the variable:

```python
x: int = 0    # works — verification proceeds normally
```

The verification succeeds in this form. The crash is specific to the
no-initializer placement inside the loop.

## Impact

The crash is silent at conversion time — no source location, no symbol
name, no traceback through the Python AST. A user porting non-trivial
Python code (in our case, AWS NKI kernels with conditional integer
bindings inside iteration loops) sees the verifier abort and has to
bisect their source to find the offending line. The workaround is
mechanical but the diagnostic is unhelpful.

We hit this while building the NKI/ESBMC PoC at
https://github.com/lucasccordeiro/AWS-Neuron — see the
`kernels/interpolate_bilinear.py` port and the comment near the
top about the workaround.

## Suggested fix area

The exception type (`nlohmann::json_abi`) suggests the crash is in
whichever frontend pass serialises or queries the Python AST through
the nlohmann::json carrier. Likely a key lookup that assumes an
initializer node is present. A bounded type check before the access,
or a clean error message ("annotation without initializer in loop body
is not yet supported") would be a clear improvement over the silent
abort.

A reduced repro under `regression/python/` would be straightforward to
add — happy to provide a PR if it's useful.
