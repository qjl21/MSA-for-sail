# Contributing to MSA

Thank you for your interest in contributing to **MSA** (MiniMax Sparse
Attention, package `msa_for_sail`)! This document covers the contribution
process, what we look for in a change, and the coding conventions we follow.

It deliberately does not repeat setup details: environment requirements,
install, test commands, benchmark invocations, and the repository layout are
in [README.md](README.md).

## Before You Start

- **Open an issue first** for significant changes, new kernel variants, or
  anything touching the public API, so we can align on the approach before you
  invest the time.
- **Check existing issues** to see whether someone is already working on it.
- **License**: the project is MIT. Self-authored files carry an SPDX header
  (see [File headers and licensing](#file-headers-and-licensing)); preserve
  existing copyright and provenance notices, and respect the licenses of the
  third-party code you touch — [LICENSE](LICENSE), [NOTICE](NOTICE),
  [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES).
- **Security issues**: do not open a public issue. Report them privately to
  the maintainers.
- **Hardware matters**: these are GPU/PPU kernels. Most changes cannot be
  validated by reading code alone, so plan for time on a real device and be
  explicit about what you actually ran.

## Getting Started

Set up once: clone with `--recursive` (the ACTLIZE submodule is required for
compilation) and do an editable install, per
[Install](README.md#install); for PPU, also follow
[Build and Test for PPU](README.md#build-and-test-for-ppu).

Then, for each change:

1. **Create a branch.** Prefer `<type>/<short-description>` in lowercase with
   hyphens, e.g. `fix/topk-tie-break`, `perf/sparse-prefill-810e`,
   `docs/csr-metadata`.
2. **Make the change**, keeping it focused on one problem.
3. **Validate** with a fresh JIT cache — stale artifacts are the most common
   reason a kernel change appears to do nothing:

   ```bash
   export MINFER_FMHA_CACHE_DIR=/tmp/fmha_my_change   # csrc JIT cache
   ```

4. **Open a PR** following [Pull Request Guidelines](#pull-request-guidelines).

## What We're Looking For

- **Bug fixes** with a reproducer — wrong results, bad shapes, illegal memory
  access, plan/run mismatches, JIT build failures.
- **Numerical accuracy fixes**, especially ones that let a test pass at a
  *tighter* tolerance rather than a looser one.
- **Performance improvements** backed by measurements from
  `benchmarks/bench_sparse_attention_ops.py`.
- **Hardware and dtype coverage**: new PPU architectures, new quantization
  paths (FP8), decode and prefill shapes.
- **Tests**: regression cases for bugs you hit, and coverage for shapes or
  dtypes that are currently untested.
- **Documentation**: README corrections, docstrings, kernel comments, usage
  examples.

## Making Your Contribution Successful

### Scope and Focus

- One issue or feature per PR; split large changes into reviewable pieces.
- Do not mix a fix with unrelated reformatting, renaming, or refactoring —
  it hides the actual change and makes upstream merges of derived code harder.

### Quality Standards

- The change compiles cleanly, including every JIT variant it affects, not
  just the one you exercised.
- You ran the relevant test tiers ([Test](README.md#test)) on real hardware
  and can say which device.
- Performance claims come with the command, the device, and before/after
  numbers.
- New behavior comes with a test in the right tier: fast shape-level checks in
  `tests/smoke/`, API contracts and multi-stage pipelines in
  `tests/integration/`, numerical correctness against a reference in
  `tests/regression/`.
- Public API changes were discussed first (see below).

### Common Challenges

These need extra discussion before we can accept them:

- **Public API changes.** `msa_for_sail.api` is the stable entry point;
  everything else is internal and may change without notice. New arguments,
  changed defaults, different return shapes, or altered dtype/layout contracts
  affect downstream callers, so open an issue to agree on the design first.
- **Vendored and derived code.** `python/msa_for_sail/csrc/include/` and the
  `actlize/` submodule derive from FlashInfer, CUTLASS, TensorRT-LLM, and
  T-Head ACTLIZE. Wide reformatting or renaming there costs us future upstream
  merges; keep diffs minimal and preserve provenance comments.
- **Architecture-specific code.** Some paths only compile on certain devices —
  sparse FP8 requires PPU0015 and is gated with
  `static_assert(!IsFP8 || __HGGC_ARCH__ == 150, ...)`. Contributions targeting
  an architecture we cannot test on need a clear maintenance plan.
- **Numerical tolerance changes.** Loosening a threshold to make a failing test
  pass is a red flag. Explain the error source and why the new bound is correct.
- **Workarounds for external problems.** If the change works around a
  toolchain, driver, or SDK issue (`nvcc`, PPU SDK), let's discuss
  the root cause first — we may prefer to fix or document it upstream.
- **Incomplete implementations.** Partial variants that silently fall back or
  only handle some shapes are hard to maintain. If you need help finishing one,
  say so on the issue.

## New Features

For substantial work — a new kernel variant, quantization path, scheduling
strategy, or architectural change — we follow a structured process:

1. **Open an issue** describing the feature, the use case, and the target
   hardware.
2. **Discussion**: we confirm it fits the project direction and does not break
   the public support contract.
3. **Design note**: outline the approach, the affected stack (csrc JIT),
   a test and validation plan, and expected performance impact.
4. **Review**: maintainers review the design.
5. **Implement**: we recommend waiting until the review concludes.
6. **PR**: reference the original issue and design note.

## Code Style

There is no repo-wide formatter configuration — match the surrounding code.

### General

- Python: 4 spaces, no tabs. Type hints on public functions.
- CUDA / C++: 2 spaces, no tabs; K&R braces; clang-format-style continuation
  alignment as already used in `csrc/`.
- Keep lines around 100 characters where practical.
- No trailing whitespace; no commented-out code.

### Naming

- Python: `snake_case` functions and variables, `CamelCase` classes,
  `_leading_underscore` for module-private helpers. Input checks at the API
  boundary live in `_validate_*` helpers.
- C++ / CUDA: `CamelCase` types, `snake_case` locals, following the existing
  headers.
- In files derived from upstream projects, keep upstream identifiers so diffs
  against the source stay readable.
- Ported runtime calls use the PPU names (`hggcStream_t`, `hggcMemcpyAsync`,
  `__HGGC_ARCH__`), not their CUDA spellings.

### Kernels

- Keep register pressure in mind for performance-critical kernels, and avoid
  warp divergence where possible.
- Document launch configuration, tile sizes, and occupancy assumptions where
  they are not obvious from the code.
- Fail loudly on unsupported combinations — a `static_assert` or an early
  error beats silently wrong numbers.
- Remember that csrc variants are generated from `csrc/*.cu.jinja` across
  dtype × tile × sparse × page × split_kv × pack_factor: a new variant usually
  needs the template, the dispatch in `jit.py`, and the plan/run wiring in
  `api.py` changed together.

### File headers and licensing

New self-authored files start with an SPDX block naming you as copyright
holder:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 Your Name or Organization
# SPDX-License-Identifier: MIT
```

When you substantially modify an existing file, add your line and keep the ones
already there. Never delete or rewrite upstream provenance comments. If you add
a new vendored or derived component, update [NOTICE](NOTICE),
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES), and the third-party table in
[README.md](README.md#third-party-licenses) in the same PR.

### Comments

Explain *why*, not *what*: hardware constraints, numerical assumptions,
invariants, tile/register trade-offs, and workarounds with a pointer to the
underlying problem.

## Pull Request Guidelines

Use a clear, descriptive title. A `feat:`, `fix:`, `perf:`, `test:`, or
`docs:` prefix is welcome but not required.

In the description, cover:

- **What and why** — the problem, the approach, and any limitations.
- **Validation** — which test tiers you ran, on which device and architecture
  (e.g. PPU0010 / 810E, PPU0015 / 890), and which JIT cache you used.
- **Performance** — for kernel or scheduling changes, the benchmark command,
  dtype, shapes, top-k, TP, output mode, and before/after numbers. Call out any
  shape that regresses.
- **Breaking changes** — anything affecting the stable API surface, with
  migration notes.
- **Gaps** — what you could not test and why.

Before submitting:

- Rebase onto the latest target branch if you have fallen behind or conflict.
- Keep build artifacts, JIT caches, profiles, and generated benchmark output
  out of the commit; `.gitignore` covers the common cases.

## Code Review Process

1. **Maintainer review** — we read the diff, the validation evidence, and the
   numbers.
2. **Discussion** — we may ask questions or request changes.
3. **Iteration** — push updates to the same branch.
4. **Merge** — once approved and validated.

Review focuses on correctness, numerical accuracy, performance, and
maintainability rather than style, as long as the code matches its
surroundings.

## Getting Help

- Comment on the related issue or PR, or open a new issue for bug reports and
  feature requests.
- Read [README.md](README.md) and
  [`docs/MiniMaxSparseAttention.pdf`](docs/MiniMaxSparseAttention.pdf) for the
  algorithm itself.

Thank you for helping make MSA better — happy coding!
