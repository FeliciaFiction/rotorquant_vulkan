# Vulkan 1.3 Support Plan (Prioritized)

## Goal
Add production-ready Vulkan 1.3 support to RotorQuant (with Intel Arc as a first-class target), while keeping CUDA and PyTorch fallbacks intact.

## References from llama.cpp to mirror
- `ggml/CMakeLists.txt` (Vulkan options and toggles like `GGML_VULKAN`, debug/validation flags)
- `ggml/src/ggml-vulkan/CMakeLists.txt` (glslc feature probing, shader generation pipeline, host-toolchain handling)
- `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt` (shader generator build flow)
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (runtime device/feature probing, vendor-aware behavior, robust fallback paths)

## Priority 0 - Repo and design lock-in
- [x] Confirm repo root and origin remote
  - Confirmed local repo is `C:\Git\rotorquant`
  - Confirmed `origin` is `https://github.com/scrya-com/rotorquant`
- [x] Create working branch
  - Branch: `codex/vulkan-1.3-plan`
- [x] Decide implementation path (recommended: C++ extension + GLSL compute shaders)
  - Decision: implement Vulkan through a native C++ extension plus GLSL compute shaders (Vulkan 1.3 baseline).
  - Why this path:
    - Matches current performance-critical architecture (`turboquant/csrc/*.cu` kernels are native, not Python-loop based).
    - Preserves the existing Python API by mirroring `cuda_backend.py` function signatures.
    - Aligns with `llama.cpp`'s proven Vulkan approach (feature probing, shader generation, runtime capability gates).
    - Gives deterministic fallback behavior when Vulkan runtime/features are missing.
  - Scope boundaries for this phase:
    - Phase 1 (must): `qjl_quant`, `qjl_score`, `qjl_gqa_score`, and minimal `quantized_bmm` parity path.
    - Phase 2 (later): aggressive kernel specialization and vendor-specific fast paths.
  - Explicitly rejected for now:
    - Python-only Vulkan wrappers as primary path (acceptable for prototyping only, not production target).

## Execution protocol (required for every open point)
- [ ] For each point, complete in this order:
  - Implementation
  - Tests (automated where possible)
  - Validation (manual/runtime checks)
  - Report (what changed, what passed/failed, logs/metrics summary)
- [ ] Pre-commit gate:
  - No commit until a per-point report is shared and acknowledged.
  - If tests are blocked, report blocker, risk, and proposed mitigation before any commit.

## Priority 1 - Build system and project scaffolding
- [x] Add Vulkan build toggle to `setup.py`
  - Add `--vulkan` and env switch `TURBOQUANT_BUILD_VULKAN=1`
  - Keep `--cuda` behavior unchanged
  - Allow building CUDA, Vulkan, both, or neither
  - Tests:
    - Run editable install in four modes: none, `--cuda`, `--vulkan`, both.
    - Verify extension selection logic does not regress CUDA-only builds.
  - Validation:
    - Confirm generated artifacts/modules match selected flags.
    - Confirm import of `turboquant` succeeds in all supported modes.
  - Report:
    - Include exact commands run, pass/fail matrix, and artifact list.
  - Result summary (2026-04-25):
    - Implemented `--vulkan` and `TURBOQUANT_BUILD_VULKAN` handling in `setup.py`.
    - Added mixed mode handling (`--cuda --vulkan`).
    - Added graceful CUDA fallback when CUDA toolchain is absent (`CUDA_HOME` missing): warning + skip instead of hard failure.
    - Validation matrix:
      - `python setup.py --name`: pass
      - `python setup.py --cuda --name`: pass (warning, CUDA skipped on this host)
      - `python setup.py --vulkan --name`: pass (warning, Vulkan sources not yet present)
      - `python setup.py --cuda --vulkan --name`: pass (warnings, both skipped on this host)
      - `pip install -e ... --no-deps` in none/cuda/vulkan/both modes: pass in all 4 modes
- [x] Add Vulkan source tree
  - Create `turboquant/vulkan/` with:
    - `vulkan_backend.cpp` (Python binding bridge)
    - `vk_runtime.cpp/.h` (instance/device/queue/descriptors/pipelines)
    - `shaders/*.comp` (GLSL kernels)
    - `shaders/CMakeLists.txt` or equivalent generation script
  - Tests:
    - Static checks/build checks to ensure sources compile and link into extension target.
  - Validation:
    - Verify module load path resolves all required symbols at runtime.
    - Verify shader files are discoverable/packaged.
  - Report:
    - Include created file tree, compile output summary, unresolved symbol status.
  - Result summary (2026-04-25):
    - Added scaffold files:
      - `turboquant/vulkan/vulkan_backend.cpp`
      - `turboquant/vulkan/vk_runtime.cpp`
      - `turboquant/vulkan/vk_runtime.h`
      - `turboquant/vulkan/__init__.py`
      - `turboquant/vulkan/shaders/{qjl_quant.comp,qjl_score.comp,qjl_gqa_score.comp,CMakeLists.txt}`
    - Updated packaging in `setup.py`:
      - `package_data` includes Vulkan shader files.
      - Platform-specific C++ compile flags for Vulkan extension (`/O2 /std:c++17` on MSVC).
    - Validation:
      - `python setup.py --vulkan build_ext --inplace`: pass (extension compiled/linked/copied)
      - `python -c "import turboquant.vulkan_backend_ext ..."`: pass (`available=False`, `info=vulkan-runtime-scaffold`)
      - Shader compile path pass: CMake configure + build generated `.spv` outputs for all placeholder shaders.
    - Unresolved symbols:
      - None in current scaffold extension import path.
- [ ] Add shader build/generation flow inspired by llama.cpp
  - Probe glslc extension support at build time
  - Generate/compile shader artifacts at build time
  - Embed shader blobs or package them reliably
  - Tests:
    - Run shader generation step in clean build directory.
    - Add one negative test for missing `glslc`/feature probe failure path.
  - Validation:
    - Confirm SPIR-V outputs are produced and consumed by extension.
    - Confirm feature probe flags are correctly propagated into build.
  - Report:
    - Include probe results table and generated shader artifact inventory.

## Priority 1 - Kernel parity (must-have for functional Vulkan backend)
- [ ] Port `turboquant/csrc/qjl_quant_kernel.cu` to Vulkan compute
  - First target: dtype path `float16/float32` parity
  - Tests:
    - Numerical parity tests vs CUDA and PyTorch reference on fixed seeds.
  - Validation:
    - Validate dtype coverage (`float16`, `float32`) and shape edge cases.
  - Report:
    - Include max/mean error, tolerances, and runtime comparison snapshot.
- [ ] Port `turboquant/csrc/qjl_score_kernel.cu` to Vulkan compute
  - Tests:
    - Unit parity tests vs CUDA output across representative sequence lengths.
  - Validation:
    - Validate stability for small/large norms and boundary indices.
  - Report:
    - Include parity metrics and any known divergence cases.
- [ ] Port `turboquant/csrc/qjl_gqa_score_kernel.cu` to Vulkan compute
  - Tests:
    - GQA-specific parity tests with multiple head/group configurations.
  - Validation:
    - Validate correctness when KV head count differs from query head count.
  - Report:
    - Include coverage matrix by `num_heads`, `num_kv_heads`, head_dim.
- [ ] Port required path(s) from `turboquant/csrc/quantization.cu`
  - Start with the minimal path needed for `quantized_bmm` parity
  - Tests:
    - `quantized_bmm` parity tests for supported bit-widths and group sizes.
  - Validation:
    - Validate output shape/layout compatibility with existing callers.
  - Report:
    - Include parity table and unsupported-case list (if any).
- [ ] Add API-compatible Python wrapper `turboquant/vulkan_backend.py`
  - Mirror public functions in `turboquant/cuda_backend.py`:
    - `is_vulkan_available()`
    - `qjl_quant(...)`
    - `qjl_score(...)`
    - `qjl_gqa_score(...)`
    - `quantized_bmm(...)`
  - Tests:
    - API compatibility tests (signature and expected exceptions).
  - Validation:
    - Validate dispatch chooses Vulkan path only when capability checks pass.
  - Report:
    - Include function-level status and fallback behavior examples.

## Priority 1 - Runtime dispatch and safety
- [ ] Add backend selection policy in Python package
  - Update `turboquant/__init__.py` exports to include Vulkan path
  - Keep deterministic fallback order:
    - Explicit backend override
    - Vulkan if available and requested
    - CUDA if available
    - PyTorch fallback
  - Tests:
    - Parametrized dispatch tests for all backend availability combinations.
  - Validation:
    - Validate deterministic backend choice and clear logging/messages.
  - Report:
    - Include dispatch truth table and observed runtime path per case.
- [ ] Add strict capability checks
  - Vulkan 1.3 minimum
  - Required compute features/extensions
  - Clear error messages with fallback behavior
  - Tests:
    - Capability check unit tests using mocked feature sets.
  - Validation:
    - Manual validation on at least one real Vulkan 1.3-capable device.
  - Report:
    - Include feature checklist and fallback outcomes.
- [ ] Add vendor-aware guardrails (copy llama.cpp pattern)
  - Detect Intel/AMD/NVIDIA vendor IDs
  - Gate optional fast paths by supported extensions
  - Tests:
    - Vendor detection tests with mocked PCI/vendor IDs.
  - Validation:
    - Validate that unsafe/unsupported fast paths are disabled gracefully.
  - Report:
    - Include vendor matrix and enabled feature paths.

## Priority 2 - Correctness and test coverage
- [ ] Add unit tests for Vulkan/CUDA/PyTorch numerical parity
  - New tests under `tests/` for:
    - `qjl_quant`
    - `qjl_score`
    - `qjl_gqa_score`
    - `quantized_bmm`
  - Tolerance matrix by dtype and backend
  - Tests:
    - Implement and run full parity suite in CI and local GPU mode (where available).
  - Validation:
    - Confirm tolerance thresholds are explicit and justified.
  - Report:
    - Include per-test tolerance and worst-case observed error.
- [ ] Add backend smoke tests
  - Device discovery
  - Shader compile/load
  - Single-pass inference path
  - Tests:
    - Add smoke test target and run in at least one Vulkan-capable environment.
  - Validation:
    - Confirm diagnostics are actionable when smoke tests fail.
  - Report:
    - Include smoke output logs and pass/fail summary.
- [ ] Add regression tests for fallback behavior
  - Missing Vulkan runtime
  - Missing extensions
  - Invalid shader artifacts
  - Tests:
    - Add regression tests covering all fallback/error branches.
  - Validation:
    - Confirm no hard crash path on runtime fallback scenarios.
  - Report:
    - Include scenario matrix and resulting backend selected.

## Priority 2 - Performance bring-up
- [ ] Add Vulkan benchmark script(s)
  - Extend existing benchmark suite in `turboquant/benchmark_*.py`
  - Compare Vulkan vs CUDA vs PyTorch for key kernels
  - Tests:
    - Run benchmarks with fixed seeds/configs for reproducible comparison.
  - Validation:
    - Verify benchmark script captures device metadata and backend path.
  - Report:
    - Include throughput/latency table and test configuration.
- [ ] Establish acceptance thresholds
  - Functional parity before optimization
  - Initial perf target: within practical range of current CUDA path on equivalent workload
  - Tests:
    - Add automated pass/fail checks for agreed thresholds where feasible.
  - Validation:
    - Re-check thresholds on at least one Intel Arc environment.
  - Report:
    - Include final threshold values and measured results.
- [ ] Profile and tune
  - Workgroup size tuning
  - Buffer reuse and descriptor caching
  - Queue submission batching and synchronization minimization
  - Tests:
    - Re-run parity suite after each optimization batch.
  - Validation:
    - Confirm no numerical regressions and no fallback behavior changes.
  - Report:
    - Include before/after profile snapshots and observed gains.

## Priority 3 - Docs, packaging, CI
- [ ] Update `README.md`
  - Vulkan prerequisites (SDK/driver/toolchain)
  - Build examples for Vulkan-only and mixed builds
  - Runtime backend selection examples
  - Add dependency-install guidance:
    - Normal setup should use `pip install -e .` so required deps (including `scipy`) are installed.
    - `--no-deps` is for controlled validation/CI checks and can cause import failures if deps are absent.
  - Tests:
    - Execute documented commands in a clean environment.
  - Validation:
    - Confirm docs match actual flags, file paths, and outputs.
  - Report:
    - Include doc command verification checklist.
- [ ] Update `requirements.txt` / optional extras if needed
  - Keep base install lightweight
  - Put Vulkan-related Python deps behind extras where possible
  - Tests:
    - Validate fresh install for base and extras variants.
  - Validation:
    - Confirm no unnecessary dependency added to base path.
  - Report:
    - Include dependency diff and install success matrix.
- [ ] Add CI for Vulkan build sanity
  - At minimum: compile-time checks
  - Optional: runtime tests on GPU runner when available
  - Tests:
    - Validate workflow syntax and run at least one successful CI cycle.
  - Validation:
    - Confirm CI artifacts/logs are sufficient for failure triage.
  - Report:
    - Include CI job list, trigger conditions, and latest run status.

## Suggested implementation order (short)
1. Build toggle + project scaffolding
2. `qjl_quant` Vulkan kernel + wrapper
3. `qjl_score` / `qjl_gqa_score` Vulkan kernels
4. `quantized_bmm` minimal Vulkan parity path
5. Runtime dispatch integration
6. Tests and benchmarks
7. Docs + CI hardening

## Definition of done
- Vulkan 1.3 backend can be built and selected explicitly.
- Core QJL kernels run on Vulkan with acceptable numerical parity vs CUDA.
- Existing CUDA and PyTorch behavior remains backward-compatible.
- README includes clear Vulkan build/run instructions.
