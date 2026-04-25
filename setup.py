"""
Build script for TurboQuant with optional CUDA/Vulkan native extensions.

Usage:
    pip install -e .                                          # Python-only (no native extensions)
    pip install -e . --config-settings="--build-option=--cuda"    # CUDA extensions
    pip install -e . --config-settings="--build-option=--vulkan"  # Vulkan extensions
    python setup.py build_ext --inplace --cuda --vulkan            # Build requested native extensions in-place
"""

from setuptools import setup, find_packages
import os
import sys

ext_modules = []
cmdclass = {}

def _pop_flag(flag: str) -> bool:
    found = flag in sys.argv
    if found:
        sys.argv.remove(flag)
    return found


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


build_cuda = _pop_flag('--cuda') or _env_flag('TURBOQUANT_BUILD_CUDA')
build_vulkan = _pop_flag('--vulkan') or _env_flag('TURBOQUANT_BUILD_VULKAN')

_build_ext_requested = 'build_ext' in sys.argv

def _cpp_flags():
    if os.name == "nt":
        return ["/O2", "/std:c++17"]
    return ["-O3", "-std=c++17"]


def _ensure_build_extension(build_extension_cls):
    if 'build_ext' not in cmdclass:
        cmdclass['build_ext'] = build_extension_cls

if build_cuda or _build_ext_requested:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            # If we detect Blackwell (12.x), force "+ PTX" for compatibility
            if major >= 12:
                arch_list = f"{major}.{minor}+PTX"
            else:
                arch_list = f"{major}.{minor}"
            print(f"Auto-detected GPU architecture: {arch_list}")
            # Force-set the architecture environment variable so PyTorch sees it
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list

        def nvcc_flags():
            nvcc_threads = os.getenv("NVCC_THREADS", "8")
            return [
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                f"--threads={nvcc_threads}",
            ]

        csrc_dir = os.path.join(os.path.dirname(__file__), 'turboquant', 'csrc')

        try:
            ext_modules.extend([
                CUDAExtension(
                    name='turboquant.cuda_qjl_score',
                    sources=[os.path.join(csrc_dir, 'qjl_score_kernel.cu')],
                    extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": nvcc_flags()}
                ),
                CUDAExtension(
                    name='turboquant.cuda_qjl_quant',
                    sources=[os.path.join(csrc_dir, 'qjl_quant_kernel.cu')],
                    extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": nvcc_flags()}
                ),
                CUDAExtension(
                    name='turboquant.cuda_qjl_gqa_score',
                    sources=[os.path.join(csrc_dir, 'qjl_gqa_score_kernel.cu')],
                    extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": nvcc_flags()}
                ),
                CUDAExtension(
                    name='turboquant.quantization',
                    sources=[os.path.join(csrc_dir, 'quantization.cu')],
                    extra_compile_args={
                        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
                        "nvcc": nvcc_flags() + [
                            "-DENABLE_BF16",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        ]
                    }
                ),
            ])
            _ensure_build_extension(BuildExtension)
            print("CUDA extensions will be built.")
        except OSError as e:
            print(f"WARNING: CUDA build requested but CUDA toolchain is unavailable ({e}). Skipping CUDA extension build.")
    except ImportError:
        print("WARNING: torch not found, CUDA extensions will not be built.")

if build_vulkan:
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension

        vulkan_dir = os.path.join(os.path.dirname(__file__), 'turboquant', 'vulkan')
        vulkan_sources = [
            os.path.join(vulkan_dir, 'vulkan_backend.cpp'),
            os.path.join(vulkan_dir, 'vk_runtime.cpp'),
        ]
        existing_sources = [src for src in vulkan_sources if os.path.exists(src)]

        if existing_sources:
            ext_modules.append(
                CppExtension(
                    name='turboquant.vulkan_backend_ext',
                    sources=existing_sources,
                    extra_compile_args={"cxx": _cpp_flags()},
                )
            )
            _ensure_build_extension(BuildExtension)
            print("Vulkan extensions will be built.")
        else:
            print("WARNING: Vulkan build requested, but turboquant/vulkan C++ sources are not present yet. Skipping Vulkan extension build.")
    except ImportError:
        print("WARNING: torch not found, Vulkan extension build is unavailable.")

if not build_cuda and not build_vulkan:
    print("Native extension build: disabled (Python-only install).")
else:
    requested = []
    if build_cuda:
        requested.append("CUDA")
    if build_vulkan:
        requested.append("Vulkan")
    print(f"Native extension request: {', '.join(requested)}")

setup(
    name='turboquant',
    version='0.2.0',
    description='TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "turboquant.vulkan": [
            "shaders/*.comp",
            "shaders/CMakeLists.txt",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'scipy>=1.10.0',
    ],
    extras_require={
        'validate': ['transformers>=4.40.0', 'accelerate>=0.25.0', 'bitsandbytes>=0.43.0'],
    },
)
