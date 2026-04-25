import os

from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from .cuda_backend import is_cuda_available, QJLSketch, QJLKeyQuantizer
from .vulkan_backend import is_vulkan_available
from .isoquant import IsoQuantMSE, IsoQuantProd
from .planarquant import PlanarQuantMSE, PlanarQuantProd
from .rotorquant import RotorQuantMSE, RotorQuantProd, RotorQuantKVCache
from .literatiquant import (
    LiteratiQuantMSE, LiteratiQuantRotated, LiteratiQuantLinear,
    LiteratiQuantEmbedding, LiteratiQuantKVCache,
    literati_replace, export_literati_to_gguf_tensors,
)
from .clifford import geometric_product, make_random_rotor, rotor_sandwich

# IsoQuant is the recommended default (5.8x faster, same quality)
QuantMSE = IsoQuantMSE
QuantProd = IsoQuantProd

_VALID_BACKENDS = {"auto", "vulkan", "cuda", "pytorch"}


def select_backend(backend: str | None = None, request_vulkan: bool = False):
    """
    Deterministic backend selection policy.

    Order:
      1) explicit backend override (function arg or TURBOQUANT_BACKEND)
      2) Vulkan if requested and available
      3) CUDA if available
      4) PyTorch fallback

    Returns:
      (backend_name, reason_message)
    """
    env_override = os.environ.get("TURBOQUANT_BACKEND", "").strip().lower()
    explicit = (backend or env_override or "").strip().lower()
    if explicit:
        if explicit not in _VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend override '{explicit}'. "
                f"Supported values: {sorted(_VALID_BACKENDS)}"
            )
        if explicit != "auto":
            if explicit == "vulkan":
                if is_vulkan_available():
                    return "vulkan", "explicit override selected Vulkan backend"
                raise RuntimeError("Explicit backend override requested Vulkan, but Vulkan is unavailable")
            if explicit == "cuda":
                if is_cuda_available():
                    return "cuda", "explicit override selected CUDA backend"
                raise RuntimeError("Explicit backend override requested CUDA, but CUDA is unavailable")
            return "pytorch", "explicit override selected PyTorch backend"

    if request_vulkan:
        if is_vulkan_available():
            return "vulkan", "Vulkan requested and available"

    if is_cuda_available():
        if request_vulkan:
            return "cuda", "Vulkan requested but unavailable; falling back to CUDA"
        return "cuda", "CUDA available"

    if request_vulkan:
        return "pytorch", "Vulkan requested but unavailable; CUDA unavailable; falling back to PyTorch"
    return "pytorch", "CUDA unavailable; using PyTorch fallback"

# Triton kernels (optional, requires triton >= 3.0 and an active GPU driver)
try:
    from .triton_planarquant import (
        triton_planar2_fused,
        triton_planar2_quantize,
        triton_planar2_dequantize,
    )
    from .fused_planar_attention import (
        triton_fused_planar_quantize_attend,
        triton_planar_cached_attention,
        pre_rotate_query_planar,
        PlanarQuantCompressedCache,
    )
    from .triton_isoquant import (
        triton_iso_full_fused,
        triton_iso_fast_fused,
    )
    from .triton_literatiquant import (
        triton_literati_fused,
        triton_literati_quantize,
        triton_literati_dequantize,
    )
    from .triton_kernels import (
        triton_rotor_sandwich,
        triton_rotor_full_fused,
        triton_rotor_inverse_sandwich,
        triton_fused_attention,
        pack_rotors_for_triton,
    )
    _triton_available = True
except (ImportError, RuntimeError):
    _triton_available = False
