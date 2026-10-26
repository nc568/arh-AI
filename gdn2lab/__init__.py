"""gdn2lab — small, from-scratch Gated DeltaNet-2 models for free Colab GPU/TPU.

The implementation follows the equations of the paper
"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
(arXiv:2605.22791v1), but shares no source code with the NVIDIA reference
implementation (which is distributed under the non-commercial NVIDIA Source
Code License-NC).  No Triton kernels are used: everything is plain PyTorch so
that the same model trains on CUDA, TPU (torch-xla) and CPU.
"""

__version__ = "0.1.0"

from .config import LabConfig, ModelConfig, DataConfig, TrainConfig, ExportConfig  # noqa: F401
