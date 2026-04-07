# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class FluxEncoderConfig:
    """Configuration for Flux encoders (T5 text encoder, CLIP text encoder, and autoencoder)."""

    t5_encoder: str | None = "google/t5-v1_1-small"
    """HuggingFace model name or local path for the T5 text encoder.

    Set to ``None`` to use precomputed T5 encodings from the dataset.
    """

    clip_encoder: str | None = "openai/clip-vit-large-patch14"
    """HuggingFace model name or local path for the CLIP text encoder.

    Set to ``None`` to use precomputed CLIP encodings from the dataset.
    """

    autoencoder_path: str | None = (
        "torchtitan/experiments/flux/assets/autoencoder/ae.safetensors"
    )
    """Autoencoder checkpoint path to load.

    Set to ``None`` to use precomputed latent statistics from the dataset.
    """

    autoencoder_shift: float | None = None
    """Optional latent shift factor used when the autoencoder is not loaded."""

    autoencoder_scale: float | None = None
    """Optional latent scale factor used when the autoencoder is not loaded."""

    empty_encodings_path: str | None = None
    """Optional folder containing ``t5_empty.npy`` and ``clip_empty.npy``.

    This is used by precomputed-encoding training recipes that still need
    empty-condition embeddings for classifier-free guidance dropout.
    """

    random_init: bool = False
    """If True, initialize encoders with random weights instead of loading pretrained weights (for testing only)."""


@dataclass(kw_only=True, slots=True)
class FluxMLPerfConfig:
    """MLPerf-specific training controls for Flux recipes."""

    enable: bool = False
    """Enable MLPerf compliance logging and convergence-based early stop."""

    target_eval_loss: float = 0.586
    """Validation loss threshold used for MLPerf convergence."""

    eval_samples: int = 262_144
    """Number of training samples between MLPerf validation runs."""


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Shared configuration for image generation sampling (used by both validation and inference)."""

    enable_classifier_free_guidance: bool = False
    """Whether to use classifier-free guidance (CFG) during image generation.

    When enabled, the model runs two forward passes per denoising step — one with
    the text prompt and one without — then interpolates the results using
    `classifier_free_guidance_scale` to produce images that more closely follow
    the prompt. This typically yields higher-quality, more prompt-adherent images
    but doubles the compute cost per sampling step.
    """

    classifier_free_guidance_scale: float = 5.0
    """Interpolation weight for classifier-free guidance during sampling.

    Higher values steer the output more strongly toward the text prompt, producing
    sharper and more prompt-adherent images, but may reduce diversity or introduce
    artifacts. Typical values range from 1.0 (no guidance) to 10.0 (strong guidance).
    Only takes effect when `enable_classifier_free_guidance` is True.
    """

    denoising_steps: int = 50
    """How many denoising steps to sample when generating an image."""


@dataclass(kw_only=True, slots=True)
class Inference:
    """Inference configuration"""

    save_img_folder: str = "inference_results"
    """Path to save the inference results"""
    prompts_path: str = "./torchtitan/experiments/flux/inference/prompts.txt"
    """Path to file with newline separated prompts to generate images for"""
    local_batch_size: int = 2
    """Batch size for inference"""
    img_size: int = 256
    """Image size for inference"""
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    """Sampling configuration for image generation"""
