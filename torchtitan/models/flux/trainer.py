# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.flux.configs import FluxEncoderConfig, FluxMLPerfConfig, Inference
from torchtitan.models.flux.mlperf_logging import MLPerfLogger
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.autoencoder_utils import (
    generate_unscaled_latent_from_mean_logvar,
)
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.models.flux.parallelize import parallelize_encoders
from torchtitan.models.flux.tokenizer import FluxTokenizerContainer
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    IMAGE_LATENT_SIZE_RATIO,
    pack_latents,
    PATCH_HEIGHT,
    PATCH_WIDTH,
    preprocess_data,
)
from torchtitan.trainer import Trainer
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)

from .validate import FluxValidator


class ThroughputTimer:
    """Tracks cumulative train-only and end-to-end throughput in samples/s."""

    def __init__(self, global_batch_size: int):
        self.global_batch_size = global_batch_size
        self._step_start_time: float | None = None
        self._train_start_time: float | None = None
        self._train_elapsed_time = 0.0
        self._samples = 0

    def start(self) -> None:
        now = time.perf_counter()
        self._step_start_time = now
        if self._train_start_time is None:
            self._train_start_time = now

    def stop(self) -> None:
        if self._step_start_time is None:
            raise ValueError("Throughput timer has not been started.")
        now = time.perf_counter()
        self._train_elapsed_time += now - self._step_start_time
        self._samples += self.global_batch_size
        self._step_start_time = None

    def train_throughput(self) -> float:
        if self._train_elapsed_time <= 0:
            return 0.0
        return self._samples / self._train_elapsed_time

    def combined_throughput(self) -> float:
        if self._train_start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self._train_start_time
        if elapsed <= 0:
            return 0.0
        return self._samples / elapsed

    def time_to_converge(self) -> float | None:
        if self._train_start_time is None:
            return None
        return time.perf_counter() - self._train_start_time


class FluxTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        # Overwrite parent class tokenizer
        tokenizer: BaseTokenizer.Config = (  # pyrefly: ignore [bad-override]
            field(default_factory=FluxTokenizerContainer.Config)
        )
        encoder: FluxEncoderConfig = field(default_factory=FluxEncoderConfig)
        """Configuration for Flux encoders (T5 text encoder, CLIP text encoder, and autoencoder)."""
        inference: Inference = field(default_factory=Inference)
        mlperf: FluxMLPerfConfig = field(default_factory=FluxMLPerfConfig)

    def __init__(self, config: Config):
        if config.mlperf.enable and not config.validator.enable:
            raise ValueError("Flux MLPerf mode requires validator.enable=True.")

        # Compute image token count: autoencoder downscales the image,
        # then pack_latents tiles the latent into 2×2 patches.
        # pyrefly: ignore [missing-attribute]
        img_size = config.dataloader.img_size
        ae_downscale = IMAGE_LATENT_SIZE_RATIO
        latent_side_width = img_size // ae_downscale // PATCH_WIDTH
        latent_side_height = img_size // ae_downscale // PATCH_HEIGHT
        seq_len_img = latent_side_width * latent_side_height

        # pyrefly: ignore [missing-attribute]
        seq_len_txt = config.tokenizer.max_t5_encoding_len
        config.training.seq_len = seq_len_img + seq_len_txt

        super().__init__(config)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Flux model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["fsdp", "dp_replicate"],
        )
        self.mlperf_seed = int(torch.initial_seed())

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        # load components
        assert config.model_spec is not None
        model_args = config.model_spec.model

        if config.encoder.autoencoder_path:
            self.autoencoder = load_ae(
                config.encoder.autoencoder_path,
                # pyrefly: ignore [missing-attribute]
                model_args.autoencoder_params,
                device=self.device,
                dtype=self._dtype,
                random_init=config.encoder.random_init,
            )
        else:
            self.autoencoder = None

        if (
            config.encoder.autoencoder_shift is not None
            and config.encoder.autoencoder_scale is not None
        ):
            self.autoencoder_shift_factor = config.encoder.autoencoder_shift
            self.autoencoder_scale_factor = config.encoder.autoencoder_scale
        elif self.autoencoder is not None:
            self.autoencoder_shift_factor = self.autoencoder.shift_factor
            self.autoencoder_scale_factor = self.autoencoder.scale_factor
        else:
            raise ValueError(
                "Flux recipes without an autoencoder must set both "
                "encoder.autoencoder_shift and encoder.autoencoder_scale."
            )

        self.clip_encoder = (
            FluxEmbedder(
                version=config.encoder.clip_encoder,
                random_init=config.encoder.random_init,
            ).to(device=self.device, dtype=self._dtype)
            if config.encoder.clip_encoder
            else None
        )
        self.t5_encoder = (
            FluxEmbedder(
                version=config.encoder.t5_encoder,
                random_init=config.encoder.random_init,
            ).to(device=self.device, dtype=self._dtype)
            if config.encoder.t5_encoder
            else None
        )

        # Apply FSDP to the T5 model / CLIP model
        # pyrefly: ignore [bad-assignment]
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            parallel_dims=self.parallel_dims,
            training=config.training,
        )

        self.empty_t5_encodings: torch.Tensor | None = None
        self.empty_clip_encodings: torch.Tensor | None = None
        if config.encoder.empty_encodings_path:
            self.empty_t5_encodings = torch.from_numpy(
                np.load(f"{config.encoder.empty_encodings_path}/t5_empty.npy")
            ).to(device=self.device, dtype=self._dtype)[0]
            self.empty_clip_encodings = torch.from_numpy(
                np.load(f"{config.encoder.empty_encodings_path}/clip_empty.npy")
            ).to(device=self.device, dtype=self._dtype)[0]
        elif self.t5_encoder is not None and self.clip_encoder is not None:
            if not isinstance(self.tokenizer, FluxTokenizerContainer):
                raise ValueError("FluxTrainer requires FluxTokenizerContainer.")
            with torch.no_grad():
                empty_tokens = self.tokenizer.encode("")
                self.empty_t5_encodings = self.t5_encoder(
                    empty_tokens["t5"].to(device=self.device, dtype=torch.int)
                )[0].to(dtype=self._dtype)
                self.empty_clip_encodings = self.clip_encoder(
                    empty_tokens["clip"].to(device=self.device, dtype=torch.int)
                )[0].to(dtype=self._dtype)

        if (
            getattr(config.dataloader, "prompt_dropout_prob", 0.0) > 0.0
            and self.empty_t5_encodings is None
        ):
            raise ValueError(
                "Prompt dropout requires either online text encoders or "
                "encoder.empty_encodings_path for precomputed encodings."
            )

        if config.validator.enable:
            # pyrefly: ignore [missing-attribute]
            self.validator.flux_init(
                device=self.device,
                _dtype=self._dtype,
                autoencoder=self.autoencoder,
                t5_encoder=self.t5_encoder,
                clip_encoder=self.clip_encoder,
                autoencoder_shift_factor=self.autoencoder_shift_factor,
                autoencoder_scale_factor=self.autoencoder_scale_factor,
                dump_folder=config.dump_folder,
            )

        if config.mlperf.enable:
            global_batch_size = self.metrics_processor.global_batch_size
            assert global_batch_size is not None
            self.throughput_timer = ThroughputTimer(global_batch_size)
            self._mlperf_eval_freq_steps = max(
                1, math.ceil(config.mlperf.eval_samples / global_batch_size)
            )
            self.mlperf_logger: MLPerfLogger | None = MLPerfLogger(
                filename=f"{config.dump_folder}/mlperf_compliance.log",
                root_dir=os.path.dirname(os.path.realpath(__file__)),
                log_every_n_steps=config.metrics.log_freq,
            )
        else:
            global_batch_size = self.metrics_processor.global_batch_size
            assert global_batch_size is not None
            self.throughput_timer = ThroughputTimer(global_batch_size)
            self._mlperf_eval_freq_steps = 0
            self.mlperf_logger = None

    def batch_generator(
        self,
        data_iterable: Iterable[
            tuple[dict[str, torch.Tensor], torch.Tensor | list[torch.Tensor]]
        ],
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor | list[torch.Tensor]]]:
        """Override to count transformer tokens (image patches + text tokens)
        instead of raw pixel count from labels.numel().
        """
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            bsz = labels[0].shape[0] if isinstance(labels, list) else labels.shape[0]
            ntokens_batch = bsz * self.config.training.seq_len
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            yield input_dict, labels

    def _prepare_flux_batch(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor | list[torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(labels, list):
            mean, logvar = labels
            mean = mean.to(device=self.device, dtype=self._dtype)
            logvar = logvar.to(device=self.device, dtype=self._dtype)
            input_dict["t5_encodings"] = input_dict["t5_encodings"].to(
                device=self.device, dtype=self._dtype
            )
            input_dict["clip_encodings"] = input_dict["clip_encodings"].to(
                device=self.device, dtype=self._dtype
            )
            labels = (
                generate_unscaled_latent_from_mean_logvar(mean, logvar)
                - self.autoencoder_shift_factor
            ) * self.autoencoder_scale_factor

            drop_mask = input_dict.pop("drop_encodings", None)
            if drop_mask is not None:
                if self.empty_t5_encodings is None or self.empty_clip_encodings is None:
                    raise ValueError(
                        "Missing empty encodings for precomputed Flux batch."
                    )
                if not isinstance(drop_mask, torch.Tensor):
                    drop_mask = torch.as_tensor(drop_mask, device=self.device)
                drop_mask = drop_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
                if drop_mask.any():
                    input_dict["t5_encodings"][drop_mask] = self.empty_t5_encodings
                    input_dict["clip_encodings"][drop_mask] = self.empty_clip_encodings
            return input_dict, labels

        input_dict["image"] = labels
        input_dict = preprocess_data(
            device=self.device,
            dtype=self._dtype,
            autoencoder=self.autoencoder,
            clip_encoder=self.clip_encoder,
            t5_encoder=self.t5_encoder,
            batch=input_dict,
        )
        return input_dict, input_dict["img_encodings"]

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor | list[torch.Tensor],
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a single forward and backward pass through the model.

        Args:
            input_dict: Dictionary containing input data including prompts and other metadata
            labels: Target tensor containing the ground truth image data
            global_valid_tokens: Optional tensor tracking the total number of valid tokens across all processes.
                This field is a placeholder for now as we rescale the loss within forward_backward_step for FLUX.

        Returns:
            torch.Tensor: The computed loss value for this training step
        """

        assert (
            global_valid_tokens is None
        ), "FLUX model don't need to rescale loss by number of global valid tokens"

        input_dict, labels = self._prepare_flux_batch(input_dict, labels)

        # rewrite the global_valid_tokens because the `labels` are reset after image encoder.
        local_valid_tokens = torch.tensor(
            labels.numel(), dtype=torch.float32, device=self.device
        )

        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            # pyrefly: ignore [bad-assignment]
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        # explicitly convert flux model to be Bfloat16 no matter FSDP is applied or not
        model = self.model_parts[0]

        # image in latent space transformed by self.auto_encoder
        clip_encodings = input_dict["clip_encodings"]
        t5_encodings = input_dict["t5_encodings"]

        bsz = labels.shape[0]

        with torch.no_grad(), torch.device(self.device):
            noise = torch.randn_like(labels)
            timesteps = torch.rand((bsz,))
            sigmas = timesteps.view(-1, 1, 1, 1)
            latents = (1 - sigmas) * labels + sigmas * noise

        bsz, _, latent_height, latent_width = latents.shape

        POSITION_DIM = 3  # constant for Flux flow model
        with torch.no_grad(), torch.device(self.device):
            # Create positional encodings
            latent_pos_enc = create_position_encoding_for_latents(
                bsz, latent_height, latent_width, POSITION_DIM
            )
            text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

            # Patchify: Convert latent into a sequence of patches
            latents = pack_latents(latents)
            target = pack_latents(noise - labels)

        # Apply CP sharding if enabled
        if self.parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import cp_shard

            (
                latents,
                latent_pos_enc,
                t5_encodings,
                text_pos_enc,
                target,
            ), _ = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (latents, latent_pos_enc, t5_encodings, text_pos_enc, target),
                None,  # No attention masks for Flux
                load_balancer_type=None,
            )

        with self.train_context():
            with self.maybe_enable_amp:
                latent_noise_pred = model(
                    img=latents,
                    img_ids=latent_pos_enc,
                    txt=t5_encodings,
                    txt_ids=text_pos_enc,
                    y=clip_encodings,
                    timesteps=timesteps,
                )

                # Scale loss as we used SUM reduction for mse loss function
                # pyrefly: ignore [unsupported-operation]
                loss = self.loss_fn(latent_noise_pred, target) / global_valid_tokens
            # latent_noise_pred.shape=(bs, seq_len, vocab_size)
            # need to free to before bwd to avoid peaking memory
            # pyrefly: ignore[unsupported-delete]
            del (latent_noise_pred, noise, target)
            loss.backward()

        return loss

    def train_step(
        self,
        data_iterator: Iterable[
            tuple[dict[str, torch.Tensor], torch.Tensor | list[torch.Tensor]]
        ],
    ):
        if self.mlperf_logger is not None:
            self.mlperf_logger.log_train_step_start(self.step)

        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        if self.gradient_accumulation_steps > 1:
            raise ValueError("FLUX doesn't support gradient accumulation for now.")

        # pyrefly: ignore [no-matching-overload]
        input_dict, labels = next(data_iterator)
        self.throughput_timer.start()

        loss = self.forward_backward_step(input_dict=input_dict, labels=labels)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()
        self.throughput_timer.stop()

        should_log_metrics = self.metrics_processor.should_log(self.step)
        should_log_mlperf = (
            self.mlperf_logger is not None and self.mlperf_logger.should_log_step(self.step)
        )
        if not should_log_metrics and not should_log_mlperf:
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # NOTE: the loss returned by train
            global_avg_loss = dist_utils.dist_sum(loss, loss_mesh)
            global_max_loss = dist_utils.dist_max(loss, loss_mesh)
            if should_log_metrics:
                global_ntokens_seen = dist_utils.dist_sum(
                    torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                    loss_mesh,
                )
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            if should_log_metrics:
                global_ntokens_seen = self.ntokens_seen

        if self.mlperf_logger is not None:
            self.mlperf_logger.log_train_step_end(self.step, global_avg_loss, lr)

        if not should_log_metrics:
            return

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
            "throughput(global_samples/s)": self.throughput_timer.train_throughput(),
            "throughput(combined_global_samples/s)": self.throughput_timer.combined_throughput(),
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    def _should_run_validation(self, step: int) -> bool:
        if not self.config.validator.enable:
            return False
        if self.mlperf_logger is not None:
            return step % self._mlperf_eval_freq_steps == 0
        return self.validator.should_validate(step)

    def _get_last_eval_loss(self) -> float | None:
        if isinstance(self.validator, FluxValidator):
            return self.validator.last_loss
        return None

    def train_success(self, eval_loss: float | None) -> bool:
        return (
            self.mlperf_logger is not None
            and eval_loss is not None
            and eval_loss <= self.config.mlperf.target_eval_loss
        )

    def _log_time_to_converge(self) -> None:
        ttc = self.throughput_timer.time_to_converge()
        if ttc is None:
            return
        self.metrics_processor.logger.log(
            {"time_metrics/time_to_converge(s)": ttc},
            self.step,
        )

    @record
    def train(self):
        config = self.config
        train_started = False
        last_eval_loss: float | None = None
        training_success = False

        if self.mlperf_logger is not None:
            global_batch_size = self.metrics_processor.global_batch_size
            assert global_batch_size is not None
            self.mlperf_logger.log_run_start(
                gbs=global_batch_size,
                seed=self.mlperf_seed,
                lr=self.lr_schedulers.schedulers[0].get_last_lr()[0],
                warmup_steps=config.lr_scheduler.warmup_steps,
                gradient_clip_norm=config.training.max_norm,
                optimizer_config=self.optimizers.optimizers[0].param_groups[0],
                eval_freq=config.mlperf.eval_samples,
            )

        self.checkpointer.load(step=config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")

        with (
            maybe_enable_profiling(
                config.profiling,
                global_step=self.step,
                base_folder=config.dump_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                config.profiling,
                global_step=self.step,
                base_folder=config.dump_folder,
            ) as memory_profiler,
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                if not train_started and self.mlperf_logger is not None:
                    self.mlperf_logger.log_train_start()
                    train_started = True

                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                self.checkpointer.save(
                    self.step, last_step=(self.step == config.training.steps)
                )

                if self._should_run_validation(self.step):
                    if self.mlperf_logger is not None:
                        self.mlperf_logger.log_eval_start(self.step)
                    self.validator.validate(
                        self.model_parts,
                        self.step,
                        extra_metrics=lambda: {
                            "throughput(global_samples/s)": self.throughput_timer.train_throughput(),
                            "throughput(combined_global_samples/s)": self.throughput_timer.combined_throughput(),
                        },
                    )
                    last_eval_loss = self._get_last_eval_loss()
                    if self.mlperf_logger is not None and last_eval_loss is not None:
                        self.mlperf_logger.log_eval_end(self.step, last_eval_loss)
                    if self.train_success(last_eval_loss):
                        training_success = True
                        self._log_time_to_converge()
                        break

                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")
        if self.mlperf_logger is not None:
            self.mlperf_logger.log_train_end(success=training_success)
