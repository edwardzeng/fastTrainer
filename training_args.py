#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
File: training_args.py
Date: 2023/01/24
"""
import contextlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from packaging import version

from transformers.debug_utils import DebugOption
from transformers.trainer_utils import (
    IntervalStrategy,
    SchedulerType,
)
from transformers.utils import (
    ExplicitEnum,
    cached_property,
    is_accelerate_available,
    is_torch_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_tf32_available,
    logging,
    requires_backends,
)


if is_torch_available():
    import torch
    import torch.distributed as dist


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


TORCH_COMPILE_BACKENDS = [
    "eager",
    "aot_eager",
    "inductor",
    "nvfuser",
    "aot_nvfuser",
    "aot_cudagraphs",
    "ofi",
    "fx2trt",
    "onnxrt",
    "ipex",
]


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAFACTOR = "adafactor"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    SGD = "sgd"
    ADAGRAD = "adagrad"


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.
    """

    framework = "pt"
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " evaluation_strategy."
            )
        },
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    log_level: Optional[str] = field(
        default="passive",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: Optional[str] = field(
        default="passive",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    use_mps_device: bool = field(
        default=False, metadata={"help": "Whether to use Apple Silicon chip based `mps` device."}
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
    jit_mode_eval: bool = field(
        default=False, metadata={"help": "Whether or not to use PyTorch jit trace for inference"}
    )
    use_ipex: bool = field(
        default=False,
        metadata={
            "help": (
                "Use Intel extension for PyTorch when it is available, installation:"
                " 'https://github.com/intel/intel-extension-for-pytorch'"
            )
        },
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (no_cuda). This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    half_precision_backend: str = field(
        default="auto",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["auto", "cuda_amp", "cpu_amp"],
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )
    tf32: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    debug: str = field(
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: Optional[int] = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already"
                " loaded json file as a dict"
            )
        },
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    optim: Union[OptimizerNames, str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_bucket_cap_mb: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )
    # Deprecated arguments
    _n_gpu: int = field(init=False, repr=False, default=-1)

    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                " a CUDA Out-of-Memory was reached"
            )
        },
    )
    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training"
            )
        },
    )
    ddp_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": "Overrides the default timeout for distributed training (value should be given in seconds)."
        },
    )
    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )
    torch_compile_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which backend to use with `torch.compile`, passing one will trigger a model compilation.",
            "choices": TORCH_COMPILE_BACKENDS,
        },
    )
    torch_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which mode to use with `torch.compile`, passing one will trigger a model compilation.",
            "choices": ["default", "reduce-overhead", "max-autotune"],
        },
    )

    def __post_init__(self):
        # Handle --use_env option in torch.distributed.launch (local_rank not passed as an arg then).
        # This needs to happen before any call to self.device or self.n_gpu.
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        # see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir
        if self.framework == "pt" and is_torch_available():
            if self.bf16 or self.bf16_full_eval:

                if self.no_cuda and not is_torch_bf16_cpu_available():
                    # cpu
                    raise ValueError("Your setup doesn't support bf16/(cpu, tpu, neuroncore). You need torch>=1.10")
                elif not self.no_cuda and torch.cuda.is_available() and not is_torch_bf16_gpu_available():
                    # gpu
                    raise ValueError(
                        "Your setup doesn't support bf16/gpu. You need torch>=1.10, using Ampere GPU with cuda>=11.0"
                    )

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError("At most one of fp16 and bf16 can be True for full eval, but not both")

        self.optim = OptimizerNames(self.optim)

        if (
            self.framework == "pt"
            and is_torch_available()
            and (self.device.type != "cuda")
            and (self.fp16 or self.fp16_full_eval)
        ):
            raise ValueError(
                "FP16 Mixed precision training with AMP (`--fp16`) and FP16 half precision evaluation"
                " (`--fp16_full_eval`) can only be used on CUDA devices."
            )

        if (
            self.framework == "pt"
            and is_torch_available()
            and (self.device.type != "cuda")
            and (self.device.type != "cpu")
            and (self.bf16 or self.bf16_full_eval)
        ):
            raise ValueError(
                "BF16 Mixed precision training with AMP (`--bf16`) and BF16 half precision evaluation"
                " (`--bf16_full_eval`) can only be used on CUDA or CPU/TPU/NeuronCore devices."
            )

        if (self.torch_compile_mode is not None or self.torch_compile_backend is not None) and not self.torch_compile:
            self.torch_compile = True
        if self.torch_compile and self.torch_compile_backend is None:
            self.torch_compile_backend = "inductor"
        if self.framework == "pt" and is_torch_available() and self.torch_compile:
            if is_torch_tf32_available():
                if self.tf32 is None and not self.fp16 or self.bf16:
                    logger.info(
                        "Setting TF32 in CUDA backends to speedup torch compile, you won't see any improvement"
                        " otherwise."
                    )
                    torch.backends.cuda.matmul.allow_tf32 = True
            else:
                logger.warning(
                    "The speedups for torchdynamo mostly come wih GPU Ampere or higher and which is not detected here."
                )
        if self.framework == "pt" and is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                # no need to assert on else

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                " during training"
            )

        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]

        if self.deepspeed:
            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            if not is_accelerate_available():
                raise ValueError("--deepspeed requires Accelerate to be installed: `pip install accelerate`.")
            from transformers.deepspeed import HfTrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)


    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_device_train_batch_size` in distributed training).
        """
        per_device_batch_size = self.per_device_train_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from `per_device_eval_batch_size` in distributed training).
        """
        per_device_batch_size = self.per_device_eval_batch_size
        eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    @property
    def ddp_timeout_delta(self) -> timedelta:
        """
        The actual timeout for torch.distributed.init_process_group since it expects a timedelta variable.
        """
        return timedelta(seconds=self.ddp_timeout)

    @cached_property
    def _setup_devices(self) -> "torch.device":
        requires_backends(self, ["torch"])
        logger.info("PyTorch: setting up devices")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
            self.local_rank = get_int_from_env(
                ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
                self.local_rank,
            )
            if self.local_rank != -1 and not torch.distributed.is_initialized():
                # Initializes distributed backend for cpu
                raise ValueError("CPU distributed backend is not supported")
        elif self.deepspeed:
            # deepspeed inits torch.distributed internally
            from transformers.deepspeed import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed(timeout=timedelta(seconds=self.ddp_timeout))

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            if self.use_mps_device:
                if not torch.backends.mps.is_available():
                    if not torch.backends.mps.is_built():
                        raise AssertionError(
                            "MPS not available because the current PyTorch install was not "
                            "built with MPS enabled. Please install torch version >=1.12.0 on "
                            "your Apple silicon Mac running macOS 12.3 or later with a native "
                            "version (arm64) of Python"
                        )
                    else:
                        raise AssertionError(
                            "MPS not available because the current MacOS version is not 12.3+ "
                            "and/or you do not have an MPS-enabled device on this machine."
                        )
                else:
                    if not version.parse(version.parse(torch.__version__).base_version) > version.parse("1.12.0"):
                        warnings.warn(
                            "We strongly recommend to install PyTorch >= 1.13 (nightly version at the time of writing)"
                            " on your MacOS machine. It has major fixes related to model correctness and performance"
                            " improvements for transformer based models. Please refer to"
                            " https://github.com/pytorch/pytorch/issues/82707 for more details."
                        )
                    device = torch.device("mps")
                    self._n_gpu = 1

            else:
                # if n_gpu is > 1 we'll use nn.DataParallel.
                # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
                # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
                # trigger an error that a device index is missing. Index 0 takes into account the
                # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
                # will use the first GPU in that env, i.e. GPU#1
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
                # the default value.
                self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", timeout=self.ddp_timeout_delta)
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        requires_backends(self, ["torch"])
        return self._setup_devices

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        requires_backends(self, ["torch"])
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu

    @property
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs cores are available. One of:

        - `ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - `ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses `torch.nn.DataParallel`).
        - `ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          `torch.nn.DistributedDataParallel`).
        """
        requires_backends(self, ["torch"])
        if self.local_rank != -1:
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        requires_backends(self, ["torch"])

        if self.local_rank != -1:
            return torch.distributed.get_world_size()
        return 1

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        requires_backends(self, ["torch"])
        if self.local_rank != -1:
            return torch.distributed.get_rank()
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        requires_backends(self, ["torch"])

        if self.local_rank != -1:
            return self.local_rank
        return 0

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        if self.log_on_each_node:
            return self.local_process_index == 0
        else:
            return self.process_index == 0

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        if self.save_on_each_node:
            return self.local_process_index == 0
        else:
            return self.process_index == 0

    def get_process_log_level(self):
        """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to `logging.INFO` unless overridden by `log_level` argument.

        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.

        The choice between the main and replica process settings is made according to the return value of `should_log`.
        """

        # convert to int
        log_level = trainer_log_levels[self.log_level]
        log_level_replica = trainer_log_levels[self.log_level_replica]

        log_level_main_node = logging.INFO if log_level == -1 else log_level
        log_level_replica_node = logging.WARNING if log_level_replica == -1 else log_level_replica
        return log_level_main_node if self.should_log else log_level_replica_node

    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        return True

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        return not self.deepspeed

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
        A context manager for torch distributed environment where on needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.

        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.

        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

        """
        if is_torch_available() and self.world_size > 1:
            main_process_desc = "main process"
            if local:
                is_main_process = self.local_process_index == 0
                main_process_desc = "main local process"
            else:
                is_main_process = self.process_index == 0

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")
                    torch.distributed.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    torch.distributed.barrier()
        else:
            yield

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = dict((field.name, getattr(self, field.name)) for field in fields(self) if field.init)

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
