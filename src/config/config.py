import functools
import json
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from pydantic.types import DirectoryPath, FilePath

from .add_assimilator import AssimConfig

# Import model argument classes from model_args.py
from .model_args import ModelArgs


class Features(BaseModel):
    dynamic: dict[str, list[str]]
    static: list[str] | None = None
    target: list[str]


class Encoding(BaseModel):
    categorical: dict[str, list[str]] = Field(default_factory=dict)
    bitmask: dict[str, list[int]] = Field(default_factory=dict)


class StepKwargs(BaseModel):
    loss_name: Literal["mse", "mae", "huber", "nse", "spin_up_nse", "gmm_nll", "manning_nll"] = (
        "mse"
    )
    target_weights: dict[str, float] = None
    corr_keys: dict = {}
    # gmm_nll only: caps the per-observation z-score before squaring, so a single
    # catastrophically-mispredicted (basin, date) sample can't dominate the batch
    # mean / gradient. Lower = more robust to outlier basins, at the cost of less
    # gradient signal on genuinely-hard basins.
    gmm_clip: float = Field(20.0, gt=0.0)
    # nse only: epsilon added to each gauge's target std in the NSE* denominator
    # (Kratzert et al., 2019). In units of the normalized target.
    nse_eps: float = Field(0.1, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def check_deprecated_loss(cls, values):
        if isinstance(values, dict) and "loss" in values:
            warnings.warn("'loss' is deprecated. Use 'loss_name' instead.", DeprecationWarning)
            # Only override loss_name if it wasn't explicitly passed
            values.setdefault("loss_name", values["loss"])
        return values


class EarlyStopKwargs(BaseModel):
    patience: int = Field(5, gt=0)
    threshold: float = Field(..., gt=0, lt=1.0)
    # "rel": improvement measured relative to the best value (unstable when the tracked
    # value is near zero or changes sign, e.g. NLL). "abs": improvement in raw units.
    threshold_mode: Literal["rel", "abs"] = "rel"


class ValueFilter(BaseModel):
    column: str
    operation: str
    value: Any
    feature_list: list[str] | None = None
    feature_group: str | None = None

    @model_validator(mode="before")
    @classmethod
    def feature_type(cls, values: dict):
        f_list = values.get("feature_list")
        f_group = values.get("feature_group")
        if f_list and f_group:
            raise ValueError("Specify only one of 'feature_list' or 'feature_group', not both.")
        if not f_list and not f_group:
            raise ValueError("Specify exactly one of 'feature_list' or 'feature_group'.")
        return values


class DataSubset(str, Enum):
    train = "train"  # Loss, gradients, and back propagation
    validate = "validate"  # Loss only
    test = "test"  # Error statistics only
    predict = "predict"  # Same as test but for sequences without test data


class Config(BaseModel):
    def to_json(self, path: str | Path, exclude_none: bool = True):
        """Dump the config to a JSON file."""
        cfg_out = self.model_copy(deep=True)
        json_str = cfg_out.model_dump_json(
            indent=4, exclude_none=exclude_none, exclude_unset=False, exclude={"cfg_path"}
        )
        with open(path, "w") as f:
            f.write(json_str)

    @classmethod
    def from_file(cls, file_path: Path | str) -> "Config":
        """
        Load the config from a YAML or JSON file. Automatically detects file type by extension.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        ext = file_path.suffix.lower()
        if ext in [".yml", ".yaml"]:
            with open(file_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
        elif ext == ".json":
            with open(file_path, "r") as f:
                cfg_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

        cfg = cls(**cfg_dict)
        cfg.cfg_path = file_path
        return cfg

    def __str__(self) -> str:
        """
        Provides a nicely formatted JSON string of all model attributes,
        """
        return self.model_dump_json(indent=4)

    # Data paths
    data_root: DirectoryPath
    zarr_dir: DirectoryPath
    attributes_file: FilePath
    train_basin_file: FilePath
    validate_basin_file: FilePath
    test_basin_file: FilePath
    graph_network_file: FilePath | None = None

    # Data processing
    features: Features
    time_gaps: dict
    precomp_scaling: bool = True
    dynamic_encoding: Encoding = Field(default_factory=Encoding)
    static_encoding: Encoding = Field(default_factory=Encoding)
    # Applies to dynamic, target, AND static columns (static columns are log1p'd then
    # z-scored). Use for heavy-tailed, non-negative static attributes.
    log_norm_cols: list[str] = Field(default_factory=list)
    no_norm_cols: list[str] = Field(default_factory=list)
    range_norm_cols: list[str] = Field(default_factory=list)
    in_memory: bool = False
    # "precomputed": target normalization stats come from the per-basin zarr attrs, which
    # cover every subbasin (incl. validation/test gauges) and every date. "train_subset":
    # recompute target stats from train subbasins over train_date_range only.
    target_norm_stats: Literal["precomputed", "train_subset"] = "precomputed"
    # Cache key version. 2 leaves out static-only settings (features.static, static_encoding,
    # attributes_file), which never affect the cached dynamic data, so experiments that only
    # change static features share one cache. 1 keeps existing caches' keys unchanged.
    cache_hash_version: Literal[1, 2] = 1

    # Data slicing
    train_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    validate_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    test_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    add_rolling_means: list[int] | None = None
    clip_feature_range: dict[str, list[float | None]] = Field(default_factory=dict)
    # Shift dynamic features in time before caching, in days, with xarray's shift semantics:
    # -1 moves every value one day earlier (the value stamped D+1 is used for D). ERA5-Land
    # accumulations (tp, ro, sro, e, ssrd, strd) are stamped at 00 UTC of the following day,
    # so they need -1. Edge dates left empty by the shift repeat the nearest shifted value.
    feature_shift_days: dict[str, int] = Field(default_factory=dict)
    value_filters: list[ValueFilter] = Field(default_factory=list)
    exclude_target_from_index: list[str] = Field(default_factory=list)

    # DataLoader
    data_subset: DataSubset = DataSubset.train
    target_nodes_per_batch: int
    candidate_pool_size: int
    shuffle: bool = True
    num_workers: int = Field(..., ge=0)
    pin_memory: bool = False
    drop_last: bool = True
    backend: Literal["cpu", "gpu", "tpu"] | None = None
    num_devices: int | None = Field(None, gt=0)

    dataset_type: Literal["graph", "tabular"] = "graph"
    batch_size: int | None = Field(None, gt=0)

    # Tablar args
    pad_size: int | None = Field(None, gt=0)
    parquet_file: FilePath | None = None
    hw_corr_thresh: float | None = Field(None, lte=1.0)
    median_w: bool = False
    median_s: bool = False

    # Model
    sequence_length: int | None = None
    model_args: ModelArgs = Field(discriminator="name")

    # Trainer
    max_training_steps: int = Field(..., gt=0)  # Required
    max_training_hours: int | None = Field(None, gt=0)
    log_every_n_steps: int = Field(..., gt=0)
    log_every_n_minutes: int | None = Field(None, gt=0)
    validate_every_n_steps: int | None = Field(None, gt=0)
    # Validation uses a fixed, evenly spaced subset of validation sequences so every check
    # scores the same dates. max_validation_steps bounds the number of batches per check.
    max_validation_steps: int | None = Field(None, gt=0)
    # "loss": best-checkpoint/early-stop track step_kwargs.loss_name on validation
    # (existing behavior). "nse": track median per-basin NSE instead, computed via
    # the same evaluate.inference/evaluate.metrics pipeline used for post-hoc
    # test/train evaluation, capped at max_validation_steps batches per check.
    validation_metric: Literal["loss", "nse"] = "loss"

    warmup_steps: int = Field(0, ge=0)
    initial_lr: float = Field(..., gt=0, lt=1.0)
    final_lr: float = Field(..., gt=0.0, lt=1.0)
    max_grad_norm: float | None = None
    # Average gradients over this many batches before each optimizer update (optax.MultiSteps).
    # Raises the effective batch size without extra GPU memory. max_training_steps and
    # warmup_steps still count batches; the LR schedule is rescaled to optimizer updates.
    grad_accum_steps: int = Field(1, gt=0)
    step_kwargs: StepKwargs
    early_stop_kwargs: EarlyStopKwargs = None
    optimizer_name: Literal["adam", "adamw"] = "adam"
    weight_decay: float = Field(0.01, gt=0.0, lt=1.0)

    # Meta
    parameter_search_grid: dict = Field(default_factory=dict)

    # Outputs
    quiet: bool = True
    log: bool = True

    # Internal (not from YML)
    cfg_path: Path = Path()

    # MODEL / BEFORE
    @model_validator(mode="before")
    @classmethod
    def validate_paths(cls, values: dict):
        def resolve_path(attr_str):
            str_path = values.get(attr_str)
            if str_path:
                path = Path(str_path)
                # If already absolute, use as-is; otherwise resolve
                return path if path.is_absolute() else path.resolve()
            else:
                return None

        abs_paths = [
            "data_root",
            "zarr_dir",
        ]
        for attr in abs_paths:
            values[attr] = resolve_path(attr)

        rel_paths = [
            "attributes_file",
            "train_basin_file",
            "validate_basin_file",
            "test_basin_file",
            "graph_network_file",
        ]
        for attr in rel_paths:
            if values.get(attr) is not None:
                values[attr] = values["data_root"] / values[attr]

        return values

    # MODEL / AFTER
    @model_validator(mode="after")
    def validate_dataset_type(self):
        # This should probably be refactored to have pydantic models for each dataset.
        if self.dataset_type == "mcfli":
            required_keys = ["batch_size", "pad_size"]
            missing_keys = [k for k in required_keys if getattr(self, k) is None]
            if any(missing_keys):
                raise ValueError(f"incomplete config for MCFLIDataset. {missing_keys = }")

        return self

    @model_validator(mode="after")
    def validate_log_intervals(self):
        """
        Ensure at least one of log_every_n_steps or log_every_n_minutes is not inf.
        """
        steps_inf = self.log_every_n_steps == np.inf
        minutes_inf = self.log_every_n_minutes == np.inf
        if steps_inf and minutes_inf:
            raise ValueError(
                "At least one of log_every_n_steps or log_every_n_minutes must be set to a finite value."
            )
        return self

    @model_validator(mode="after")
    def validate_nse_requires_seq2seq(self):
        """
        Ensure that if loss_name == 'nse', then model_args.seq2seq == True.
        Only checks models that actually have the seq2seq attribute.
        """
        if self.step_kwargs.loss_name == "spin_up_nse":
            if hasattr(self.model_args, "seq2seq"):
                if not self.model_args.seq2seq:
                    raise ValueError(
                        "When using loss_name='spin_up_nse', model_args.seq2seq must be set to True."
                    )
            else:
                raise ValueError(
                    "loss_name='spin_up_nse' requires a model type with a 'seq2seq' attribute AND seq2seq=True."
                )
        return self

    @model_validator(mode="after")
    def validate_feature_shift_days(self):
        dynamic_cols = {c for cols in self.features.dynamic.values() for c in cols}
        unknown = set(self.feature_shift_days) - dynamic_cols
        if unknown:
            raise ValueError(
                f"feature_shift_days names columns that are not dynamic features: {sorted(unknown)}"
            )
        return self

    @model_validator(mode="after")
    def validate_spin_up(self):
        spin_up = getattr(self.model_args, "spin_up", 0)
        if spin_up and not getattr(self.model_args, "seq2seq", False):
            raise ValueError("model_args.spin_up only applies when model_args.seq2seq is True.")
        if spin_up and self.sequence_length is not None and spin_up >= self.sequence_length:
            raise ValueError(
                f"model_args.spin_up ({spin_up}) must be less than sequence_length "
                f"({self.sequence_length})."
            )
        return self

    @model_validator(mode="after")
    def validate_time_gaps_keys(self):
        """
        Ensure that time_gaps dict has all the same keys as features.dynamic dict,
        and that all values in time_gaps are single booleans.
        """
        dynamic_keys = set(self.features.dynamic.keys())
        time_gaps_keys = set(self.time_gaps.keys())
        if dynamic_keys != time_gaps_keys:
            missing = dynamic_keys - time_gaps_keys
            extra = time_gaps_keys - dynamic_keys
            msg = []
            if missing:
                msg.append(f"Missing keys in time_gaps: {sorted(missing)}")
            if extra:
                msg.append(f"Extra keys in time_gaps: {sorted(extra)}")
            raise ValueError("time_gaps keys must match features.dynamic keys. " + "; ".join(msg))
        # Check that all values are single booleans
        for k, v in self.time_gaps.items():
            if not isinstance(v, bool):
                raise TypeError(f"time_gaps['{k}'] must be a boolean, got {type(v)}.")
        return self

    # FIELD / BEFORE
    @field_validator(
        "train_date_range",
        "validate_date_range",
        "test_date_range",
        mode="before",
    )
    def parse_time_slice(cls, v):
        if (not isinstance(v, list)) or (len(v) != 2):
            raise ValueError("date ranges must be a list of two elements (start, end).")
        # Convert to datetime objects
        t_start = datetime.fromisoformat(v[0])
        t_end = datetime.fromisoformat(v[1])
        return [t_start, t_end]

    @field_validator("clip_feature_range", mode="before")
    def process_clip_feature_range(cls, v):
        if not v:
            return {}
        processed_ranges = {}
        for key, range_list in v.items():
            if (not isinstance(range_list, list)) or (len(range_list) != 2):
                raise ValueError(f"Each clip range for '{key}' must have exactly 2 elements.")
            lower = -np.inf if range_list[0] is None else range_list[0]
            upper = np.inf if range_list[1] is None else range_list[1]
            processed_ranges[key] = [lower, upper]
        return processed_ranges

    # FIELD / AFTER
    @field_validator("step_kwargs", mode="after")
    def validate_step_kwargs(cls, v: StepKwargs, info):
        cfg = info.data  # Access the entire config data

        # Convert target_weights dict to list based on features.target order
        if v.target_weights is None:
            v.target_weights = {}

        if isinstance(v.target_weights, dict):
            v.target_weights = {
                target: v.target_weights.get(target, 1) for target in cfg["features"].target
            }
        else:
            raise TypeError(f"target_weights must be a dict or None, got {type(v.target_weights)}.")

        return v

    # CONFIG MANIPULATION
    def add_assimilation(self, assim: AssimConfig):
        """
        Inserts new feature groups, encodings, time gaps, etc. based on the assimilation target
        """
        self.features.dynamic.update(assim.new_features)
        self.dynamic_encoding.categorical.update(assim.categorical_encoding)
        self.dynamic_encoding.bitmask.update(assim.bitmask_encoding)
        for new_group in assim.new_features.keys():
            self.time_gaps[new_group] = True

        self.model_args.assim_sizes.update(assim.model_args)

        self.max_training_steps += assim.additional_steps

        # Hours do not accumulate from saved states.
        self.max_training_hours = assim.additional_hours

    def rgetattr(self, attr: str, *args):
        """
        Recursively get attribute from the instance using dot notation.
        """

        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [self] + attr.split("."))

    def rsetattr(self, attr: str, value: Any):
        """
        Recursively set attribute on the instance using dot notation.
        """
        pre, _, post = attr.rpartition(".")
        return setattr(self.rgetattr(pre) if pre else self, post, value)

    def update_from_grid(self, idx: int) -> "Config":
        param_dict = self.parameter_search_grid
        if param_dict == {}:
            raise AttributeError("param_search_dict is empty. Cannot update from grid.")

        def flatten_param_dict(d: dict, prefix=""):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield from flatten_param_dict(v, full_key)
                else:
                    yield full_key, v

        dotted_keys, value_set = zip(*flatten_param_dict(param_dict))

        # Compute the total number of combinations
        sizes = [len(v) for v in value_set]
        n_total = np.prod(sizes)
        if idx < 0 or idx >= n_total:
            raise IndexError(f"Index {idx} is out of range for grid of size {n_total}")

        # Do not change the seed! This ensures deterministic grid shuffling between runs.
        rng = np.random.default_rng(42)
        # Shuffle the grid indices (permutation of size n_total) and select our index
        grid_idx = rng.permutation(n_total)[idx]

        # Convert grid_idx to N-dimensional indices for each parameter
        unravel = np.unravel_index(grid_idx, sizes)
        new_values = [v[i] for v, i in zip(value_set, unravel)]

        for attr, value in zip(dotted_keys, new_values):
            self.rsetattr(attr, value)
