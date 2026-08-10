import pandas as pd
import numpy as np
from tqdm import tqdm
from jaxtyping import Array
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from typing import NamedTuple
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from config import Config


def _get_xp(x):
    return jnp if isinstance(x, (jax.Array, jax.core.Tracer)) else np


class ShiftedLogScaler(BaseEstimator, TransformerMixin):
    def fit(self, X: Array, y=None):
        xp = _get_xp(X)

        self.shift_ = xp.abs(np.min(X))
        X_shifted = xp.log1p(X + self.shift_)
        self.mean_ = xp.mean(X_shifted)
        self.std_ = xp.std(X_shifted)
        return self

    def transform(self, X):
        xp = _get_xp(X)

        X_clipped = xp.clip(X, -self.shift_, None)
        return (xp.log1p(X_clipped + self.shift_) - self.mean_) / (self.std_ + 1e-8)

    def inverse_transform(self, X):
        xp = _get_xp(X)

        return xp.expm1((X * self.std_) + self.mean_) - self.shift_


class TabularReachDataset:
    def __init__(self, samples, scalers):
        self.samples = samples
        self.scalers = scalers

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pad_size = sample["hws"].shape[0]
        metadata = {
            "svs_idx": np.tile(sample["svs_idx"], pad_size),
            "reach_id": np.tile(sample["reach_id"], pad_size),
            "date": sample["date"].reindex(range(pad_size), fill_value=pd.NaT),
        }
        data = {k: sample[k] for k in ["x_d", "d_mask", "x_s", "hws", "y"]}

        return metadata, data

    def normalize(self, x_raw_arr: Array, name: str):
        scaler = self.scalers.get(name)
        if scaler is None:
            return x_raw_arr

        original_shape = x_raw_arr.shape
        x_scaled = scaler.transform(x_raw_arr.reshape(-1, 1))
        return x_scaled.reshape(original_shape)

    def denormalize(self, x_scaled_arr: Array, name: str):
        scaler = self.scalers.get(name)
        if scaler is None:
            return x_scaled_arr

        original_shape = x_scaled_arr.shape
        x_inv = scaler.inverse_transform(x_scaled_arr.reshape(-1, 1))
        return x_inv.reshape(original_shape)


def _read_list(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def build_tabular_datasets(cfg: Config):
    dynamic_df = pd.read_parquet(cfg.parquet_file)
    static_df = pd.read_parquet(cfg.attributes_file)

    sites_dict = {
        subset: _read_list(getattr(cfg, f"{subset}_basin_file"))
        for subset in ["train", "validate", "test"]
    }

    dynamic_df["raw_h"] = dynamic_df["d_wse"]
    dynamic_df["raw_w"] = dynamic_df["width"]
    dynamic_df["raw_s"] = dynamic_df["slope"]

    train_dynamic_df = dynamic_df[dynamic_df["reach_id"].isin(sites_dict["train"])]
    train_static_df = static_df[static_df["reach_id"].isin(sites_dict["train"])]

    dynamic_cols = [ff for f in cfg.features.dynamic.values() for ff in f]
    dyn_cols_to_scale = dynamic_cols + cfg.features.target
    static_cols_to_scale = cfg.features.static

    scalers = {}

    def scale_features(full_df: pd.DataFrame, train_df: pd.DataFrame, columns: list[str]):
        full_df = full_df.copy()
        protected_cols = {"reach_id", "svs_idx", "raw_h", "raw_w", "raw_s"}

        for col in columns:
            if col in protected_cols or col in cfg.no_norm_cols:
                continue

            train_data = train_df[[col]].values
            full_data = full_df[[col]].values

            if col in cfg.log_norm_cols:
                scaler = ShiftedLogScaler()
            else:
                scaler = StandardScaler()

            scaler.fit(train_data)
            full_df[col] = scaler.transform(full_data).flatten()
            scalers[col] = scaler

        return full_df

    dynamic_df = scale_features(dynamic_df, train_dynamic_df, dyn_cols_to_scale)
    static_df = scale_features(static_df, train_static_df, static_cols_to_scale)

    static_df = static_df.set_index("reach_id")

    splits = {}
    for subset, sites in sites_dict.items():
        splits[subset] = []
        subset_df = dynamic_df[dynamic_df["reach_id"].isin(sites)]

        for site_id, site_df in tqdm(subset_df.groupby("reach_id"), disable=cfg.quiet):
            n_samples = len(site_df)
            if n_samples < 10:
                continue
            if n_samples > cfg.pad_size:
                raise ValueError(
                    f"Location {site_id} has {n_samples} samples, exceeding {cfg.pad_size=}"
                )

            svs_idx = site_df["svs_idx"].iloc[0]
            date = site_df["datetime"].dt.date

            # Dyanmic data
            x_d_arr = site_df[dynamic_cols].values.astype(np.float32)
            x_pad = np.ones((cfg.pad_size, len(dynamic_cols)), dtype=np.float32)
            x_pad[:n_samples, :] = x_d_arr

            # Dynamic data mask
            d_mask = np.zeros((cfg.pad_size), dtype=np.int8)
            d_mask[:n_samples] = 1

            # Static data
            x_s = static_df.loc[site_id, static_cols_to_scale].values.astype(np.float32)

            # HWS data
            hws_arr = site_df[["raw_h", "raw_w", "raw_s"]].values.astype(np.float32)

            if cfg.hw_corr_thresh is not None:
                corr_matrix = np.corrcoef(hws_arr[:, 0], hws_arr[:, 1])
                if np.isnan(corr_matrix[0, 1]) or corr_matrix[0, 1] <= cfg.hw_corr_thresh:
                    continue

            if cfg.median_w:
                hws_arr[:, 1] = np.median(hws_arr[:, 1])
            if cfg.median_s:
                hws_arr[:, 2] = np.median(hws_arr[:, 2])

            hws_pad = np.ones((cfg.pad_size, 3), dtype=np.float32)
            hws_pad[:n_samples, :] = hws_arr

            # Target data
            y_pad = {}
            for target_col in cfg.features.target:
                target_arr = site_df[target_col].values.astype(np.float32)
                target_pad = np.ones(cfg.pad_size, dtype=np.float32)
                target_pad[:] = np.nan
                target_pad[:n_samples] = target_arr

                y_pad[target_col] = target_pad

            splits[subset].append(
                {
                    "svs_idx": svs_idx,
                    "reach_id": site_id,
                    "date": date,
                    "x_d": x_pad,
                    "d_mask": d_mask,
                    "x_s": x_s,
                    "hws": hws_pad,
                    "y": y_pad,
                }
            )

    return {subset: TabularReachDataset(split, scalers) for subset, split in splits.items()}


class TabularBatch(NamedTuple):
    x_d: Array
    d_mask: Array
    x_s: Array
    hws: Array
    y: dict[str, Array]

    def to_jax(self):
        return TabularBatch(
            x_d=jnp.asarray(self.x_d),
            d_mask=jnp.asarray(self.d_mask),
            x_s=jnp.asarray(self.x_s),
            hws=jnp.asarray(self.hws),
            y={k: jnp.asarray(v) for k, v in self.y.items()},
        )


def numpy_collate_fn(batch: list[tuple[dict, dict]]) -> tuple[dict, TabularBatch]:
    """
    Stacks nested tuples of dictionaries.
    Returns metadata as a dictionary and training data as a TabularBatch.
    """
    batched_metadata_dict, batched_data_dict = jtu.tree_map(lambda *xs: np.stack(xs), *batch)
    return batched_metadata_dict, TabularBatch(**batched_data_dict)


class TabularReachDataLoader(DataLoader):
    """
    Custom DataLoader that handles a single subset (Train, Val, or Test).
    Exposes dataset scaling utilities for convenience.
    """

    def __init__(self, cfg: Config, dataset: TabularReachDataset, infinite_shuffle=False):
        # PyTorch's default collate_fn automatically handles a list of dictionaries
        # by stacking the tensors for each key into batched tensors.
        super().__init__(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=infinite_shuffle,
            num_workers=cfg.num_workers,
            collate_fn=numpy_collate_fn,
            persistent_workers=True,
        )
        self.infinite_shuffle = infinite_shuffle

    def __iter__(self):
        if self.infinite_shuffle:
            while True:
                yield from super().__iter__()
        else:
            yield from super().__iter__()

    def normalize(self, x_raw: Array, name: str):
        """Pass-through to the dataset's normalize method."""
        return self.dataset.normalize(x_raw, name)

    def denormalize(self, x_scaled: Array, name: str):
        """Pass-through to the dataset's denormalize method."""
        return self.dataset.denormalize(x_scaled, name)
