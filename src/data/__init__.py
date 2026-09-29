from typing import TypeAlias

from .basindatalake import BasinDataLake
from .zarr_store import ZarrBasinStore
from .cached_basingraphdataloader import CachedBasinGraphDataLoader
from .cached_basingraphdataset import CachedBasinGraphDataset, DynamicCacheManager, GraphBatch
from .mcfli_dataset import MCFLIDataset, MCFLIDataLoader
from .tabular_dataset import (
    TabularBatch,
    build_tabular_datasets,
    TabularReachDataset,
    TabularReachDataLoader,
)

from config import Config, DataSubset

__all__ = [
    GraphBatch,
    DynamicCacheManager,
    CachedBasinGraphDataset,
    CachedBasinGraphDataLoader,
    TabularBatch,
    build_tabular_datasets,
    TabularReachDataset,
    TabularReachDataLoader,
    MCFLIDataset,
    MCFLIDataLoader,
    BasinDataLake,
    ZarrBasinStore,
]

Batch: TypeAlias = GraphBatch | TabularBatch
Dataset: TypeAlias = CachedBasinGraphDataset | TabularReachDataset
DataLoader: TypeAlias = CachedBasinGraphDataLoader | TabularReachDataLoader


def kill_dataloader(dl: DataLoader):
    if dl is None:
        return
    if dl._iterator:
        dl._iterator._shutdown_workers()
    del dl


def kill_dataloaders(dls: dict[str, DataLoader]):
    for dl in dls.values():
        kill_dataloader(dl)


def _samples_per_batch(cfg: Config, dataset: CachedBasinGraphDataset) -> int:
    """Lower bound on how many sequences the packing sampler fits in one batch."""
    largest = max(dataset.basin_subbasin_counts.values())
    return max(1, (cfg.target_nodes_per_batch - 1) // largest)


def get_dataloaders(
    cfg: Config, splits: list[DataSubset] = None, inf_shuffle: dict[DataSubset, bool] = {}
) -> dict:
    """
    Factory function to build and return requested dataloaders.

    Args:
        cfg: The configuration object.
        splits: List of splits to generate (e.g., ["train", "val", "test"]).
                Defaults to ["train", "val"].

    Returns:
        dict: A dictionary mapping split names to their DataLoaders.
    """
    if splits is None:
        splits = [DataSubset.train, DataSubset.validate]

    # Create DataSubset objects for validation. Sometimes I'm lazy and pass in a string.
    splits = [DataSubset[split] for split in splits]

    loaders = {}

    match cfg.dataset_type:
        case "graph":
            manager = DynamicCacheManager(cfg)
            for split in splits:
                cache_dir = manager.create_cache(split)
                dataset = CachedBasinGraphDataset(cfg, cache_dir, split)
                # Only train shuffles by default. Validation iterates a fixed, evenly spaced
                # subset so that successive checks are comparable.
                shuffle = inf_shuffle.get(split, split == DataSubset.train)
                max_samples = None
                if split == DataSubset.validate and cfg.max_validation_steps is not None:
                    max_samples = cfg.max_validation_steps * _samples_per_batch(cfg, dataset)
                loaders[split.name] = CachedBasinGraphDataLoader(
                    cfg, dataset, shuffle, max_samples=max_samples
                )

        case "tabular":
            datasets = build_tabular_datasets(cfg)
            for split in splits:
                shuffle = inf_shuffle.get(split, split == DataSubset.train)
                loaders[split.name] = TabularReachDataLoader(cfg, datasets[split], shuffle)

        case _:
            raise ValueError(f"Unknown dataset_type argument: {getattr(cfg, 'dataset_type')}")

    return loaders
