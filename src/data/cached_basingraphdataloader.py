import random
from collections import defaultdict, deque
from functools import partial
from typing import Iterator

import jax
import numpy as np
import torch
from jaxtyping import Array
from torch.utils.data import DataLoader, Sampler

from config.config import Config

from .cached_basingraphdataset import CachedBasinGraphDataset, GraphBatch


class GraphPackingSampler(Sampler[list[int]]):
    """
    A sampler that intelligently packs graphs to a target node count.

    This sampler continuously yields batches and re-shuffles the data
    when one pass is complete. It uses a "best-fit" heuristic to fill batches,
    reducing sampling bias against large graphs and minimizing wasted space.
    """

    def __init__(
        self,
        basin_index_map: dict[str, list[int]],
        basin_node_counts: dict[str, int],
        target_nodes_per_batch: int,
        candidate_pool_size: int,
        infinite_shuffle: bool,
    ):
        self.basin_index_map = basin_index_map
        self.basin_node_counts = basin_node_counts
        self.target_nodes = target_nodes_per_batch
        self.candidate_pool_size = candidate_pool_size
        self.infinite_shuffle = infinite_shuffle

        # Ensure there is room for at least one padding node per batch.
        # Otherwise, the padded edges will have to point at real nodes, which even if
        # mask the results out later can cause gradient problems.
        self.target_real_nodes = target_nodes_per_batch - 1

        self.samples = []
        for basin, indices in self.basin_index_map.items():
            node_count = self.basin_node_counts.get(basin, 0)
            for index in indices:
                self.samples.append({"index": index, "nodes": node_count})

        # Check for any basin(s) that would be excluded from sampling
        oversized_basins = {
            basin: count
            for basin, count in self.basin_node_counts.items()
            if count > self.target_real_nodes
        }
        if oversized_basins:
            raise ValueError(
                "Arg target_nodes_per_batch cannot be less than the node count for any basin. "
                f"Found {len(oversized_basins)} basins with more than {self.target_real_nodes} nodes.\n"
                f"{oversized_basins=}"
            )

        self._len = int(1e12) if self.infinite_shuffle else self._calculate_actual_len()

    def _calculate_actual_len(self) -> int:
        """Simulation of the __iter__ logic to determine the exact batch count."""
        count = 0
        # Use a list of node counts to simulate the deque
        nodes = [s["nodes"] for s in self.samples]
        # Note: If infinite_shuffle is False, we use the original order
        dq = deque(nodes)
        while dq:
            current_nodes = dq.popleft()
            while current_nodes < self.target_real_nodes:
                remaining = self.target_real_nodes - current_nodes
                best_fit_idx = -1
                best_fit_val = 0
                pool_size = min(self.candidate_pool_size, len(dq))
                for i in range(pool_size):
                    if best_fit_val < dq[i] <= remaining:
                        best_fit_val = dq[i]
                        best_fit_idx = i
                if best_fit_idx != -1:
                    current_nodes += dq[best_fit_idx]
                    del dq[best_fit_idx]
                else:
                    break
            count += 1
        return count

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[list[int]]:
        # Yields a list of basin/date indices that will make up a batch
        while True:
            # Runs indefinitely if self.infinite_shuffle is true.
            samples_for_epoch = self.samples[:]
            if self.infinite_shuffle:
                random.shuffle(samples_for_epoch)

            samples_deque = deque(samples_for_epoch)
            while samples_deque:
                # 1. Seed the batch with the next available sample
                seed_sample = samples_deque.popleft()
                current_batch_indices = [seed_sample["index"]]
                nodes_in_batch = seed_sample["nodes"]

                # 2. Intelligently fill the rest of the batch
                while nodes_in_batch < self.target_real_nodes:
                    remaining_space = self.target_real_nodes - nodes_in_batch
                    best_fit_candidate_idx = -1
                    best_fit_nodes = 0

                    # 3. Look ahead at a pool of candidates
                    pool_size = min(self.candidate_pool_size, len(samples_deque))
                    for i in range(pool_size):
                        candidate = samples_deque[i]
                        if best_fit_nodes < candidate["nodes"] <= remaining_space:
                            best_fit_nodes = candidate["nodes"]
                            best_fit_candidate_idx = i

                    # 4. If a suitable candidate was found, add it to the batch
                    if best_fit_candidate_idx != -1:
                        best_fit_sample = samples_deque[best_fit_candidate_idx]
                        current_batch_indices.append(best_fit_sample["index"])
                        nodes_in_batch += best_fit_sample["nodes"]
                        del samples_deque[best_fit_candidate_idx]
                    else:
                        break  # No candidate in the pool fits, finalize the batch

                yield current_batch_indices

            if not self.infinite_shuffle:
                break


def pack_inflow_data(
    samples: list[GraphBatch],
    num_nodes_per_graph: list[int],
    num_real_nodes: int,
    num_padding_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Offsets and pads inflow indices/masks for models requiring fixed-degree adjacency.
    """
    node_offset = 0
    offset_inflow_list = []
    inflow_mask_list = []

    for i, sample in enumerate(samples):
        valid_mask = sample.inflow_mask
        # Offset valid indices; point invalid slots to -1 temporarily
        current_inflow = np.where(valid_mask, sample.inflow_indices + node_offset, -1)

        offset_inflow_list.append(current_inflow)
        inflow_mask_list.append(valid_mask)
        node_offset += num_nodes_per_graph[i]

    unpadded_inflow = np.concatenate(offset_inflow_list, axis=0)
    unpadded_inflow_mask = np.concatenate(inflow_mask_list, axis=0)

    # Point all invalid/empty slots to the first padding node (num_real_nodes)
    # This prevents OOB and ensures gather() hits a masked padding feature row
    unpadded_inflow = np.where(unpadded_inflow_mask, unpadded_inflow, num_real_nodes)

    padded_inflow = np.pad(
        unpadded_inflow, ((0, num_padding_nodes), (0, 0)), constant_values=num_real_nodes
    )
    padded_inflow_mask = np.pad(
        unpadded_inflow_mask, ((0, num_padding_nodes), (0, 0)), constant_values=False
    )

    return padded_inflow, padded_inflow_mask


def padding_collate_fn(
    batch: list[tuple[str, list[str], np.datetime64, GraphBatch]], target_nodes_per_batch: int
) -> GraphBatch:
    """
    Collates (basin, time) samples and then pads them to a fixed size.
    """
    metadata, samples = zip(*batch)  # List of tuples -> lists

    # Step 1: Pre-calculate sizes from all samples in the batch
    num_nodes_per_graph = [s.static.shape[0] for s in samples]
    num_real_nodes = sum(num_nodes_per_graph)
    num_padding_nodes = target_nodes_per_batch - num_real_nodes
    assert num_padding_nodes >= 0, "batch was created with more nodes than target"

    # We repeat the basin/date for every node in the graph so indices align with tensors.
    # We are NOT padding them as that is only necessary for the tensors passed into the model.
    # These are just metadata to track the IDs etc.
    flat_basins = []
    flat_dates = []
    flat_subbasins = []
    for i, n_nodes in enumerate(num_nodes_per_graph):
        flat_basins.extend([metadata[i]['basin']] * n_nodes)
        flat_dates.extend([metadata[i]['date']] * n_nodes)
        flat_subbasins.extend(metadata[i]['subbasins'])

    # Step 2: Pack the "real" data by concatenating
    # Dynamic features
    packed_dynamic = defaultdict(list)
    for sample in samples:
        for source, data in sample.dynamic.items():
            packed_dynamic[source].append(data)
    unpadded_dynamic = {
        source: np.concatenate(data_list, axis=1) for source, data_list in packed_dynamic.items()
    }

    # Targets
    packed_targets = defaultdict(list)
    for sample in samples:
        for target, data in sample.y.items():
            packed_targets[target].append(data)
    unpadded_targets = {
        target: np.concatenate(data_list, axis=1) for target, data_list in packed_targets.items()
    }

    # Static features and Targets
    unpadded_static = np.concatenate([s.static for s in samples if s.static is not None], axis=0)
    unpadded_graph_idx = np.repeat(np.arange(len(samples)), repeats=np.array(num_nodes_per_graph))

    node_offset = 0
    offset_edges_list = []
    for i, sample in enumerate(samples):
        offset_edges_list.append(sample.graph_edges + node_offset)
        node_offset += num_nodes_per_graph[i]

    unpadded_edges = np.concatenate(offset_edges_list, axis=1)
    num_real_edges = unpadded_edges.shape[1]
    target_edges_per_batch = target_nodes_per_batch
    num_padding_edges = target_edges_per_batch - num_real_edges
    assert num_padding_edges >= 0, "Batch has more edges than target"

    # --- Step 3: Pad All Tensors to Target Size ---
    # Pad node features
    padded_static = np.pad(unpadded_static, ((0, num_padding_nodes), (0, 0)))
    # padded_target = np.pad(unpadded_y, ((0, 0), (0, num_padding_nodes), (0, 0)))
    padded_graph_idx = np.pad(unpadded_graph_idx, (0, num_padding_nodes))

    padded_dynamic = {
        s: np.pad(d, ((0, 0), (0, num_padding_nodes), (0, 0))) for s, d in unpadded_dynamic.items()
    }
    padded_targets = {
        t: np.pad(d, ((0, 0), (0, num_padding_nodes), (0, 0))) for t, d in unpadded_targets.items()
    }

    # Pad with the index of the FIRST PADDED NODE (num_real_nodes).
    # This creates harmless self-loops on a non-existent node.
    padded_edges = np.pad(
        unpadded_edges, ((0, 0), (0, num_padding_edges)), constant_values=num_real_nodes
    )

    # --- Step 4: Create Final Masks ---
    node_mask = np.concatenate(
        [np.ones(num_real_nodes, dtype=np.bool_), np.zeros(num_padding_nodes, dtype=np.bool_)]
    )
    edge_mask = np.concatenate(
        [np.ones(num_real_edges, dtype=np.bool_), np.zeros(num_padding_edges, dtype=np.bool_)]
    )

    # Optional Inflow Packing
    inflow_indices = None
    inflow_mask = None
    if samples[0].inflow_indices is not None:
        inflow_indices, inflow_mask = pack_inflow_data(
            samples, num_nodes_per_graph, num_real_nodes, num_padding_nodes
        )

    # --- Step 5: Assemble Final Batch ---
    final_batch = GraphBatch(
        dynamic=padded_dynamic,
        static=padded_static,
        graph_edges=padded_edges,
        inflow_indices=inflow_indices,
        inflow_mask=inflow_mask,
        graph_idx=padded_graph_idx,
        node_mask=node_mask,
        edge_mask=edge_mask,
        y=padded_targets,
    )

    batch_metadata = {
        "basins": flat_basins,
        "subbasins": flat_subbasins,
        "dates": flat_dates,
    }

    return batch_metadata, final_batch


class CachedBasinGraphDataLoader(DataLoader):
    def __init__(
        self, cfg: Config, dataset: CachedBasinGraphDataset, infinite_shuffle: bool = True
    ):
        torch.manual_seed(cfg.model_args.seed)

        collate_fn = partial(padding_collate_fn, target_nodes_per_batch=cfg.target_nodes_per_batch)

        batch_sampler = GraphPackingSampler(
            dataset.basin_index_map,
            dataset.basin_subbasin_counts,
            cfg.target_nodes_per_batch,
            cfg.candidate_pool_size,
            infinite_shuffle,
        )

        threading = cfg.num_workers > 0
        timeout = 0  # 900 if threading else 0
        persistent_workers = False
        # persistent_workers = True if threading else False

        super().__init__(
            dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            timeout=timeout,
            persistent_workers=persistent_workers,
            prefetch_factor=1 if cfg.num_workers > 0 else None,
        )
        print(f"Dataloader using {self.num_workers} parallel CPU worker(s).")

    # Expose these dataset methods for convenience.
    def normalize(self, x: Array, name: str):
        return self.dataset.normalize(x, name)

    def denormalize(self, x: Array, name: str):
        return self.dataset.denormalize(x, name)

    def denormalize_std(self, mu_norm: Array, sigma_norm: Array, name: str):
        return self.dataset.denormalize_std(mu_norm, sigma_norm, name)

    def denormalize_target(self, y_normalized: Array):
        return self.dataset.denormalize_target(y_normalized)


def _get_available_devices():
    """
    Returns a dict of number of available backend devices
    """
    devices = {}
    for backend in ["cpu", "gpu", "tpu"]:
        try:
            n = jax.local_device_count(backend=backend)
            devices[backend] = n
        except RuntimeError:
            pass
    return devices
