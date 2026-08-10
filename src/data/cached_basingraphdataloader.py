# ============================================================
# cached_basingraphdataloader.py
# ============================================================

import random
from collections import defaultdict, deque
from functools import partial
from typing import Iterator

import numpy as np
import torch
from jaxtyping import Array
from torch.utils.data import DataLoader, Sampler

from config.config import Config

from .cached_basingraphdataset import (
    CachedBasinGraphDataset,
    GraphBatch,
    build_batched_level_structure,
)


class GraphPackingSampler(Sampler[list[int]]):
    """
    A sampler that intelligently packs graphs to a target node count.
    (UNCHANGED - same as your existing code)
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
        self.target_real_nodes = target_nodes_per_batch - 1

        self.samples = []
        for basin, indices in self.basin_index_map.items():
            node_count = self.basin_node_counts.get(basin, 0)
            for index in indices:
                self.samples.append({"index": index, "nodes": node_count})

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
        count = 0
        nodes = [s["nodes"] for s in self.samples]
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
        while True:
            samples_for_epoch = self.samples[:]
            if self.infinite_shuffle:
                random.shuffle(samples_for_epoch)

            samples_deque = deque(samples_for_epoch)
            while samples_deque:
                seed_sample = samples_deque.popleft()
                current_batch_indices = [seed_sample["index"]]
                nodes_in_batch = seed_sample["nodes"]

                while nodes_in_batch < self.target_real_nodes:
                    remaining_space = self.target_real_nodes - nodes_in_batch
                    best_fit_candidate_idx = -1
                    best_fit_nodes = 0

                    pool_size = min(self.candidate_pool_size, len(samples_deque))
                    for i in range(pool_size):
                        candidate = samples_deque[i]
                        if best_fit_nodes < candidate["nodes"] <= remaining_space:
                            best_fit_nodes = candidate["nodes"]
                            best_fit_candidate_idx = i

                    if best_fit_candidate_idx != -1:
                        best_fit_sample = samples_deque[best_fit_candidate_idx]
                        current_batch_indices.append(best_fit_sample["index"])
                        nodes_in_batch += best_fit_sample["nodes"]
                        del samples_deque[best_fit_candidate_idx]
                    else:
                        break

                yield current_batch_indices

            if not self.infinite_shuffle:
                break


def pack_inflow_data(
    samples: list[GraphBatch],
    num_nodes_per_graph: list[int],
    num_real_nodes: int,
    num_padding_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Offsets and pads inflow indices/masks. (UNCHANGED)"""
    node_offset = 0
    offset_inflow_list = []
    inflow_mask_list = []

    for i, sample in enumerate(samples):
        valid_mask = sample.inflow_mask
        current_inflow = np.where(valid_mask, sample.inflow_indices + node_offset, -1)
        offset_inflow_list.append(current_inflow)
        inflow_mask_list.append(valid_mask)
        node_offset += num_nodes_per_graph[i]

    unpadded_inflow = np.concatenate(offset_inflow_list, axis=0)
    unpadded_mask = np.concatenate(inflow_mask_list, axis=0)

    total_nodes = num_real_nodes + num_padding_nodes
    padded_inflow = np.full((total_nodes, 2), num_real_nodes, dtype=np.int32)
    padded_inflow[:num_real_nodes] = np.where(unpadded_mask, unpadded_inflow, num_real_nodes)
    padded_mask = np.zeros((total_nodes, 2), dtype=np.bool_)
    padded_mask[:num_real_nodes] = unpadded_mask

    return padded_inflow, padded_mask


def pack_tree_topology(
    samples: list[GraphBatch],
    num_nodes_per_graph: list[int],
    num_real_nodes: int,
    num_padding_nodes: int,
) -> dict[str, np.ndarray]:
    """
    Pack and pad tree topology arrays from multiple graph samples into a single
    batched structure for the model.

    The key insight: since the downstream scan processes nodes level-by-level and
    different graphs in the batch are disconnected, we can merge nodes from
    different graphs at the same topological level into one scan step. They won't
    interact because their parent indices point within their own graph.

    Args:
        samples: list of GraphBatch samples (each has tree_topology dict)
        num_nodes_per_graph: number of nodes per sample
        num_real_nodes: total real nodes in the batch
        num_padding_nodes: padding nodes to add

    Returns:
        dict with batched topology arrays:
            level_nodes: (max_levels, max_per_level) padded node indices per level
            level_mask: (max_levels, max_per_level) validity mask
            level_parents: (total_padded_nodes, 2) parent indices per node (offset)
            level_parent_mask: (total_padded_nodes, 2) parent validity
            dilated_downstream: (total_padded_nodes, num_dilations) downstream at 2^k
            dilated_valid: (total_padded_nodes, num_dilations) validity
            level_assignments: (total_padded_nodes,) topological level per node
    """
    total_nodes = num_real_nodes + num_padding_nodes

    # --- Per-node arrays: concatenate with offset, then pad ---
    node_offset = 0
    all_level_assignments = []
    all_level_parents = []
    all_level_parent_mask = []
    all_dilated_downstream = []
    all_dilated_valid = []

    for i, sample in enumerate(samples):
        topo = sample.tree_topology
        n_nodes = num_nodes_per_graph[i]

        # Level assignments (no offset needed - just values)
        all_level_assignments.append(topo["level_assignments"])

        # Level parents: offset indices to global batch coordinates
        parents_offset = topo["level_parents"] + node_offset
        # Keep self-references pointing to offset self (for masked slots)
        for node_local in range(n_nodes):
            for slot in range(2):
                if not topo["level_parent_mask"][node_local, slot]:
                    parents_offset[node_local, slot] = node_local + node_offset
        all_level_parents.append(parents_offset)
        all_level_parent_mask.append(topo["level_parent_mask"])

        # Dilated downstream: offset indices
        dilated_offset = topo["dilated_downstream"] + node_offset
        # Invalid slots should point to self (offset)
        for node_local in range(n_nodes):
            for k in range(topo["dilated_valid"].shape[1]):
                if not topo["dilated_valid"][node_local, k]:
                    dilated_offset[node_local, k] = node_local + node_offset
        all_dilated_downstream.append(dilated_offset)
        all_dilated_valid.append(topo["dilated_valid"])

        node_offset += n_nodes

    # Concatenate real data
    concat_level_assignments = np.concatenate(all_level_assignments)
    concat_level_parents = np.concatenate(all_level_parents, axis=0)
    concat_level_parent_mask = np.concatenate(all_level_parent_mask, axis=0)
    concat_dilated_downstream = np.concatenate(all_dilated_downstream, axis=0)
    concat_dilated_valid = np.concatenate(all_dilated_valid, axis=0)

    num_dilations = concat_dilated_downstream.shape[1]

    # Pad to total_nodes (padding nodes get self-referencing indices and False masks)
    padded_level_assignments = np.zeros(total_nodes, dtype=np.int32)
    padded_level_assignments[:num_real_nodes] = concat_level_assignments

    padded_level_parents = np.zeros((total_nodes, 2), dtype=np.int32)
    padded_level_parents[:num_real_nodes] = concat_level_parents
    # Padding nodes point to themselves (num_real_nodes is the first padding node)
    for i in range(num_real_nodes, total_nodes):
        padded_level_parents[i] = i

    padded_level_parent_mask = np.zeros((total_nodes, 2), dtype=np.bool_)
    padded_level_parent_mask[:num_real_nodes] = concat_level_parent_mask

    padded_dilated_downstream = np.zeros((total_nodes, num_dilations), dtype=np.int32)
    padded_dilated_downstream[:num_real_nodes] = concat_dilated_downstream
    for i in range(num_real_nodes, total_nodes):
        padded_dilated_downstream[i] = i  # self-reference

    padded_dilated_valid = np.zeros((total_nodes, num_dilations), dtype=np.bool_)
    padded_dilated_valid[:num_real_nodes] = concat_dilated_valid

    # --- Build level_nodes / level_mask for the scan ---
    max_levels = int(concat_level_assignments.max()) + 1 if num_real_nodes > 0 else 1

    # Find max nodes at any single level across the batch
    level_counts = np.bincount(concat_level_assignments, minlength=max_levels)
    max_nodes_per_level = int(level_counts.max()) if len(level_counts) > 0 else 1

    node_mask = np.zeros(total_nodes, dtype=np.bool_)
    node_mask[:num_real_nodes] = True

    level_nodes, level_mask = build_batched_level_structure(
        padded_level_assignments, node_mask, max_levels, max_nodes_per_level
    )

    return {
        "level_nodes": level_nodes,
        "level_mask": level_mask,
        "level_parents": padded_level_parents,
        "level_parent_mask": padded_level_parent_mask,
        "dilated_downstream": padded_dilated_downstream,
        "dilated_valid": padded_dilated_valid,
        "level_assignments": padded_level_assignments,
    }


def padding_collate_fn(
    batch: list[tuple[dict, GraphBatch]], target_nodes_per_batch: int
) -> tuple[dict, GraphBatch]:
    """
    Collate function that packs multiple graph samples into a single padded batch.
    Now includes tree topology packing.
    """
    metadata, samples = zip(*batch)

    num_nodes_per_graph = [s.dynamic[list(s.dynamic.keys())[0]].shape[1] for s in samples]
    num_real_nodes = sum(num_nodes_per_graph)
    num_padding_nodes = target_nodes_per_batch - num_real_nodes
    assert num_padding_nodes >= 0, "Batch has more nodes than target"

    # --- Step 1: Flatten metadata ---
    flat_basins = []
    flat_dates = []
    flat_subbasins = []
    for i, n_nodes in enumerate(num_nodes_per_graph):
        flat_basins.extend([metadata[i]["basin"]] * n_nodes)
        flat_dates.extend([metadata[i]["date"]] * n_nodes)
        flat_subbasins.extend(metadata[i]["subbasins"])

    # --- Step 2: Pack real data by concatenating ---
    packed_dynamic = defaultdict(list)
    for sample in samples:
        for source, data in sample.dynamic.items():
            packed_dynamic[source].append(data)
    unpadded_dynamic = {
        source: np.concatenate(data_list, axis=1) for source, data_list in packed_dynamic.items()
    }

    packed_targets = defaultdict(list)
    for sample in samples:
        for target, data in sample.y.items():
            packed_targets[target].append(data)
    unpadded_targets = {
        target: np.concatenate(data_list, axis=1) for target, data_list in packed_targets.items()
    }

    unpadded_static = np.concatenate([s.static for s in samples if s.static is not None], axis=0)
    unpadded_graph_idx = np.repeat(np.arange(len(samples)), repeats=np.array(num_nodes_per_graph))

    # Offset and concatenate edges
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

    # --- Step 3: Pad all tensors ---
    padded_static = np.pad(unpadded_static, ((0, num_padding_nodes), (0, 0)))
    padded_graph_idx = np.pad(unpadded_graph_idx, (0, num_padding_nodes))

    padded_dynamic = {
        s: np.pad(d, ((0, 0), (0, num_padding_nodes), (0, 0))) for s, d in unpadded_dynamic.items()
    }
    padded_targets = {
        t: np.pad(d, ((0, 0), (0, num_padding_nodes), (0, 0))) for t, d in unpadded_targets.items()
    }

    padded_edges = np.pad(
        unpadded_edges, ((0, 0), (0, num_padding_edges)), constant_values=num_real_nodes
    )

    # --- Step 4: Create masks ---
    node_mask = np.concatenate(
        [
            np.ones(num_real_nodes, dtype=np.bool_),
            np.zeros(num_padding_nodes, dtype=np.bool_),
        ]
    )
    edge_mask = np.concatenate(
        [
            np.ones(num_real_edges, dtype=np.bool_),
            np.zeros(num_padding_edges, dtype=np.bool_),
        ]
    )

    # --- Step 5: Pack tree topology ---
    tree_topology = None
    if samples[0].tree_topology is not None:
        tree_topology = pack_tree_topology(
            samples, num_nodes_per_graph, num_real_nodes, num_padding_nodes
        )

    # --- Step 6: Optional inflow packing ---
    inflow_indices = None
    inflow_mask = None
    if samples[0].inflow_indices is not None:
        inflow_indices, inflow_mask = pack_inflow_data(
            samples, num_nodes_per_graph, num_real_nodes, num_padding_nodes
        )

    # --- Step 7: Assemble final batch ---
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
        tree_topology=tree_topology,
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
        timeout = 0
        persistent_workers = False

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

    def normalize(self, x: Array, name: str):
        return self.dataset.normalize(x, name)

    def denormalize(self, x: Array, name: str):
        return self.dataset.denormalize(x, name)

    def denormalize_std(self, mu_norm: Array, sigma_norm: Array, name: str):
        return self.dataset.denormalize_std(mu_norm, sigma_norm, name)

    def denormalize_target(self, y_normalized: Array):
        return self.dataset.denormalize_target(y_normalized)
