import numpy as np


def random_walk_subgraph(indptr, indices, seed_nodes, walk_length, num_walks, max_nodes, rng):
    seed_nodes = np.asarray(seed_nodes, dtype=np.int64)
    if seed_nodes.size == 0:
        return np.empty((0,), dtype=np.int64)
    if max_nodes is not None and seed_nodes.size > max_nodes:
        seed_nodes = rng.choice(seed_nodes, size=max_nodes, replace=False)
    visited = set(seed_nodes.tolist())
    max_nodes = None if max_nodes is None else int(max_nodes)

    for seed in seed_nodes:
        if max_nodes is not None and len(visited) >= max_nodes:
            break
        for _ in range(num_walks):
            cur = int(seed)
            for _ in range(walk_length):
                start = indptr[cur]
                end = indptr[cur + 1]
                if end <= start:
                    break
                nxt = int(indices[rng.integers(start, end)])
                if nxt not in visited:
                    visited.add(nxt)
                    if max_nodes is not None and len(visited) >= max_nodes:
                        break
                cur = nxt
            if max_nodes is not None and len(visited) >= max_nodes:
                break

    return np.asarray(sorted(visited), dtype=np.int64)


def build_subgraph_edges(indptr, indices, data, sub_nodes):
    sub_nodes = np.asarray(sub_nodes, dtype=np.int64)
    if sub_nodes.size == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    rows = []
    cols = []
    vals = []
    size = sub_nodes.size
    for local_src, node in enumerate(sub_nodes):
        start = indptr[node]
        end = indptr[node + 1]
        if end <= start:
            continue
        nbrs = indices[start:end]
        weights = None if data is None else data[start:end]
        pos = np.searchsorted(sub_nodes, nbrs)
        mask = pos < size
        if np.any(mask):
            mask[mask] = sub_nodes[pos[mask]] == nbrs[mask]
        if not np.any(mask):
            continue
        dst = pos[mask].astype(np.int64)
        rows.append(np.full(dst.size, local_src, dtype=np.int64))
        cols.append(dst)
        if weights is None:
            vals.append(np.ones(dst.size, dtype=np.float32))
        else:
            vals.append(weights[mask].astype(np.float32))

    if not rows:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals)
    edge_index = np.stack([row, col], axis=0)
    return edge_index, val
