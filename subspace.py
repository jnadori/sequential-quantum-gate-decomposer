from collections import defaultdict
from math import ceil
import heapq


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def precompute_edge_masks(edges, vertices):
    """
    Represent each edge as a bitmask of its vertices.
    Vertex v gets bit position vertex_to_bit[v].
    """
    vertex_to_bit = {v: i for i, v in enumerate(sorted(vertices))}
    return [
        (1 << vertex_to_bit[e[0]]) | (1 << vertex_to_bit[e[1]])
        for e in edges
    ]


def precompute_thresholds(num_vertices):
    return [
        (n, ceil((4**n - 3 * n - 1) / 4) + 1)
        for n in range(2, num_vertices)
    ]


# ---------------------------------------------------------------------------
# Canonical form via dependency DAG (bitmask version)
# ---------------------------------------------------------------------------

def canonical_form(seq, masks):
    if not seq:
        return ()

    n = len(seq)
    adj = [[] for _ in range(n)]
    in_degree = [0] * n

    for i in range(n):
        for j in range(i):
            if masks[j] & masks[i]:  # shared vertex = dependent
                adj[j].append(i)
                in_degree[i] += 1

    heap = [(seq[i], i) for i in range(n) if in_degree[i] == 0]
    heapq.heapify(heap)

    result = []
    while heap:
        edge, idx = heapq.heappop(heap)
        result.append(edge)
        for neighbor in adj[idx]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, (seq[neighbor], neighbor))

    return tuple(result)


# ---------------------------------------------------------------------------
# Subspace check (bitmask version)
# ---------------------------------------------------------------------------

def check_new_position(window_masks, pos, thresholds):
    """
    Check windows ending at pos. Uses OR accumulation of bitmasks
    and popcount instead of set construction.
    """
    for n, j in thresholds:
        if j > pos + 1:
            continue

        combined = 0
        for i in range(pos - j + 1, pos + 1):
            combined |= window_masks[i]

        if bin(combined).count('1') <= n:
            return True

    return False


# ---------------------------------------------------------------------------
# Canonical prefix check (bitmask version)
# ---------------------------------------------------------------------------

def is_canonical_prefix(path, path_masks, depth):
    i = depth
    while i > 0:
        if path_masks[i] & path_masks[i - 1]:
            break  # dependent — order fixed
        if path[i] < path[i - 1]:
            return False  # independent and out of order
        i -= 1
    return True


# ---------------------------------------------------------------------------
# DFS generators
# ---------------------------------------------------------------------------

def unique_k_sequences(topology, k):
    edges = sorted(set(tuple(e) for e in topology))
    vertices = {v for edge in edges for v in edge}
    num_vertices = len(vertices)
    edge_masks = precompute_edge_masks(edges, vertices)
    thresholds = precompute_thresholds(num_vertices)

    # Map each edge to its precomputed mask for fast lookup
    edge_to_mask = {e: m for e, m in zip(edges, edge_masks)}

    seen = set()
    results = []
    path = [None] * k
    path_masks = [0] * k

    def dfs(depth):
        if depth == k:
            seq = tuple(path)
            masks = path_masks[:k]
            canon = canonical_form(seq, masks)
            if canon not in seen:
                seen.add(canon)
                results.append(canon)
            return

        for edge, mask in zip(edges, edge_masks):
            path[depth] = edge
            path_masks[depth] = mask

            if check_new_position(path_masks, depth, thresholds):
                continue
            if not is_canonical_prefix(path, path_masks, depth):
                continue

            dfs(depth + 1)

    dfs(0)
    return results


def unique_k_sequences_excluding(topology, k, excluded):
    edges = sorted(set(tuple(e) for e in topology))
    vertices = {v for edge in edges for v in edge}
    num_vertices = len(vertices)
    edge_masks = precompute_edge_masks(edges, vertices)
    thresholds = precompute_thresholds(num_vertices)

    edge_to_mask = {e: m for e, m in zip(edges, edge_masks)}

    excluded_canonical = set()
    for seq in excluded:
        seq = tuple(tuple(e) for e in seq)
        masks = [edge_to_mask[e] for e in seq]
        excluded_canonical.add(canonical_form(seq, masks))

    seen = set()
    results = []
    path = [None] * k
    path_masks = [0] * k

    def dfs(depth):
        if depth == k:
            seq = tuple(path)
            masks = path_masks[:k]
            canon = canonical_form(seq, masks)
            if canon not in seen and canon not in excluded_canonical:
                seen.add(canon)
                results.append(canon)
            return

        for edge, mask in zip(edges, edge_masks):
            path[depth] = edge
            path_masks[depth] = mask

            if check_new_position(path_masks, depth, thresholds):
                continue
            if not is_canonical_prefix(path, path_masks, depth):
                continue

            dfs(depth + 1)

    dfs(0)
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    topology = [(0, 2), (1, 0)]
    k = 15

    vertices = {v for e in topology for v in e}
    print("Subspace thresholds:")
    for n in range(2, len(vertices)):
        j = ceil((4**n - 3 * n - 1) / 4) + 1
        print(f"  n={n} vertices -> j={j}")

    seqs = unique_k_sequences(topology, k)
    print(f"\nk={k}: {len(seqs)} canonically unique sequences\n")
    for s in sorted(seqs):
        print(f"  {s}")