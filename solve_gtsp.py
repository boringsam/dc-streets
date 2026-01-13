import math
import random
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import osmnx as ox
import networkx as nx

# ---------------------------
# Graph and routing setup
# ---------------------------
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

_graph = None
_node_cache = {}  # (lon, lat) -> nearest node id
_dist_cache = {}  # (node1, node2) -> distance in meters


def get_graph():
    """Load or download the DC street network graph."""
    global _graph
    if _graph is not None:
        return _graph

    cache_file = CACHE_DIR / "dc_walk_graph.pkl"

    if cache_file.exists():
        print("Loading cached street network...", flush=True)
        with open(cache_file, "rb") as f:
            _graph = pickle.load(f)
    else:
        print("Downloading DC street network (this may take a minute)...", flush=True)
        _graph = ox.graph_from_place("Washington, DC", network_type="walk")
        with open(cache_file, "wb") as f:
            pickle.dump(_graph, f)
        print("Cached street network for future runs.", flush=True)

    return _graph


def get_nearest_node(lon, lat):
    """Get the nearest graph node to a point."""
    key = (round(lon, 6), round(lat, 6))
    if key in _node_cache:
        return _node_cache[key]

    G = get_graph()
    node = ox.distance.nearest_nodes(G, lon, lat)
    _node_cache[key] = node
    return node


def street_distance_m(lon1, lat1, lon2, lat2):
    """
    Get the shortest path distance along streets between two points.
    Falls back to haversine if no path exists.
    """
    node1 = get_nearest_node(lon1, lat1)
    node2 = get_nearest_node(lon2, lat2)

    if node1 == node2:
        return 0.0

    cache_key = (min(node1, node2), max(node1, node2))
    if cache_key in _dist_cache:
        return _dist_cache[cache_key]

    G = get_graph()
    try:
        dist = nx.shortest_path_length(G, node1, node2, weight="length")
    except nx.NetworkXNoPath:
        # Fall back to haversine if no path
        dist = haversine_m(lon1, lat1, lon2, lat2) * 1.4  # Penalize for no path

    _dist_cache[cache_key] = dist
    return dist


def haversine_m(lon1, lat1, lon2, lat2):
    """Fallback: straight-line distance in meters."""
    rlon1, rlat1, rlon2, rlat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371000.0 * c


# ---------------------------
# Parse TSV/CSV
# ---------------------------
def load_points(path):
    """
    Accepts tab-delimited or comma-delimited with header: state lon lat
    Returns dict[state] -> list[(lon, lat)]
    """
    clusters = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if not header:
            raise ValueError("Empty file")

        delim = "\t" if "\t" in header else ","
        cols = [c.strip().lower() for c in header.split(delim)]
        if cols[:3] != ["state", "lon", "lat"]:
            raise ValueError(f"Expected header: state{delim}lon{delim}lat. Got: {header}")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delim)
            if len(parts) < 3:
                continue
            state = parts[0].strip()
            lon = float(parts[1])
            lat = float(parts[2])
            clusters[state].append((lon, lat))

    clusters = dict(clusters)
    if len(clusters) < 2:
        raise ValueError("Need at least 2 states/clusters")
    return clusters


# ---------------------------
# 2-opt for a cycle
# ---------------------------
def route_length(order, chosen_points):
    """Calculate total cycle length using street distances."""
    total = 0.0
    n = len(order)
    for i in range(n):
        a = chosen_points[order[i]]
        b = chosen_points[order[(i+1) % n]]
        total += street_distance_m(a[0], a[1], b[0], b[1])
    return total


def two_opt(order, chosen_points, max_iters=2000):
    n = len(order)
    if n < 4:
        return order

    improved = True
    iters = 0
    best = order[:]
    best_len = route_length(best, chosen_points)

    while improved and iters < max_iters:
        improved = False
        iters += 1

        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new = best[:]
                new[i:k+1] = reversed(new[i:k+1])
                new_len = route_length(new, chosen_points)
                if new_len + 1e-9 < best_len:
                    best = new
                    best_len = new_len
                    improved = True
                    break
            if improved:
                break

    return best


# ---------------------------
# Best point choice per state given neighbors
# ---------------------------
def best_candidate_for_state(candidates, prev_pt, next_pt):
    best_pt = None
    best_cost = float("inf")
    for lon, lat in candidates:
        cost = street_distance_m(prev_pt[0], prev_pt[1], lon, lat) + \
               street_distance_m(lon, lat, next_pt[0], next_pt[1])
        if cost < best_cost:
            best_cost = cost
            best_pt = (lon, lat)
    return best_pt


def optimize_points_given_order(order, clusters, chosen_points):
    """
    For each state in the tour, pick the point that minimizes distance to its neighbors.
    Updates chosen_points in place.
    """
    n = len(order)
    for idx, state in enumerate(order):
        prev_state = order[(idx - 1) % n]
        next_state = order[(idx + 1) % n]
        prev_pt = chosen_points[prev_state]
        next_pt = chosen_points[next_state]
        chosen_points[state] = best_candidate_for_state(clusters[state], prev_pt, next_pt)


# ---------------------------
# Main GTSP-style local search with restarts
# ---------------------------
def solve(clusters, restarts=30, point_iters=30, seed=0):
    random.seed(seed)
    states = list(clusters.keys())

    # Pre-warm the graph
    get_graph()
    print(f"Starting optimization with {len(states)} states...", flush=True)

    best_order = None
    best_chosen = None
    best_len = float("inf")

    for r in range(restarts):
        # random initial point per state
        chosen = {s: random.choice(clusters[s]) for s in states}

        # random initial order
        order = states[:]
        random.shuffle(order)

        # alternating optimization
        last_len = None
        for it in range(point_iters):
            order = two_opt(order, chosen, max_iters=500)  # Reduced iters since routing is slower
            optimize_points_given_order(order, clusters, chosen)
            cur_len = route_length(order, chosen)
            if last_len is not None and abs(last_len - cur_len) < 1e-3:
                break
            last_len = cur_len

        cur_len = route_length(order, chosen)
        if cur_len < best_len:
            best_len = cur_len
            best_order = order[:]
            best_chosen = dict(chosen)

        print(f"restart {r+1}/{restarts}: route_length_km={cur_len/1000:.2f}")

    return best_order, best_chosen, best_len


def cut_cycle_to_open_path(order, chosen_points):
    """
    Convert a cycle to an open path by cutting the *largest* edge.
    Returns path order (no wraparound).
    """
    n = len(order)
    max_edge = -1.0
    cut_idx = 0
    for i in range(n):
        a = chosen_points[order[i]]
        b = chosen_points[order[(i+1) % n]]
        d = street_distance_m(a[0], a[1], b[0], b[1])
        if d > max_edge:
            max_edge = d
            cut_idx = i

    start = (cut_idx + 1) % n
    path = [order[(start + i) % n] for i in range(n)]
    return path, max_edge


def path_length(path, chosen_points):
    total = 0.0
    for i in range(len(path) - 1):
        a = chosen_points[path[i]]
        b = chosen_points[path[i+1]]
        total += street_distance_m(a[0], a[1], b[0], b[1])
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TSV/CSV with columns: state, lon, lat")
    ap.add_argument("--restarts", type=int, default=10, help="Number of random restarts (default: 10)")
    ap.add_argument("--iters", type=int, default=10, help="Alternating iterations per restart (default: 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--open_path", action="store_true", help="Output an open path by cutting the longest edge")
    ap.add_argument("--output", default="gtsp_solution_points.tsv")
    args = ap.parse_args()

    clusters = load_points(args.input)
    print(f"Loaded {len(clusters)} states, {sum(len(v) for v in clusters.values())} total points")

    order, chosen, cyc_len = solve(clusters, restarts=args.restarts, point_iters=args.iters, seed=args.seed)
    print(f"\nBest cycle length (street distance): {cyc_len/1000:.2f} km")

    if args.open_path:
        path, cut_edge = cut_cycle_to_open_path(order, chosen)
        pl = path_length(path, chosen)
        print(f"Open path length: {pl/1000:.2f} km (cut edge was {cut_edge/1000:.2f} km)")
        out_order = path
    else:
        out_order = order

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("visit_order\tstate\tlon\tlat\n")
        for i, state in enumerate(out_order, start=1):
            lon, lat = chosen[state]
            f.write(f"{i}\t{state}\t{lon}\t{lat}\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
