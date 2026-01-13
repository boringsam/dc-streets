import streamlit as st
import folium
from streamlit_folium import st_folium
import math
import random
import pickle
from pathlib import Path
from collections import defaultdict

import osmnx as ox
import networkx as nx

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="DC State Streets", page_icon="🗺️", layout="wide")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------
# Graph and routing (cached)
# ---------------------------
@st.cache_resource
def get_graph():
    """Load or download the DC street network graph."""
    cache_file = CACHE_DIR / "dc_walk_graph.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        with st.spinner("Downloading DC street network (one-time, ~1 min)..."):
            graph = ox.graph_from_place("Washington, DC", network_type="walk")
            with open(cache_file, "wb") as f:
                pickle.dump(graph, f)
            return graph


@st.cache_data
def load_state_points():
    """Load pre-generated state street points."""
    clusters = defaultdict(list)

    # Try to load from file
    for fname in ["dc_state_points.csv", "dc_state_named_street_points_100m.csv"]:
        if Path(fname).exists():
            with open(fname, "r") as f:
                header = f.readline()
                delim = "\t" if "\t" in header else ","
                for line in f:
                    parts = line.strip().split(delim)
                    if len(parts) >= 3:
                        state = parts[0].strip()
                        lon = float(parts[1])
                        lat = float(parts[2])
                        clusters[state].append((lon, lat))
            return dict(clusters)

    # Generate if not found
    return generate_state_points()


@st.cache_data
def generate_state_points():
    """Generate state street points from OSM data."""
    with st.spinner("Finding state-named streets in DC..."):
        dc_polygon = ox.geocode_to_gdf("Washington, DC").geometry.iloc[0]

        states = [
            "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
            "Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky",
            "Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi",
            "Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico",
            "New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
            "Rhode Island","South Carolina","South Dakota","Tennessee","Texas",
            "Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"
        ]

        highways = ["trunk","primary","secondary","tertiary","residential","unclassified","service","road"]

        roads = ox.features.features_from_polygon(dc_polygon, tags={"highway": highways})
        roads = roads[roads.geom_type == "LineString"]
        roads = roads[roads["name"].notna()]

        import geopandas as gpd

        clusters = {}
        for state in states:
            matches = roads[roads["name"].str.contains(state, case=False, regex=False)]
            if matches.empty:
                continue

            merged = matches.geometry.union_all()
            merged_m = gpd.GeoSeries([merged], crs="EPSG:4326").to_crs("EPSG:32618").iloc[0]

            pts = []
            segments = list(merged_m.geoms) if hasattr(merged_m, "geoms") else [merged_m]

            for seg in segments:
                d = 0
                while d <= seg.length:
                    p = seg.interpolate(d)
                    p_ll = gpd.GeoSeries([p], crs="EPSG:32618").to_crs("EPSG:4326").iloc[0]
                    pts.append((round(p_ll.x, 6), round(p_ll.y, 6)))
                    d += 100

            pts = list(dict.fromkeys(pts))
            if pts:
                clusters[state] = pts

        return clusters


# Distance caches
_node_cache = {}
_dist_cache = {}


def get_nearest_node(G, lon, lat):
    key = (round(lon, 6), round(lat, 6))
    if key not in _node_cache:
        _node_cache[key] = ox.distance.nearest_nodes(G, lon, lat)
    return _node_cache[key]


def street_distance_m(G, lon1, lat1, lon2, lat2):
    node1 = get_nearest_node(G, lon1, lat1)
    node2 = get_nearest_node(G, lon2, lat2)

    if node1 == node2:
        return 0.0

    cache_key = (min(node1, node2), max(node1, node2))
    if cache_key not in _dist_cache:
        try:
            _dist_cache[cache_key] = nx.shortest_path_length(G, node1, node2, weight="length")
        except nx.NetworkXNoPath:
            # Fallback to haversine with penalty
            rlon1, rlat1, rlon2, rlat2 = map(math.radians, [lon1, lat1, lon2, lat2])
            dlon, dlat = rlon2 - rlon1, rlat2 - rlat1
            a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
            _dist_cache[cache_key] = 6371000.0 * 2 * math.asin(math.sqrt(a)) * 1.4

    return _dist_cache[cache_key]


def route_length(G, order, chosen):
    total = 0.0
    n = len(order)
    for i in range(n):
        a, b = chosen[order[i]], chosen[order[(i+1) % n]]
        total += street_distance_m(G, a[0], a[1], b[0], b[1])
    return total


def two_opt(G, order, chosen, max_iters=200):
    n = len(order)
    if n < 4:
        return order

    best = order[:]
    best_len = route_length(G, best, chosen)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new = best[:]
                new[i:k+1] = reversed(new[i:k+1])
                new_len = route_length(G, new, chosen)
                if new_len + 1e-9 < best_len:
                    best, best_len = new, new_len
                    improved = True
                    break
            if improved:
                break
    return best


def best_candidate(G, candidates, prev_pt, next_pt):
    best_pt, best_cost = None, float("inf")
    for lon, lat in candidates:
        cost = street_distance_m(G, prev_pt[0], prev_pt[1], lon, lat) + \
               street_distance_m(G, lon, lat, next_pt[0], next_pt[1])
        if cost < best_cost:
            best_cost, best_pt = cost, (lon, lat)
    return best_pt


def optimize_points(G, order, clusters, chosen):
    n = len(order)
    for idx, state in enumerate(order):
        prev_pt = chosen[order[(idx - 1) % n]]
        next_pt = chosen[order[(idx + 1) % n]]
        chosen[state] = best_candidate(G, clusters[state], prev_pt, next_pt)


def solve(G, clusters, restarts, iters, seed, progress_bar):
    random.seed(seed)
    states = list(clusters.keys())

    best_order, best_chosen, best_len = None, None, float("inf")

    for r in range(restarts):
        chosen = {s: random.choice(clusters[s]) for s in states}
        order = states[:]
        random.shuffle(order)

        for _ in range(iters):
            order = two_opt(G, order, chosen, max_iters=100)
            optimize_points(G, order, clusters, chosen)

        cur_len = route_length(G, order, chosen)
        if cur_len < best_len:
            best_len, best_order, best_chosen = cur_len, order[:], dict(chosen)

        progress_bar.progress((r + 1) / restarts, f"Restart {r+1}/{restarts}: {cur_len/1000:.1f} km")

    return best_order, best_chosen, best_len


def cut_to_open_path(G, order, chosen):
    n = len(order)
    max_edge, cut_idx = -1, 0
    for i in range(n):
        a, b = chosen[order[i]], chosen[order[(i+1) % n]]
        d = street_distance_m(G, a[0], a[1], b[0], b[1])
        if d > max_edge:
            max_edge, cut_idx = d, i

    start = (cut_idx + 1) % n
    return [order[(start + i) % n] for i in range(n)]


# ---------------------------
# UI
# ---------------------------
st.title("🗺️ DC State Streets Route Finder")
st.markdown("Find the optimal walking route to visit all state-named streets in Washington DC.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    restarts = st.slider("Optimization restarts", 1, 20, 5)
    iters = st.slider("Iterations per restart", 1, 20, 5)
    seed = st.number_input("Random seed", 0, 1000, 42)
    open_path = st.checkbox("Open path (not a loop)", value=True)

    run_button = st.button("🚀 Find Route", type="primary", use_container_width=True)

# Load data
clusters = load_state_points()

if not clusters:
    st.error("Could not load state street points. Please run dc_streets.py first.")
    st.stop()

st.success(f"Loaded {len(clusters)} state streets with {sum(len(v) for v in clusters.values())} points")

# Run solver
if run_button:
    G = get_graph()

    progress = st.progress(0, "Starting optimization...")
    order, chosen, length = solve(G, clusters, restarts, iters, seed, progress)

    if open_path:
        order = cut_to_open_path(G, order, chosen)
        # Recalculate open path length
        length = sum(
            street_distance_m(G, chosen[order[i]][0], chosen[order[i]][1],
                            chosen[order[i+1]][0], chosen[order[i+1]][1])
            for i in range(len(order) - 1)
        )

    st.session_state.order = order
    st.session_state.chosen = chosen
    st.session_state.length = length

# Display results
if "order" in st.session_state:
    order = st.session_state.order
    chosen = st.session_state.chosen
    length = st.session_state.length

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Route: {length/1000:.2f} km")

        # Create map
        center_lat = sum(chosen[s][1] for s in order) / len(order)
        center_lon = sum(chosen[s][0] for s in order) / len(order)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # Add route line
        route_coords = [(chosen[s][1], chosen[s][0]) for s in order]
        folium.PolyLine(route_coords, weight=3, color="blue", opacity=0.8).add_to(m)

        # Add markers
        for i, state in enumerate(order, 1):
            lon, lat = chosen[state]
            folium.Marker(
                [lat, lon],
                popup=f"{i}. {state}",
                tooltip=f"{i}. {state}",
                icon=folium.DivIcon(html=f'<div style="font-size:10px;background:white;border-radius:50%;width:20px;height:20px;text-align:center;line-height:20px;border:1px solid blue;">{i}</div>')
            ).add_to(m)

        st_folium(m, width=700, height=500)

    with col2:
        st.subheader("Visit Order")
        for i, state in enumerate(order, 1):
            lon, lat = chosen[state]
            st.write(f"**{i}.** {state}")

        # Download button
        tsv_data = "visit_order\tstate\tlon\tlat\n"
        for i, state in enumerate(order, 1):
            lon, lat = chosen[state]
            tsv_data += f"{i}\t{state}\t{lon}\t{lat}\n"

        st.download_button(
            "📥 Download Route (TSV)",
            tsv_data,
            "dc_state_streets_route.tsv",
            "text/tab-separated-values"
        )

else:
    # Show preview map with all points
    st.subheader("State Street Locations")

    all_points = [(state, lon, lat) for state, pts in clusters.items() for lon, lat in pts]
    center_lat = sum(p[2] for p in all_points) / len(all_points)
    center_lon = sum(p[1] for p in all_points) / len(all_points)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightblue", "darkgreen", "cadetblue", "pink"]
    for i, (state, pts) in enumerate(clusters.items()):
        color = colors[i % len(colors)]
        for lon, lat in pts[:10]:  # Limit points for performance
            folium.CircleMarker([lat, lon], radius=3, color=color, fill=True,
                              popup=state, tooltip=state).add_to(m)

    st_folium(m, width=700, height=500)

    st.info("👆 Click 'Find Route' in the sidebar to calculate the optimal walking route.")
