import osmnx as ox
import geopandas as gpd
import csv

print("osmnx version:", ox.__version__)
print("Script started", flush=True)

# DC polygon
dc_polygon = ox.geocode_to_gdf("Washington, DC").geometry.iloc[0]

# US states to search for (keys only)
states = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
    "Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky",
    "Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi",
    "Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico",
    "New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
    "Puerto Rico","Rhode Island","South Carolina","South Dakota","Tennessee","Texas",
    "Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"
]

HIGHWAYS = [
    "trunk","primary","secondary","tertiary",
    "residential","unclassified","service","living_street","road"
]

DISTANCE_M = 100
WGS84 = "EPSG:4326"
METRIC = "EPSG:32618"  # UTM zone for DC

# ---- helpers ----

def to_metric(geom):
    return gpd.GeoSeries([geom], crs=WGS84).to_crs(METRIC).iloc[0]

def to_wgs84(pt):
    return gpd.GeoSeries([pt], crs=METRIC).to_crs(WGS84).iloc[0]

# ---- 1) Fetch all DC roads ONCE ----

print("Downloading DC road network...", flush=True)

roads = ox.features.features_from_polygon(
    dc_polygon,
    tags={"highway": HIGHWAYS}
)

roads = roads[roads.geom_type == "LineString"]
roads = roads[roads["name"].notna()]

print(f"Total named road segments: {len(roads)}", flush=True)

# ---- 2) Process per state ----

street_points = {}
missing = []

for state in states:
    print(f"\nProcessing {state}", flush=True)

    matches = roads[
        roads["name"].str.contains(state, case=False, regex=False)
    ]

    if matches.empty:
        print("  No matches", flush=True)
        street_points[state] = []
        missing.append(state)
        continue

    merged = matches.geometry.union_all()
    merged_m = to_metric(merged)

    pts = []
    segments = list(merged_m.geoms) if hasattr(merged_m, "geoms") else [merged_m]

    for seg in segments:
        d = 0
        while d <= seg.length:
            p = seg.interpolate(d)
            p_ll = to_wgs84(p)
            pts.append((round(p_ll.x, 6), round(p_ll.y, 6)))
            d += DISTANCE_M

    pts = list(dict.fromkeys(pts))  # dedupe
    street_points[state] = pts

    print(f"  matched segments: {len(matches)} | points: {len(pts)}", flush=True)

# ---- 3) Output ----

out_csv = "dc_state_named_street_points_100m.csv"

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["state", "lon", "lat"])
    for state, pts in street_points.items():
        for lon, lat in pts:
            w.writerow([state, lon, lat])

print("\nDone.")
print("Missing states:", missing)
print(f"Wrote {out_csv}")
