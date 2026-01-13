# DC Streets

Find the optimal route to visit all state-named streets in Washington DC.

## What This Does

Washington DC has streets named after US states (Pennsylvania Avenue, Georgia Avenue, etc.). This project:

1. Finds all streets in DC named after each US state
2. Generates points every 100m along those streets
3. Solves a Generalized Traveling Salesman Problem (GTSP) to find the shortest route that visits one point on each state-named street

## Scripts

### `dc_streets.py`

Downloads DC's road network and finds streets matching state names. Outputs a CSV with points along each state-named street.

```bash
python dc_streets.py
# Output: dc_state_named_street_points_100m.csv
```

### `solve_gtsp.py`

Solves the GTSP: given clusters of points (one per state), find the shortest route visiting exactly one point from each cluster.

```bash
python solve_gtsp.py --input dc_state_points.csv --output best_route.tsv --open_path
```

Options:
- `--restarts N` - Number of random restarts (default: 30)
- `--seed N` - Random seed for reproducibility
- `--open_path` - Output an open path instead of a cycle (cuts the longest edge)

## Requirements

```bash
pip install osmnx geopandas
```

## How It Works

1. **Street matching**: Uses OpenStreetMap data via `osmnx` to find roads with state names
2. **Point generation**: Interpolates points every 100m along matched streets
3. **GTSP solver**: Uses 2-opt local search with random restarts, alternating between:
   - Optimizing the visit order (which state to visit next)
   - Optimizing which point to visit for each state (given neighbors)

## Example Output

```
visit_order  state          lon         lat
1            Maryland       -77.0365    38.9076
2            Pennsylvania   -77.0304    38.8951
3            Virginia       -77.0189    38.9012
...
```
