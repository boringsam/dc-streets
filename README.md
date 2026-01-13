# DC Streets

Find the optimal walking route to visit all state-named streets in Washington DC.

## What This Does

Washington DC has streets named after US states (Pennsylvania Avenue, Georgia Avenue, etc.). This project:

1. Finds all streets in DC named after each US state
2. Generates points every 100m along those streets
3. Solves a Generalized Traveling Salesman Problem (GTSP) to find the shortest **walking route** that visits one point on each state-named street

**Key feature**: Distances are calculated using actual street networks, not straight-line ("as the crow flies"). The route follows real walkable paths.

## Scripts

### `dc_streets.py`

Downloads DC's road network and finds streets matching state names. Outputs a CSV with points along each state-named street.

```bash
python dc_streets.py
# Output: dc_state_named_street_points_100m.csv
```

### `solve_gtsp.py`

Solves the GTSP using real street network distances. Given clusters of points (one per state), finds the shortest walking route visiting exactly one point from each cluster.

```bash
python solve_gtsp.py --input dc_state_points.csv --output best_route.tsv --open_path
```

Options:
- `--restarts N` - Number of random restarts (default: 10)
- `--iters N` - Optimization iterations per restart (default: 10)
- `--seed N` - Random seed for reproducibility
- `--open_path` - Output an open path instead of a cycle (cuts the longest edge)

Note: First run downloads and caches the DC street network (~1 min). Subsequent runs are faster.

## Requirements

```bash
pip install osmnx geopandas networkx
```

## How It Works

1. **Street matching**: Uses OpenStreetMap data via `osmnx` to find roads with state names
2. **Point generation**: Interpolates points every 100m along matched streets
3. **Street network routing**: Downloads DC's walkable street network and calculates shortest path distances between points (not straight-line)
4. **GTSP solver**: Uses 2-opt local search with random restarts, alternating between:
   - Optimizing the visit order (which state to visit next)
   - Optimizing which point to visit for each state (given neighbors)
5. **Caching**: Street network and distance calculations are cached for speed

## Example Output

```
visit_order  state          lon         lat
1            Maryland       -77.0365    38.9076
2            Pennsylvania   -77.0304    38.8951
3            Virginia       -77.0189    38.9012
...
```
