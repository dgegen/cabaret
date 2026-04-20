# Local database

`GaiaQuery` supports querying offline SQLite catalogs in addition to online
TAP services. You can select sources either by a circular region
(`center` + `radius`) or by rectangular `bounds` `(ra_min, ra_max, dec_min, dec_max)`.

## Using a local SQLite catalog

Create a `GaiaSQLiteSource` to point Cabaret at a local SQLite file and
optional table name. The table is expected to expose Gaia-like columns named
similarly to the TAP service (for example: `ra`, `dec`, `phot_g_mean_mag`,
`h_m`, ...). Example:

```python
import cabaret
from astropy.coordinates import SkyCoord

sqlite_source = cabaret.GaiaSQLiteSource(
    database="/data/catalogs/gaia_subset.sqlite",
    table="gaia_sources",
)

center = SkyCoord(ra=323.36, dec=-0.82, unit="deg")
table = cabaret.GaiaQuery.query(
    center=center,
    radius=0.1,  # degrees
    filter_bands=[cabaret.Filters.G, cabaret.Filters.H],
    tap_source=sqlite_source,  # or "sqlite:///path/to/file.sqlite"
)
```

You may pass the `tap_source` as either a `GaiaSQLiteSource` instance or a
string. Accepted string forms include a full SQLite URI (`sqlite:///...`) or a
plain path ending in `.db`, `.sqlite` or `.sqlite3`.

## Sharded (declination-ring) catalogs

Some offline catalogs are split into declination "rings" (shards) named like
`-25_-24`, `-24_-23`, etc. If the configured `table` name is not present in
the database, Cabaret will auto-detect these ring tables and only query the
shards that overlap your requested declination range. This keeps queries fast
when only a small declination slice is required.

Example using a sharded SQLite file (RA wraparound shown below):

```python
import cabaret

table = cabaret.GaiaQuery.query(
    bounds=(359.5, 0.5, -2.0, 2.0),  # ra_min, ra_max wrap across 0 deg
    filter_bands="G",
    tap_source="sqlite:///data/catalogs/gaia_tmass_sharded.sqlite",
)
```

## RA wraparound (bounds that cross RA=0)

RA values are interpreted modulo 360. If `ra_min <= ra_max` the interval is
treated as contiguous; if `ra_min > ra_max` the interval is interpreted as
wrapping across RA=0 (for example `(350.0, 10.0, -5.0, 5.0)`). Both the
TAP and SQLite query code handle wraparound automatically, selecting rows
whose RA falls into the wrapped interval.

## Notes and tips

- Use `filter_bands` to request one or more bands (e.g. `Filters.G` or
  `['G','H']`). Including a 2MASS band (J/H/KS) will add the required
  cross-match columns when available.
- If your SQLite table contains many rows, prefer a rectangular pre-filter
  (`bounds`) or a small `limit` to keep memory and runtime reasonable.
- Cabaret will only query the shard tables that intersect the requested
  declination range; you don't need to list shard names manually.

### Changing the default TAP source

You can change the module-wide default TAP source so that calls which omit
`tap_source` use your local SQLite file by default. For example:

```python
import cabaret

cabaret.GaiaQuery.DEFAULT_TAP_SOURCE = cabaret.GaiaSQLiteSource(
    database="/data/catalogs/gaia_subset.sqlite",
    table="table_name",
)

# Subsequent calls that omit `tap_source` will use the SQLite file.
image = cabaret.Observatory().generate_image(
    ra=12.33230,  # right ascension in degrees
    dec=30.4343,  # declination in degrees
    exp_time=10,  # exposure time in seconds
)
```

## See also

- The `GaiaQuery` implementation: [src/cabaret/queries.py](src/cabaret/queries.py)
- Prebuilt sharded Gaia+2MASS SQLite catalogs: [ppp-one/gaia-tmass-sqlite](https://github.com/ppp-one/gaia-tmass-sqlite)
