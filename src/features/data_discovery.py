"""
Data discovery: turns folders of raw files into the dicts the rest of the
pipeline consumes -- replacing every hand-maintained, dataset-specific
config.py entry (the old `_RASTERS` dict, per-region `faults`/`margins`/
`worms`, and even the hardcoded region list itself) with an automatic
(recursive) scan of whatever files someone reusing this pipeline actually
has.

Two independent discovery passes:
  - `discover_all()` -- raster/vector/point FEATURE LAYERS, shared
    identically across every region (interactive: exclude stray files,
    choose GLCM rasters).
  - `discover_regions()` -- the REGIONS themselves: each outline file IS a
    region, matched to the shared H3 grid and to its own deposit CSV by
    filename prefix (non-interactive -- there's nothing to choose, a
    region either has the files it needs or discovery raises).

Deliberately kept OUT of src/config.py: config.py's own docstring promises
"nothing here has a side effect (no file I/O, no plotting)", and scanning
directories + prompting on the terminal are both side effects. Call these
explicitly (once, before `config.build_regions(...)`) from the notebook or
a driver script instead.

Four interactive prompts happen here, in this order -- one exclude-prompt
per folder (raster, then vector, then point), then one GLCM-selection
prompt over whatever rasters remain after exclusion:
  1-3. per folder: which discovered files to EXCLUDE entirely (e.g. a
       stray file you don't want treated as a feature layer at all).
  4. of the remaining rasters: which ones should ALSO get GLCM texture
     features (replaces the old hardcoded GLCM_RASTERS list in glcm.py).
All four are skippable non-interactively via `prompt_fn` (see below) for
scripted/CI use.
"""
import fnmatch
import os

# CSVs the pipeline itself WRITES, which discover_regions() must not mistake
# for a region's input deposit table. Both start with the region name, so a
# bare prefix match would find them: driving src/ from a directory that is
# both DEPOSITS_ROOT and the output directory (both default to '.') would
# then succeed on the first run and fail on the second, once
# 'Cordillera_features_semisup_combined_dedup.csv' sits next to
# 'Cordillera_REE_deposits_deduped_2.0km.csv'. Keep this in sync with what
# build_region()/run_all_regions() write (and with .gitignore).
PIPELINE_OUTPUT_CSV_PATTERNS = (
    '*_features_semisup*.csv',
    '*_top*_potential_targets*.csv',
)

RASTER_EXTENSIONS = ('.tif', '.flt')
VECTOR_EXTENSIONS = ('.geojson', '.shp')
POINT_EXTENSIONS  = ('.geojson', '.shp')


def _make_key(path):
    """Derive a feature-column-safe key from a filename: strip directory and
    extension, replace whitespace/hyphens with underscores. Casing is left
    as-is so the discovered key stays recognizable against the source file
    (e.g. 'Canada_BA.flt' -> 'Canada_BA')."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace(' ', '_').replace('-', '_')


def discover_layers(directory, extensions):
    """Recursively scan `directory` (any depth of subfolders, symlinked
    subfolders included -- real GIS datasets are routinely organised into
    nested folders by data family, e.g. 'converted data/Gravity/*.flt' vs
    'converted data/Magnetic/*.flt', and someone assembling RASTER_DIR/
    VECTOR_DIR/POINT_DIR out of an existing, differently-laid-out dataset
    may reasonably do it with symlinks rather than flattening/copying) for
    files ending in one of `extensions` (case-insensitive), return
    {key: path} sorted by key.

    Raises FileNotFoundError if `directory` doesn't exist, and ValueError if
    two files collide on the same derived key (e.g. 'Canada BA.tif' and
    'Canada_BA.tif', whether in the same subfolder or two different ones) --
    silently letting one overwrite the other would drop a layer without any
    warning."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f'{directory!r} is not a directory -- point RASTER_DIR / VECTOR_DIR / '
            f'POINT_DIR (see src/config.py) at real folders before calling discover_all().')
    layers = {}
    for root, _dirs, files in os.walk(directory, followlinks=True):
        for fname in sorted(files):
            if not fname.lower().endswith(extensions):
                continue
            key = _make_key(fname)
            path = os.path.join(root, fname)
            if key in layers:
                raise ValueError(
                    f'Two files under {directory!r} both map to the key {key!r} '
                    f'({layers[key]!r} and {path!r}) -- rename one so their derived '
                    f'feature-column names do not collide.')
            layers[key] = path
    return dict(sorted(layers.items()))


def choose_exclusions(layers, layer_type_label, prompt_fn=input):
    """Interactively ask which discovered files (of one type -- raster,
    vector, or point) to drop entirely, e.g. a stray file in the folder that
    isn't meant to be a feature layer. Returns a NEW {key: path} dict with
    the excluded keys removed; `layers` itself is untouched.

    `prompt_fn` is injectable (pass e.g. `lambda _: ''` for a non-interactive
    "keep everything" run, such as scripted/CI use)."""
    keys = sorted(layers)
    if not keys:
        return dict(layers)
    print(f'\nFound {len(keys)} {layer_type_label} layer(s):')
    for i, k in enumerate(keys, 1):
        print(f'  [{i}] {k}')
    print(f'Exclude any of these {layer_type_label} layers entirely?')
    print("  (comma-separated numbers to exclude -- blank/'none' to keep all)")
    raw = prompt_fn('> ').strip().lower()

    if raw in ('', 'none'):
        return dict(layers)

    excluded = set()
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit() or not (1 <= int(tok) <= len(keys)):
            raise ValueError(f'{tok!r} is not a valid choice (1-{len(keys)}).')
        excluded.add(keys[int(tok) - 1])

    kept = {k: p for k, p in layers.items() if k not in excluded}
    print(f'  Excluded: {sorted(excluded)}  ->  {len(kept)} {layer_type_label} layer(s) kept.')
    return kept


def choose_glcm_rasters(raster_layers, prompt_fn=input):
    """Interactively ask which of the (already-exclusion-filtered) raster
    layers should also get GLCM texture features (src/features/glcm.py).
    Returns a list of chosen raster keys, in the order they were discovered
    (not selection order)."""
    keys = sorted(raster_layers)
    if not keys:
        return []
    print(f'\n{len(keys)} raster layer(s) available for GLCM texture extraction:')
    for i, k in enumerate(keys, 1):
        print(f'  [{i}] {k}')
    print('Which of these should also get GLCM texture features?')
    print("  (comma-separated numbers, e.g. '1,3,4' -- 'all' for every layer -- blank/'none' for none)")
    raw = prompt_fn('> ').strip().lower()

    if raw in ('', 'none'):
        return []
    if raw == 'all':
        return keys

    chosen_idx = set()
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit() or not (1 <= int(tok) <= len(keys)):
            raise ValueError(f'{tok!r} is not a valid choice (1-{len(keys)}).')
        chosen_idx.add(int(tok) - 1)
    return [keys[i] for i in sorted(chosen_idx)]


def discover_all(raster_dir, vector_dir, point_dir, prompt_fn=input):
    """Full discovery pass: scan all three folders, let the user exclude
    stray files from each, then choose the GLCM raster subset.

    Returns (rasters, vectors, points, glcm_rasters):
      rasters      -- {key: path}, replaces the old hardcoded _RASTERS
      vectors       -- {key: path}, line-feature layers (replaces faults/margins)
      points         -- {key: path}, point-feature layers (replaces worms)
      glcm_rasters    -- list of raster keys chosen for GLCM texture extraction
    """
    rasters_all = discover_layers(raster_dir, RASTER_EXTENSIONS)
    vectors_all = discover_layers(vector_dir, VECTOR_EXTENSIONS)
    points_all  = discover_layers(point_dir, POINT_EXTENSIONS)
    print(f'Discovered {len(rasters_all)} raster, {len(vectors_all)} vector, '
          f'{len(points_all)} point layer(s) before exclusions.')

    rasters = choose_exclusions(rasters_all, 'raster', prompt_fn=prompt_fn)
    vectors = choose_exclusions(vectors_all, 'vector', prompt_fn=prompt_fn)
    points  = choose_exclusions(points_all, 'point', prompt_fn=prompt_fn)

    glcm_rasters = choose_glcm_rasters(rasters, prompt_fn=prompt_fn)
    print(f'\nFinal: {len(rasters)} raster, {len(vectors)} vector, {len(points)} point layer(s); '
          f'{len(glcm_rasters)} raster(s) selected for GLCM: {glcm_rasters}')
    return rasters, vectors, points, glcm_rasters


# ── Region discovery (outline / H3 grid / deposit CSV) ─────────────────────
OUTLINE_EXTENSIONS = ('.geojson', '.shp')
H3_GRID_EXTENSIONS = ('.geojson', '.shp', '.gpkg')


def _is_pipeline_output(filename):
    """True if `filename` is one of the CSVs this pipeline writes itself
    (see PIPELINE_OUTPUT_CSV_PATTERNS) rather than an input deposit table."""
    return any(fnmatch.fnmatch(filename, pat) for pat in PIPELINE_OUTPUT_CSV_PATTERNS)


def discover_regions(outline_dir, h3_grid_dir, deposits_root):
    """Discover regions themselves, not just their feature layers: unlike
    raster/vector/point layers (shared identically across every region),
    outline/h3_grid/dep each name or contain a SPECIFIC region, so
    discovering them also discovers the region set itself -- add a new
    outline file, get a new region, no config.py edit needed.

    - Each file in `outline_dir` IS one region -- its discovered key (see
      discover_layers/_make_key) becomes the region name.
    - `h3_grid_dir` must contain exactly ONE geodata file (the H3 terrane
      grid is a single national resource shared by every region, not
      indexed by region name) -- raises ValueError otherwise.
    - Each region's deposit CSV is found in `deposits_root` by REGION-NAME
      PREFIX (e.g. region 'Cordillera' matches
      'Cordillera_REE_deposits_deduped_2.0km.csv' -- the suffix is not
      hardcoded, so any deposit-CSV naming convention works as long as the
      filename starts with the region name) -- raises ValueError if a
      region has zero or more than one match. CSVs the pipeline itself
      writes (PIPELINE_OUTPUT_CSV_PATTERNS) are skipped, so a run whose
      output directory happens to be DEPOSITS_ROOT stays re-runnable.

    Returns {region_name: {'outline': path, 'h3_grid': path, 'dep': path}},
    the exact shape config.build_regions() expects as its first argument.
    """
    outlines = discover_layers(outline_dir, OUTLINE_EXTENSIONS)
    if not outlines:
        raise ValueError(f'no outline files found in {outline_dir!r} -- '
                          f'at least one region outline is required.')

    h3_grid_files = discover_layers(h3_grid_dir, H3_GRID_EXTENSIONS)
    if len(h3_grid_files) != 1:
        raise ValueError(
            f'expected exactly one H3 terrane-grid file in {h3_grid_dir!r}, '
            f'found {len(h3_grid_files)}: {sorted(h3_grid_files)!r}')
    h3_grid_path = next(iter(h3_grid_files.values()))

    all_csvs = [f for f in sorted(os.listdir(deposits_root)) if f.lower().endswith('.csv')]
    dep_files = [f for f in all_csvs if not _is_pipeline_output(f)]
    skipped = [f for f in all_csvs if _is_pipeline_output(f)]
    if skipped:
        print(f'  (ignoring {len(skipped)} pipeline-written CSV(s) in {deposits_root!r}: '
              f'{skipped!r})')

    regions = {}
    for region_name, outline_path in outlines.items():
        matches = [f for f in dep_files if f.startswith(region_name)]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one deposit CSV starting with {region_name!r} in "
                f'{deposits_root!r}, found {len(matches)}: {matches!r} '
                f'(pipeline-written CSVs matching {list(PIPELINE_OUTPUT_CSV_PATTERNS)!r} '
                f'are ignored; keep this folder to deposit tables only, or point '
                f'REE_DEPOSITS_ROOT / REE_TABLE_DIR at separate folders)')
        regions[region_name] = {
            'outline': outline_path,
            'h3_grid': h3_grid_path,
            'dep':     os.path.join(deposits_root, matches[0]),
        }
    print(f'Discovered {len(regions)} region(s): {sorted(regions)}')
    return regions
