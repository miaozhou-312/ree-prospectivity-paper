# Data

This folder holds the **small, public-derived inputs** that define the
problem (positive labels and study-area outlines). The large national
geophysical layers are third-party datasets and must be downloaded
separately; this guide lists every one of them, where to get it, and where
to put it.

```
data/
├── README.md                      <- this file
├── labels/                        <- positive samples (known REE occurrences)
│   ├── Cordillera_REE_deposits_deduped_2.0km.csv   <- the 3 tables the
│   ├── Superior_REE_deposits_deduped_2.0km.csv        main pipeline uses
│   ├── Grenville_REE_deposits_deduped_2.0km.csv
│   ├── ablation_no_dedup/          <- same pool, no 2 km deduplication
│   │                                  (input to experiments/05_ablations)
│   └── source/                     <- the two raw compilations the tables
│                                      above were built from
└── outlines/                       <- one study-area polygon per region
    ├── Cordillera.geojson
    ├── Superior.geojson
    └── Grenville.geojson
```

## 1. Setting up a run

The pipeline expects the downloaded layers in a `dataset/` folder that sits
**next to** this repository (not inside it), organised by geometry:

```
<parent>/
├── dataset/
│   ├── rasters/     <- 17 national grids (.tif / .flt + sidecars), any subfolders
│   ├── vectors/     <- 2 line layers (.geojson / .shp)
│   ├── points/      <- 4 point layers (.shp + sidecars)
│   └── h3_grid/     <- exactly one file set: Geology_H3Grid_Canada.shp (+ sidecars)
└── <this repo>/
    └── data/
        ├── labels/      <- deposit tables, read in place (DEPOSITS_ROOT)
        └── outlines/    <- study-area outlines, read in place (OUTLINE_DIR)
```

**With the project's prepared data archive** (`ree_dataset_public.zip` and
`ree_dataset_getech.zip`, kept privately and not distributed with this
repository): unzip both in the folder that contains this repository, which
creates `../dataset/` with every layer and the geochemistry in place. The
archive also carries copies of the labels and outlines, which are not needed:
the pipeline and the notebook read them from `data/` in this repository.

**Assembling it yourself:** download the layers in Section 2 into
`../dataset/{rasters,vectors,points,h3_grid}`. Every root can be pointed
elsewhere with the environment variables listed in `src/config.py`.

Budget roughly **5 GB** of disk for the layers (about 3.4 GB of rasters and a
~1.8 GB H3 terrane grid) plus ~1 GB of clipped-raster cache per full run. The
H3 grid is read whole when it is clipped to each region, which sets the
pipeline's ~6 GB peak memory (16 GB RAM is comfortable, 8 GB is not).

## 2. Source datasets and download links

All links were checked in October 2026. File names are the ones the stored
notebook outputs were produced with; keep them, because each file's name
(extension stripped) becomes its feature name.

### 2.1 Public layers

**USGS Tri-National Critical Minerals Mapping Initiative data release** --
McCafferty et al. (2023), <https://doi.org/10.5066/P970GDD5> (USGS
ScienceBase; U.S. Government work, public domain). One release supplies
most of the inputs:

| File used | Put in | Feature |
|---|---|---|
| `GeophysicsGravity_Up30km_USCanada.tif` | `rasters/` | Bouguer gravity upward-continued 30 km |
| `GeophysicsGravity_Up30km_HGM_USCanada.tif` | `rasters/` | Horizontal-gradient magnitude of the above |
| `USCanadaMagRTP_DeepSources.tif` | `rasters/` | Long-wavelength (deep-source) RTP magnetic anomaly |
| `USCanadaMagRTP_HGMDeepSources.tif` | `rasters/` | Horizontal-gradient magnitude of the deep RTP |
| `GeophysicsLAB_USCanada.tif` | `rasters/` | Depth to lithosphere-asthenosphere boundary |
| `USCanada_Moho.tif` | `rasters/` | Depth to Moho |
| `ShallowGravitySources_Worms_USCanada.shp` | `points/` | Shallow gravity "worms" (THD maxima) |
| `DeepGravitySources_Worms_USCanada.shp` | `points/` | Deep gravity worms |
| `ShallowMagSources_Worms_USCanada.shp` | `points/` | Shallow magnetic worms |
| `DeepMagSources_Worms_USCanada.shp` | `points/` | Deep magnetic worms |
| `Geology_H3Grid_Canada.shp` | `h3_grid/` | H3 (res 7) grid carrying geological era and lithology (from Wheeler et al., 1996) |
| `GeologyPassiveMargins_World.gdb`, exported as `AncientPassiveMargins.geojson` | `vectors/` | Ancient passive margins (Bradley, 2008 compilation) |
| `Faults_Canada.geojson` | `vectors/` | Faults, Canada subset (source to confirm: see [KNOWN_ISSUES #3](../KNOWN_ISSUES.md#3-provenance-of-the-fault-layer-is-unverified)) |

**Positive-label sources** (already in [`labels/source/`](labels/source/);
only needed if you want to rebuild the label tables):

| File | Source | Notes |
|---|---|---|
| `REE_Canada_Carbonatite.csv`, `REE_Canada_Alkaline.csv` | Orris et al. (2018), *Global rare earth element occurrence database*, USGS, <https://doi.org/10.5066/F7DR2TN4> (public domain) | Canadian records, split by deposit type (carbonatite / alkaline) |
| `Parsa_2024_REE_occurrences.csv` | Parsa et al. (2024), <https://doi.org/10.1007/s11053-024-10369-7> | Occurrence table from the paper, as received (Latin-1 encoded) |
| `Parsa_2024_REE_occurrences_fixed.csv` | as above | Same table with the encoding repaired; its `Northing`/`Easting` headers are swapped relative to the values, which `experiments/02_label_analysis/positive_label_dedup.ipynb` corrects on read |

**Study-area outlines** ([`outlines/`](outlines/)): the Cordilleran Orogen,
Superior Province and Grenville Province polygons (`GEOLPROV` field) from
Wheeler et al. (1996), *Geological map of Canada*, Map 1860A,
<https://doi.org/10.4095/208175>.

**Whole-rock geochemistry** (used only by the abandoned geochemistry
experiments in [`../experiments/01_geochemistry/`](../experiments/01_geochemistry/)):
Gard, Hasterok & Halpin (2019), *Global whole-rock geochemical database
compilation*, Zenodo, <https://doi.org/10.5281/zenodo.3359791> (CC BY 4.0).
Download `complete.csv`; see Section 4 for how it was reduced.

### 2.2 Proprietary layers (Getech)

Eleven of the seventeen rasters were supplied for this research by **Getech
Group plc** as processed national grids. The supplied files are **not
redistributed** here and have no public download; researchers wanting the
exact files should contact Getech. Four of them are Getech's gridding of
public Natural Resources Canada (NRCan) maps; the other seven are Getech
derivative products.

| File used (`rasters/`) | Feature | Origin |
|---|---|---|
| `Canada_BA.flt` | Bouguer anomaly, 2 km | NRCan, Jobin et al. (2017), <https://doi.org/10.4095/299561> |
| `Canada_BA_1VD.flt` | 1st vertical derivative of BA | NRCan, Jobin et al. (2017) |
| `Canada_BA_THD.flt` | Total horizontal derivative of BA | NRCan, Jobin et al. (2017) |
| `TMI.flt` | Total magnetic intensity, 0.01 deg | NRCan, Miles & Oneschuk (2016), <https://doi.org/10.4095/297337> |
| `Canada_BA_THD_Tilt.flt` | Tilt derivative of BA_THD | Getech derivative |
| `Canada_BA_HP40km.flt` | 40 km high-pass (4th-order Butterworth) of BA | Getech derivative |
| `Canada_BA_HP100km.flt` | 100 km high-pass (4th-order Butterworth) of BA | Getech derivative |
| `DRTP.flt` | Differential reduced-to-pole of TMI | Getech derivative |
| `DRTP_1VD.flt` | 1st vertical derivative of DRTP | Getech derivative |
| `DRTP_THD.flt` | Total horizontal derivative of DRTP | Getech derivative |
| `DRTP_THD_Tilt.flt` | Tilt derivative of DRTP_THD | Getech derivative |

Every derivative above is a standard potential-field filter of a public
NRCan grid, so equivalent layers can be computed from the public parents.
They will not be numerically identical to the Getech versions, so results
will differ somewhat. Whatever their origin, the four GLCM texture rasters
pinned in the notebook (`Canada_BA`, `Canada_BA_HP40km`, `DRTP`, `TMI`) must
exist under those names for the notebook to run unchanged.

## 3. How the pipeline finds data

Nothing is hand-listed in code: both `src/` and the notebook discover inputs
by scanning folders (`src/features/data_discovery.py`; mirrored inline in the
notebook so it runs without importing `src`).

**Feature layers** -- every file under a root becomes one feature, keyed by
its filename (extension stripped, spaces/hyphens to underscores). Scans
recurse and follow symlinks, so provider subfolders can stay as they are.

| Root | Env var (`src/` only) | Default | Becomes |
|---|---|---|---|
| `RASTER_DIR` | `REE_RASTER_DIR` | `{DATA_ROOT}/rasters` | one raster feature per grid (+ GLCM textures for the chosen subset) |
| `VECTOR_DIR` | `REE_VECTOR_DIR` | `{DATA_ROOT}/vectors` | `{name}_proximity` per line layer |
| `POINT_DIR` | `REE_POINT_DIR` | `{DATA_ROOT}/points` | `worm_{name}_proximity` per point layer |

**Regions** -- discovering these also defines the region set:

| Root | Env var | Default | Rule |
|---|---|---|---|
| `OUTLINE_DIR` | `REE_OUTLINE_DIR` | `{DATA_ROOT}/outlines` | Each file **is** a region; its filename is the region name. |
| `H3_GRID_DIR` | `REE_H3_GRID_DIR` | `{DATA_ROOT}/h3_grid` | Exactly one geodata file, shared by all regions. |
| `DEPOSITS_ROOT` | `REE_DEPOSITS_ROOT` | `.` for `src/`, `..` in the notebook | Each region matches the single CSV whose name starts with the region name. Not recursive. Raises on zero or multiple matches, so keep only the three label tables there. |

`DATA_ROOT` defaults to `../dataset` (env var `REE_DATA_ROOT`). To drive
`src/` against data stored elsewhere:

```bash
export REE_DATA_ROOT=/path/to/dataset
export REE_DEPOSITS_ROOT=/path/to/this/repo/data/labels
```

```python
from src import config
from src.features.data_discovery import discover_all, discover_regions

rasters, vectors, points, glcm_rasters = discover_all(
    config.RASTER_DIR, config.VECTOR_DIR, config.POINT_DIR)
region_manual = discover_regions(
    config.OUTLINE_DIR, config.H3_GRID_DIR, config.DEPOSITS_ROOT)
config.REGIONS = config.build_regions(region_manual, rasters, vectors, points)
```

`discover_all` prompts on the terminal to exclude stray files and to choose
the GLCM rasters; the notebook pins that choice instead, so it runs
headless.

Shapefile sidecars (`.dbf`/`.shx`/`.prj`/`.cpg`/`.sbn`/`.sbx`) and raster
sidecars (`.hdr`/`.prj`/`.tfw`/`.aux.xml`/`.ovr`) must sit next to their
primary file.

## 4. How the derived inputs were built

**Positive labels.** [`../experiments/02_label_analysis/positive_label_dedup.ipynb`](../experiments/02_label_analysis/positive_label_dedup.ipynb)
pools the carbonatite records of Orris et al. (2018) with the Parsa et al.
(2024) table, keeps those inside each outline, and collapses any points
within 2 km of each other (single-linkage, transitive) to one
representative, so the same deposit listed twice -- within or across the two
sources -- counts once. Each row keeps a `dep_type` confidence class
(`Deposit/Prospect`, `Showing`, `Occurrence`).

| Region | Pooled | After 2 km dedup | Inside the modelling domain |
|---|---|---|---|
| Cordillera | 51 | 28 | 27 (`Lempriere` falls outside the A∧B domain) |
| Superior | 21 | 15 | 15 |
| Grenville | 13 | 12 | 12 |

`ablation_no_dedup/` holds the same pooled tables without deduplication,
used by the leakage ablation.

**National scale (`national/`).** Built by `paper_analyses/40_national_labels.py`.
`Canada.geojson` is the union of the onshore provinces of GSC Map 1860A
(`GEOPROV.SHP`: every named province, orogen and platform except the
continental shelves and oceanic crust; 12.3 million km²). `provinces.geojson`
holds the provinces dissolved by name, with the three study provinces as their
regional polygons and province windows smaller than 2,000 km² (such as the
Churchill-basement windows in the Cordillera) merged into the province around
them. `Canada_REE_deposits_deduped_2.0km.csv` is the national label table: the
three regional tables as they are (15 + 28 + 12) plus 13 occurrences elsewhere,
deduplicated with the same 2 km rule; records within 2 km of a regional
occurrence (Switch Creek, Terres Rompues) count as that occurrence.

**Geochemistry (experiments only).** `experiments/00_data_preparation/extract_canada_data.ipynb`
streams `complete.csv` in chunks and keeps `country_id == 32` (Canada,
78,077 samples). Per-region files (`Geochemistry_<Region>_Clip.csv`) and
per-region occurrence files (`CarbonatitehostREE_<Region>_Clip.csv`) were
then cut to each outline in GIS, outside the notebooks.

## 5. Running the archived experiments

The notebooks in [`../experiments/`](../experiments/) are kept exactly as
they were run, with their outputs. They predate the folder-discovery
refactor and read the providers' original folder names from a flat working
directory:

```
<parent>/
├── dataset/
│   ├── converted getech data/Gravity_converted/Canada_BA*.flt
│   ├── converted getech data/Magnetic_converted/{TMI,DRTP*}.flt
│   ├── new USGS data/         <- the USGS files of Section 2.1, unrenamed
│   ├── 3_areas/{Cordilleran,Superior,Grenville}.geojson
│   ├── Faults_Canada/Faults_Canada.geojson
│   ├── Passive Margins/AncientPassiveMargins.geojson
│   ├── REE_Canada/REE_Canada_{Carbonatite,Alkaline}.csv
│   └── Parsa_2024_REE_occurrences{,_fixed}.csv
├── geochem/                   <- Section 4 outputs
└── code/                      <- notebook + label CSVs here, run from here
```

The region `Cordilleran` was later renamed `Cordillera`, and these notebooks
still use the old name. When staging files for them, copy
`data/outlines/Cordillera.geojson` to `dataset/3_areas/Cordilleran.geojson`
and `data/labels/Cordillera_REE_deposits_deduped_2.0km.csv` to
`code/Cordilleran_REE_deposits_deduped_2.0km.csv` (the `ablation_no_dedup/`
tables already carry the old name). See
[`../experiments/README.md`](../experiments/README.md) for what each
notebook needs.

## References

- Bradley, D. C. (2008). Passive margins through earth history. *Earth-Science Reviews*, 91(1-4), 1-26. <https://doi.org/10.1016/j.earscirev.2008.08.001>
- Gard, M., Hasterok, D., & Halpin, J. (2019). *Global whole-rock geochemical database compilation* [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.3359791>
- Jobin, D. M., Véronneau, M., & Miles, W. (2017). *Gravity anomaly map, Canada* (No. 8081). <https://doi.org/10.4095/299561>
- McCafferty, A. E., San Juan, C. A., Lawley, C., Graham, G. E., Gadd, M. G., Huston, D. L., Kelley, K. D., Paradis, S., Peter, J. M., & Czarnota, K. (2023). *National-scale geophysical, geologic, and mineral resource data and grids for the United States, Canada, and Australia: Data in support of the Tri-National Critical Minerals Mapping Initiative* (ver. 1.1, March 2025) [Data set]. U.S. Geological Survey. <https://doi.org/10.5066/P970GDD5>
- Miles, W., & Oneschuk, D. (2016). *Magnetic anomaly map, Canada* (No. 7799). <https://doi.org/10.4095/297337>
- Orris, G. J., Seo, Y., Briggs, D. A., Dunlap, P., & Cocker, M. D. (2018). *Global rare earth element occurrence database* [Data set]. U.S. Geological Survey. <https://doi.org/10.5066/F7DR2TN4>
- Parsa, M., Lawley, C. J. M., Cumani, R., Schetselaar, E., Harris, J., Lentz, D. R., Zhang, S. E., & Bourdeau, J. E. (2024). Predictive modeling of Canadian carbonatite-hosted REE +/- Nb deposits. *Natural Resources Research*, 33(5), 1941-1965. <https://doi.org/10.1007/s11053-024-10369-7>
- Wheeler, J. O., Hoffman, P. F., Card, K. D., Davidson, A., Sanford, B. V., Okulitch, A. V., & Roest, W. R. (1996). *Geological map of Canada* (No. 1860A). <https://doi.org/10.4095/208175>
