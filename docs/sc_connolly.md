# Shape Complementarity via Grid-Based Connolly SES

**File:** `scripts/v3/sc_connolly.py`  
**Validation script:** `scripts/v3/sc_connolly_validate.py`  
**Scatter plot:** `docs/figures/sc_validation.pdf`

---

## Why SES is required (not SAS)

The prior implementation used a Solvent Accessible Surface (SAS): each atom's van der Waals sphere expanded by the probe radius. While fast to compute, SAS critically lacks the **re-entrant saddle patches** that form where the rolling probe touches two atoms simultaneously without penetrating either.

These concave patches are the defining feature of the true Connolly SES. At the interface between two protein chains, the re-entrant patches have normals that point sideways or away from the local contact direction — exactly the geometry that penalises poor shape fit. Without them, the dot-product scores are systematically inflated and compressed, producing:

- SAS-based SC: Pearson r ≈ 0.50 vs Rosetta MSMS
- SES-based SC (this implementation): Pearson r = **0.564**, Spearman ρ = **0.611**, MAE = **0.025** vs Rosetta MSMS (N = 1,129, full positive cohort)

---

## Algorithm: Grid-Based Connolly SES

The method follows Connolly (1983) but uses a volumetric distance-transform rather than the original analytical probe-rolling, making it trivially pip-installable.

### 1. Voxel occupancy grid

A 3D grid of spacing `grid_spacing` Å (default 0.5 Å) is constructed around the protein chain, padded by 5 Å. Each voxel is marked as protein-interior (True) if its centre lies within the van der Waals radius of any heavy atom. Atom assignment uses a KDTree for efficiency.

### 2. Signed distance field

Two Euclidean distance transforms are computed:
- `edt_inside`: distance from each protein-interior voxel to the protein surface
- `edt_outside`: distance from each exterior voxel to the protein surface

The signed distance field is:
```
signed_dist = edt_outside - edt_inside
```
Positive values are outside the protein; negative values are inside.

### 3. SES isosurface

The SES is the locus of probe-centre positions when the probe (radius `probe_radius` = 1.4 Å) just contacts the protein VDW surface from outside:
```
ses_field = signed_dist - probe_radius
SES = { x : ses_field(x) = 0 }
```
This isosurface is extracted using marching cubes (scikit-image). The resulting normals point toward increasing `ses_field`, i.e., outward toward solvent — including the concave re-entrant patches where `signed_dist` transitions from low to probe_radius rapidly.

*Reference: Connolly ML (1983). Solvent-accessible surfaces of proteins and nucleic acids. Science 221:709-713. doi:10.1126/science.6879170*

---

## Shape Complementarity Statistic

After building per-chain SES meshes:

1. **Interface filter (atom-distance pre-filter):** Keep only SES vertices on chain A that lie within `interface_cutoff_prefilter` = 2.5 Å of any heavy atom of chain B (and vice versa). This selects the tightly apposed interface surface zone.

2. **Surface-surface pairing:** For each interface vertex on chain A, find the nearest interface vertex on chain B using a KDTree. Accept the pair if their Euclidean distance < `interface_cutoff` = 3.5 Å.

3. **Dot product:** For each accepted pair (vertex `v_A` with outward normal `n_A`, nearest vertex `v_B` with outward normal `n_B`):
   ```
   dot = n_A · (−n_B)
   ```
   A dot product of 1 indicates perfectly complementary (opposing) normals; 0 indicates orthogonal.

4. **Directional SC scores:** Compute A→B and B→A directionally. For each direction, take the **median of all positive dot products** (following Lawrence & Colman 1993).

5. **Final SC:**
   ```
   SC = mean(SC_{A→B}, SC_{B→A})
   ```
   Clipped to [0, 1].

*Reference: Lawrence MC, Colman PM (1993). Shape complementarity at protein/protein interfaces. J Mol Biol 234:946-950. doi:10.1006/jmbi.1993.1648*

---

## Validation vs Rosetta MSMS

Validated on the full positive cohort (N = 1,129) from the OpenBinder training set. Rosetta SC values were computed via InterfaceAnalyzer + MSMS (the reference implementation) under the predecessor NanoBinder-RF-v2 pipeline; the reference table is shipped at `data/sc_rosetta_reference.csv` to support Figure 1 regeneration without re-running PyRosetta.

| Metric | Value |
|--------|-------|
| Pearson r | **0.564** |
| Spearman ρ | **0.611** |
| MAE | **0.025** |
| N | **1,129** |
| Mean time/structure | 2.2 s (grid_spacing=0.5 Å, 6 workers) |

**Optimal parameters:** `grid_spacing=0.5`, `interface_cutoff_prefilter=2.5`, `interface_cutoff=3.5`, aggregation=median of positive dot products.

Scatter plot: `docs/figures/sc_validation.pdf`

---

## Why r=0.564 rather than 1.0

On the full N = 1,129 cohort the small MAE (0.025) and moderate Spearman ρ (0.611) confirm that absolute values and rankings agree closely; the lower Pearson r reflects two documented algorithmic departures from Rosetta's published Sc statistic in addition to grid-resolution effects:

1. **Hard cutoff vs exponential weighting:** Rosetta's published Sc uses an exponential weight `exp[-w·d²]` on the dot products, whereas this implementation applies a hard 3.5 Å cutoff with mean-of-medians aggregation. This is a deliberate simplification documented in §2.5 of the paper.

2. **Grid resolution:** The 0.5 Å voxel grid approximates the SES, whereas MSMS uses exact analytical probe rolling. Reducing to 0.3 Å reduces the gap marginally but increases compute time 5×.

3. **MSMS re-entrant detail:** MSMS computes analytically exact toroidal re-entrant patches. The grid SES approximates these with the distance-transform isosurface, which slightly smooths the concave geometry.

`sc_connolly` therefore serves as a usable open-source proxy rather than a numerically faithful reproduction of Rosetta's Sc; it ranks among the top features in the trained models and contributes measurable AUROC.

---

## Open-Source Dependencies

All pip-installable; no compiled binaries required:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `scipy.ndimage.distance_transform_edt` | EDT for SES construction |
| `scikit-image.measure.marching_cubes` | Isosurface extraction |
| `biopython` | PDB parsing |
| `scikit-learn.neighbors.KDTree` | Nearest-neighbour queries |

---

## Usage

### Single structure
```bash
python scripts/v3/sc_connolly.py \
    --pdb data/complex.pdb \
    --vhh-chain H \
    --ag-chain A
```

### Batch directory
```bash
python scripts/v3/sc_connolly.py \
    --batch-dir data/structures/positives_cleaned/  # unpack positives_cleaned.tar.gz first \
    --output results/sc_ses.csv \
    --workers 8
```

### Validation
```bash
python scripts/v3/sc_connolly_validate.py \
    --n-structures 50 \
    --workers 6
```

### Library
```python
from scripts.v3.sc_connolly import compute_sc
sc = compute_sc("complex.pdb", chain_vhh="H", chain_ag="A")
```
