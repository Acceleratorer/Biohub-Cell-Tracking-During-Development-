# Biohub — Cell Tracking During Development

Knowledge base and top-1 competition plan for the Kaggle competition:

**Biohub — Cell Tracking During Development**  
Detect and track zebrafish cells through 3D space and time.

This document was prepared on **September 9, 2026** after auditing every notebook in this workspace and reading the official Kaggle competition pages and the organizers' public reference implementation.

The short version:

- The strongest reusable idea already present here is a **3D Temporal U-Net + node transformer + ILP graph solver**.
- The notebooks named `0.966` are promising, but their filenames are not proof of a reproducible local score: local evaluation cells were disabled or empty.
- The current public target is approximately **0.970**, so the immediate objective is to close a roughly **0.004** score gap with metric-faithful validation, calibrated node counts, stronger cross-embryo training, and cleaner graph post-processing.
- The negative-time hub/fork augmentation is a metric exploit and should not be the main solution. It is brittle, difficult to defend, and not suitable for a reproducible winning submission.

## 1. Competition snapshot

The following was verified against Kaggle on September 9, 2026. Re-check the Kaggle page immediately before submission because dates and competition settings can change.

| Item | Current information |
|---|---|
| Competition | Biohub — Cell Tracking During Development |
| Sponsor | Biohub SF |
| Task | Detect cells, link them across time, and identify divisions in 3D zebrafish microscopy |
| Final submission deadline | September 29, 2026, 23:59 UTC |
| Entry and team-merger deadline | September 22, 2026, 23:59 UTC |
| Maximum team size | 5 |
| Submission limit | 5 per day |
| Final submissions | Up to 2 |
| Submission mode | Notebook-only |
| Internet | Disabled during the notebook run |
| Runtime | Up to 12 hours for CPU or GPU notebooks |
| Required file | `submission.csv` |
| Prize pool | $60,000 |
| Public leaderboard snapshot | Approximately 0.970 for rank 1 and 0.966 for rank 2 |

Official pages and references:

- [Kaggle competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- [Official Kaggle rules and timeline](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)
- [Official organizers' reference repository](https://github.com/royerlab/kaggle-cell-tracking-competition)
- [Official metric definition](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)

## 2. What the data contains

### Images

Each dataset is a Zarr v3 directory containing one array at path `0`:

```text
(T, Z, Y, X)
```

The public description says the typical shape is approximately:

```text
(100, 64, 256, 256), uint16
```

The image chunks are organized one timepoint at a time:

```text
<sample>.zarr/
└── 0/
    ├── zarr.json
    └── c/
        └── <t>/0/0/0
```

### Physical voxel scale

The data is strongly anisotropic:

```text
z = 1.625    µm/voxel
y = 0.40625  µm/voxel
x = 0.40625  µm/voxel
```

The Z axis is four times coarser than either lateral axis. Every distance, matching rule, motion prior, and NMS kernel must account for this scale.

### Ground truth

Training samples contain a paired GEFF graph:

```text
<sample>.zarr
<sample>.geff
```

The GEFF graph contains:

- nodes with `(t, z, y, x)` coordinates;
- directed temporal edges;
- divisions encoded as one parent node with two outgoing edges;
- metadata containing `estimated_number_of_nodes`, a coarse estimate of the total number of cells, including cells not annotated.

The labels are **sparse**. A missing ground-truth node does not mean that the corresponding biological cell is absent.

### Train/test split

The competition states that the train and test sets are embryo-disjoint. Dataset names follow the general form:

```text
<embryo_id>_<field_of_view>
```

Validation must therefore be grouped by embryo ID. A random split of individual videos can leak biological appearance and motion statistics between train and validation.

## 3. Official metric

The score is:

```text
score = adjusted_edge_jaccard + 0.1 * division_jaccard
```

The edge term dominates. Divisions matter, but they are only 10% of the combined score.

### 3.1 Node matching

Predicted nodes and ground-truth nodes are matched independently at each timepoint using optimal bipartite assignment. A pair is eligible only when the physical centroid distance is at most:

```text
7.0 µm
```

The physical distance is:

```text
d = sqrt(
    ((z_pred - z_gt) * 1.625)^2
  + ((y_pred - y_gt) * 0.40625)^2
  + ((x_pred - x_gt) * 0.40625)^2
)
```

This is why pixel-space Euclidean distance is incorrect.

### 3.2 Edge Jaccard

A predicted edge is a true positive when:

1. its source matches a ground-truth node;
2. its target matches a ground-truth node;
3. the corresponding ground-truth edge exists.

For evaluated edges:

```text
edge_jaccard = TP / (TP + FP + FN)
```

Important details from the official evaluator:

- only edges from `t` to `t + 1` are evaluated;
- backward, same-frame, and longer skip edges are dropped before edge scoring;
- an edge is penalized when its matched source or target conflicts with a valid ground-truth adjacency;
- many edges whose endpoints are both outside the sparse annotation are ignored;
- predicted nodes that do not match an annotated node are not automatically edge false positives;
- duplicate edges and merged edges are explicitly guarded against;
- a predicted node can have at most two evaluated outgoing edges.

### 3.3 Adjusted edge Jaccard

The raw edge score is adjusted using the estimated total node count:

```text
total_node_ratio = (N_pred - N_total_estimate) / N_total_estimate

adjusted_edge_jaccard =
    max(0, edge_jaccard * (1 - 0.1 * total_node_ratio))
```

Consequences:

- over-predicting nodes lowers the score;
- under-predicting nodes can make the multiplier greater than one;
- the score can therefore exceed 1.0 for some predictions;
- node-count calibration is part of the competition, not an optional cleanup step.

Across samples, adjusted edge Jaccards are weighted by:

```text
TP + FP + FN
```

The larger samples matter more.

### 3.4 Division Jaccard

A predicted division is a node with at least two outgoing edges. The official division evaluator examines a local directed window:

```text
grandparent → parent/divider → daughter 1 → grandchild 1
                         └──→ daughter 2 → grandchild 2
```

A division can be recovered when the predicted graph provides:

- a matched parent-side anchor;
- two distinct directed daughter branches;
- valid local topology;
- no invalid branch merge;
- compatible ground-truth component evidence.

The evaluator allows a small local timing shift around the visible split, but it does not accept a merely weakly connected component. Directed topology matters.

### 3.5 Metric strategy

The metric implies this priority order:

1. get the total number of detections close to the biological count;
2. maximize correct consecutive-frame links;
3. remove re-parenting, merging, and identity-switch edges;
4. recover real divisions with low false-positive rate;
5. use gap repair only when local evidence supports it.

Adding arbitrary nodes or edges is not a robust strategy.

## 4. Workspace audit

The workspace contains five original notebooks plus the new candidate notebook:

| Notebook | Role | Main configuration | Recorded result |
|---|---|---|---|
| `demo_test.ipynb` | CPU heuristic baseline | Downsample 4, percentile-90 connected components, Hungarian linking at 15 µm | 18,959 rows written |
| `biohub-v6-ultra-best-0.964.ipynb` | U-Net/transformer pipeline | Point threshold 0.97, XY TTA, edge limit 12 µm, gap close 12 µm, `MAX_COMPONENTS=5000`, `FORKS=15` | 0.964 in filename; runtime about 10.2 minutes on two T4s |
| `improved-metric-hack-last-call_0.966.ipynb` | U-Net/transformer + hybrid repair | Point threshold 0.95, edge limit 10 µm, gap close 10 µm, `MAX_COMPONENTS=3000`, `FORKS=20` | 0.966 in filename; runtime about 6.5 minutes on two T4s |
| `biohub-solution-0.966.ipynb` | Expanded hybrid pipeline | Same core model, explicit post-processing and augmentation | 0.966 in filename; runtime about 6.0 minutes on two T4s |
| `biohub-competition-solution-0.966-2.ipynb` | Later compact variant | Point threshold 0.97, external `last_call` post-process, `MAX_COMPONENTS=1400`, `FORKS=5` | 0.966 in filename; runtime about 6.9 minutes on two T4s |

The notebook outputs show successful inference and submission generation, but they do not contain a reliable local final score for the `0.966` notebooks. Their `MODE == "local"` evaluation cells were not run in the saved output, or were absent. Treat the score in a filename as a reported experiment label, not as a reproducibility guarantee.

The recorded runs processed only four visible test datasets. Kaggle states that the notebook rerun swaps in a hidden test set approximately the size of the training data, so the short recorded runtimes are not a proof that the final notebook fits the 12-hour budget.

Recorded neural output sizes are also useful baselines:

- `biohub-v6-ultra-best-0.964.ipynb`: 256,543 raw rows, 259,564 rows after hybrid cleaning, and 265,328 rows after the final synthetic augmentation;
- `improved-metric-hack-last-call_0.966.ipynb` and `biohub-solution-0.966.ipynb`: 262,401 raw rows, 265,107 cleaned rows, and 271,644 rows after augmentation;
- `biohub-competition-solution-0.966-2.ipynb`: 251,746 raw rows, 254,497 cleaned rows, and 258,540 rows after augmentation.

The later compact variant reports 129,320 final neural node rows and 122,426 final neural edge rows before its external post-processing. These are submission-construction counts, not quality scores.

### New candidate notebook

`biohub-solution-0.970-candidate.ipynb` is a clean copy of `biohub-solution-0.966.ipynb` with one targeted inference change:

- weak candidate edges receive a soft physical-distance prior using the anisotropic `(z, y, x)` voxel scale;
- high-confidence transformer edges remain unchanged;
- the distance prior is applied before top-k parent selection and ILP;
- export now rejects duplicate edges, non-consecutive temporal links, multiple parents, and sources with more than two children.

Default candidate settings:

```text
EDGE_DISTANCE_PRIOR_SIGMA_UM = 5.0
EDGE_DISTANCE_PRIOR_WEIGHT   = 0.35
```

The original notebook is intentionally preserved. Validate this candidate on embryo-disjoint training folds before spending a Kaggle submission.

All four neural notebooks share nearly the same model and inference core. The main experimental changes are:

- point detection threshold;
- candidate edge threshold;
- top-k parent candidates;
- maximum physical edge distance;
- ILP solver weights;
- one-frame gap closing;
- short-component filtering;
- number of synthetic components/forks appended at the end.

## 5. Existing neural pipeline

The common pipeline is:

```text
Zarr volume
    ↓
quantile normalization
    ↓
spatial subsampling [1, 4, 4]
    ↓
TemporalUNet3D
    ├── detection logits
    └── per-voxel feature maps
    ↓
XY flip/rotation TTA
    ↓
3D max-pool local-max detection
    ↓
feature sampling at detected points
    ↓
sinusoidal position features
    ↓
SimpleNodeTransformer edge logits
    ↓
candidate edge filtering
    ↓
tracksdata ILP solver
    ↓
GEFF graph
    ↓
CSV
    ↓
optional gap repair / component filtering / augmentation
```

### 5.1 Resolution choice

The notebooks use:

```python
SUBSAMPLE = [1, 4, 4]
VOLUME_SHAPE = [64, 64, 64]
```

This produces an approximately isotropic model grid:

```text
z: 1 × 1.625    = 1.625 µm/grid voxel
y: 4 × 0.40625  = 1.625 µm/grid voxel
x: 4 × 0.40625  = 1.625 µm/grid voxel
```

This is computationally attractive and physically sensible, but it discards lateral detail. A top-1 effort should compare this against a multi-resolution detector or XY stride 2/full-resolution tiled detector.

### 5.2 Normalization

The notebooks read the Zarr image statistics and apply:

```python
small = (small - q_low) / (q_high - q_low + 1e-6)
small = np.clip(small, 0.0, None)
```

The usual values are the stored 0.1% and 99.9% intensity quantiles. The reference repository uses the same type of normalization.

Do not replace this with one global normalization without measuring the effect. These videos can vary by embryo, depth, and time due to attenuation and photobleaching.

### 5.3 Model

`MyUnet` contains:

```text
TemporalUNet3D:
    input channels = 1
    output feature channels = config["unet_out_channels"]
    layers = config["unet_layers"]

Detection head:
    Conv3d(unet_out_channels, 1, kernel_size=1)

SimpleNodeTransformer:
    feature dimension = U-Net channels + 32 position features
    hidden dimension = 128
    heads = 4
    blocks = 4
    dropout = 0
```

The model receives two consecutive frames. The U-Net produces a feature map and a point/detection logit for each frame. Features at candidate centroids are concatenated with position embeddings and passed to the transformer.

### 5.4 Position embeddings

`embed_position` encodes four normalized coordinates:

```text
(t, z, y, x)
```

Each axis receives eight sinusoidal features, producing 32 features total:

```text
sin(value * 2^k * π), cos(value * 2^k * π)
```

The temporal input is normalized using `TIME_LENGTH = 2` because the current inference window contains two frames.

### 5.5 Detection

`prob_to_zyx`:

1. applies a 3D max-pool;
2. keeps voxels that equal the local pooled maximum;
3. keeps peaks above `POINT_THRESHOLD`.

The local-max kernel is derived from a physical suppression distance, usually about 3 µm. In the isotropic downsampled grid this is approximately a `(3, 3, 3)` kernel.

The current detector has no sub-voxel regression head. Final coordinates are converted back to original voxel space and rounded to integers.

### 5.6 Test-time augmentation

The active TTA variant is `do_tta_8fliprot`:

- no transform;
- X flip;
- Y flip;
- XY flip;
- 90° rotation;
- 270° rotation;
- transpose;
- rotation plus transpose.

Only Y/X transformations are used. This is appropriate for the strong Z anisotropy; Z flips are out-of-distribution compared with XY symmetry.

The newer notebooks average both:

- detection logits;
- U-Net feature maps used by the edge transformer.

This is better than averaging detection logits while using only one transformed feature map.

### 5.7 Edge candidate generation

The transformer output is converted with:

```python
edge_prob = torch.softmax(edge_logit, dim=0)
```

The `dim=0` choice makes each target column compete among possible parents. It permits a parent to link to two different targets for division, while discouraging multiple parents for one target.

Candidate edges are kept when either:

- probability is above `EDGE_STRONG_THRESHOLD`, usually `0.50`; or
- the edge is among the top `EDGE_TOPK_PARENTS` candidates for a target and is above `EDGE_MIN_THRESHOLD`.

Typical notebook settings:

```text
EDGE_STRONG_THRESHOLD = 0.50
EDGE_MIN_THRESHOLD    = 0.20–0.25
EDGE_TOPK_PARENTS     = 3
EDGE_MAX_DISTANCE_UM  = 10–12
```

The competition metric matches within 7 µm, but the inference radius can be larger to retain candidates for later global optimization. The optimal radius must be measured locally; a larger candidate radius is not automatically a better prediction radius.

### 5.8 ILP graph solving

The notebooks use `tracksdata`'s ILP solver after candidate generation. Common weights are:

```text
edge weight          = -1.0 * edge probability
appearance weight    = 0.0
disappearance weight = 1.4
division weight      = 1.0
```

The solver is intended to enforce graph consistency:

- at most one parent per target;
- at most two children per source;
- flow-like appearance/disappearance behavior;
- optional divisions.

The notebooks show Gurobi warnings and fallback to SCIP because a Gurobi license is unavailable. The actual runtime path is therefore SCIP.

### 5.9 Offline dependencies and attached assets

The neural notebooks depend on an attached Kaggle support dataset rather than files in this workspace:

```text
/kaggle/input/datasets/pilkwang/
    biohub-tracking-support-pack-50ep-v1/
        wheels/
        repo/src/
        weights/unet_transformer/split_0/
```

The compact `biohub-competition-solution-0.966-2.ipynb` also imports:

```text
/kaggle/input/datasets/yoikoarmor/
    biohub-last-call-postprocess-v1/
```

Pinned offline packages include `tracksdata`, Zarr 3.2.1, GEFF, Polars, PySCIPOpt, `ilpy`, `rustworkx`, `imagecodecs`, and their supporting codecs. A final reproducible project must either keep these Kaggle datasets attached or vendor an equivalent documented package/weights bundle.

## 6. Notebook function inventory

### Common model and geometry functions

| Function | Purpose |
|---|---|
| `MyUnet.__init__` | Builds Temporal U-Net, detection head, and node transformer |
| `MyUnet.forward_unet` | Produces feature maps and point logits for two frames |
| `MyUnet.forward_transformer` | Produces pairwise edge logits from sampled features and positions |
| `embed_position` | Builds sinusoidal `(t,z,y,x)` embeddings |
| `pool_kernel_from_um` | Converts a physical distance to an odd voxel kernel |
| `prob_to_zyx` | Extracts 3D local maxima from point probabilities |
| `select_feature` | Samples U-Net features at candidate coordinates |
| `build_graph` | Creates an in-memory tracksdata graph with node and edge attributes |
| `load_model_weight` | Loads a checkpoint with `strict=False` |
| `load_volume` | Opens Zarr, reads quantile metadata, subsamples, and normalizes |

### TTA functions

| Function | Purpose |
|---|---|
| `do_tta_4flip` / `undo_tta_4flip` | Four XY flip variants |
| `do_tta_8yx` / `undo_tta_8yx` | Eight XY flip/rotation variants |
| `do_tta_8fliprot` / `undo_tta_8fliprot` | Active eight-way dihedral TTA |
| `do_tta_9public` / `undo_tta_9public` | Nine-variant public-style TTA implementation |

### Inference functions

| Function | Purpose |
|---|---|
| `predict_one` | Runs two-frame sliding inference, detection, feature sampling, edge scoring, and candidate collection |
| `predict_one_ensemble` | Averages point and edge predictions from multiple checkpoints; present in `biohub-v6-ultra-best-0.964.ipynb` but not activated in its recorded run |
| `run_worker` | Loads a checkpoint on one GPU, processes a subset of datasets, solves the graph, and writes GEFF |

### Hybrid post-processing functions

| Function | Purpose |
|---|---|
| `open_image_array` | Opens a dataset image for intensity-based repair |
| `distance_um` | Computes anisotropic physical distance |
| `refine_synthetic_node` | Moves a midpoint toward a local intensity centroid |
| `build_degrees` | Computes graph in-degree and out-degree |
| `find_components` | Finds weakly connected components with union-find |
| `postprocess_one_dataset` | Repairs one-frame gaps, filters short components, and writes rows |
| `row` | Creates a submission row |
| `augment_dataset` | Adds synthetic hub/fork structures to a graph |

### CPU baseline

`demo_test.ipynb` has no reusable function definitions. It:

1. reads raw Zarr chunks directly;
2. downsamples all three spatial axes by four;
3. smooths with `uniform_filter`;
4. thresholds at the 90th percentile;
5. labels connected components;
6. uses component centroids as nodes;
7. links consecutive frames with Hungarian assignment under 15 µm;
8. writes `submission.csv`.

This is useful as a sanity baseline, but it does not model divisions and uses a 15 µm link radius that is too permissive for a final system.

## 7. Existing post-processing and its risks

### 7.1 One-frame gap closing

The hybrid notebooks search for:

```text
source(t) → missing middle(t+1) → target(t+2)
```

For each pair of endpoint frames they:

1. collect track ends and starts;
2. compute physical distance;
3. run Hungarian matching;
4. accept pairs below a 10 or 12 µm threshold;
5. reuse an isolated middle-frame node if close enough;
6. otherwise insert a synthetic midpoint;
7. optionally move the midpoint toward a local intensity centroid;
8. connect source → middle → target.

This can recover detector dropouts, but it can also:

- connect two different cells during a crossing;
- create a false cell and two false edges;
- inflate the total-node penalty;
- create a false division context;
- overfit one public test set.

The `GAP_MAX_ADDED_FRAC` limit is a useful safety valve, but it must be tuned on embryo-disjoint validation.

### 7.2 Short-component filtering

The notebooks usually remove weak components shorter than three nodes, while retaining:

- components containing a division;
- components touching the first or last two frames.

This is a reasonable baseline, but it is not automatically correct. A real lineage may be visible for only one or two frames, and sparse detections can fragment a valid track. Measure the effect separately by component length and by dataset.

### 7.3 Negative-time hub/fork augmentation

The last cell in the hybrid notebooks:

- finds weakly connected components;
- connects their roots to a synthetic hub at a negative time;
- appends a chain of synthetic negative-time fork structures;
- writes these nodes and edges into the submission.

This is the main reason the notebooks are labeled “metric hack” or “last call.”

Problems:

- the nodes are outside the actual video time range;
- coordinates such as `-10000` are outside the image;
- the construction is not biological;
- it depends on evaluator behavior around unmatched and invalid-time nodes;
- it may violate hidden submission assumptions or future evaluator changes;
- it makes the solution hard to reproduce and defend;
- it hides the actual quality of the detector and linker.

Keep this code only as an isolated ablation. The canonical final path should emit valid nodes with:

```text
0 <= t < T
0 <= z < Z
0 <= y < Y
0 <= x < X
```

## 8. Important implementation findings

### Checkpoint loading is too permissive

Every neural notebook loads weights with:

```python
model.load_state_dict(state, strict=False)
```

The saved outputs report:

```text
missing key: ['D']
unexpected key: []
```

`D` is a dummy parameter added to expose the model device. That missing key is probably harmless, but `strict=False` can also hide a real architecture mismatch. Before serious experimentation:

1. remove the dummy parameter or add it explicitly to the checkpoint;
2. load with `strict=True`;
3. fail loudly on any unexpected parameter.

### Only one checkpoint split was actually used

`biohub-v6-ultra-best-0.964.ipynb` contains ensemble support, but its recorded run reports:

```text
Found 1 checkpoint splits: ['split_0']
Ensemble mode: False
```

This is not an ensemble. The first large improvement should be genuine checkpoint/fold diversity, not just more TTA.

### Local validation was not the default

The notebooks are configured with:

```python
MODE = "submit"
```

The local evaluation cells are either disabled or have no saved output. This leaves us without a trustworthy per-dataset breakdown of:

- edge TP/FP/FN;
- division TP/FP/FN;
- predicted node count;
- node recall;
- adjusted edge Jaccard;
- final score.

This is the biggest process weakness in the current workspace.

### The public score is not a training metric

The current model is trained around sparse edge supervision, but the competition score is a graph metric with:

- optimal node matching;
- endpoint-aware edge penalties;
- a total-node-count correction;
- local division topology.

Transformer BCE accuracy or node recall alone is not enough to choose a checkpoint.

## 9. Top-1 plan

### Phase A — Establish a trustworthy baseline

Create a single canonical pipeline before changing the model.

Required artifacts:

```text
src/
  data.py
  model.py
  inference.py
  graph_postprocess.py
  submission.py
  validation.py
configs/
  baseline.yaml
experiments/
  results.csv
```

The first validation run must:

1. group datasets by embryo ID;
2. hold out complete embryos;
3. load train Zarr + GEFF;
4. run the current U-Net/transformer checkpoint;
5. solve graphs with and without ILP;
6. run the official evaluator;
7. report per-dataset and aggregate metrics;
8. save predicted node/edge counts and error examples.

Do not submit another public experiment until this table exists.

Minimum validation output:

```text
dataset
pred_nodes
estimated_total_nodes
node_recall
edge_tp
edge_fp
edge_fn
edge_jaccard
adjusted_edge_jaccard
division_tp
division_fp
division_fn
division_jaccard
final_score
```

### Phase B — Reproduce the current notebook score honestly

Run the following ablations on the same held-out embryos:

| Ablation | Values |
|---|---|
| Point threshold | 0.85, 0.90, 0.93, 0.95, 0.97, 0.99 |
| Peak suppression | 2, 2.5, 3, 3.5, 4 µm |
| Edge activation | softmax over parents, sigmoid, calibrated softmax |
| Strong edge threshold | 0.35, 0.45, 0.50, 0.60 |
| Top-k parents | 1, 2, 3, 4 |
| Candidate radius | 7, 8, 10, 12, 15 µm |
| Parent cap | 1 |
| Child cap | 1, 2 |
| ILP disappearance cost | 0.0, 0.5, 1.0, 1.4, 2.0 |
| ILP division cost | 0.25, 0.5, 1.0, 1.5 |
| Gap closing | off, conservative, current |
| Short-component filter | off, length 2, length 3, length 4 |

Select settings by adjusted edge Jaccard first and final score second. Do not select by public leaderboard noise after one submission.

### Phase C — Calibrate detection count

The current detector is trained against sparse annotations, so a low detection threshold can produce many plausible but unannotated cells. Use `estimated_number_of_nodes` for validation calibration:

```text
target count per sample ≈ estimated_number_of_nodes
```

For each threshold, measure:

- predicted/estimated count ratio;
- node recall;
- edge Jaccard;
- adjusted edge Jaccard;
- final score.

Use a per-embryo or per-sample threshold only if it is learned from training folds and does not use hidden test information.

The goal is not maximum node recall. The goal is the best adjusted edge score.

### Phase D — Improve detection

The current `[1,4,4]` input is an efficient baseline. Test:

1. XY stride 2 with smaller spatial patches;
2. full-resolution XY tiled inference;
3. a two-stage detector:
   - low-resolution full-video candidate proposal;
   - high-resolution local centroid refinement;
4. anisotropic convolutions or anisotropic pooling;
5. heatmap plus offset regression;
6. temporal consistency loss between adjacent frames;
7. intensity/photobleaching augmentation;
8. depth-aware brightness normalization;
9. hard-negative mining around false peaks.

The best practical architecture is likely not a large foundation model. It is a well-calibrated detector that preserves close lateral neighbors without destroying the 12-hour notebook budget.

### Phase E — Improve linking

The current transformer sees two frames at a time. Add features that match the real error modes:

- physical displacement `(Δz, Δy, Δx)` in microns;
- displacement magnitude and direction;
- local intensity and shape descriptors;
- appearance similarity across a three-frame window;
- previous velocity from `t-1 → t`;
- candidate daughter-pair geometry;
- parent-to-daughter distance and divergence angle;
- edge uncertainty and margin to the second-best parent.

Candidate generation should remain permissive, but final graph selection should be globally constrained.

Recommended association formulation:

```text
candidate edges:
    all physically plausible pairs within a validation-tuned radius

edge score:
    appearance + feature similarity
  + motion prior
  + physical distance prior
  + temporal consistency

global solver:
    one parent per target
    at most two children per source
    explicit division penalty
    appearance/disappearance penalties
```

### Phase F — Train for sparse labels

The sparse-label setting means an unannotated cell is not a negative example. Training should:

- mask unlabeled node rows and columns;
- use verified ground-truth edges as positives;
- avoid treating every missing edge as a hard negative;
- add synthetic node dropout and gap simulation;
- use temporal consistency on unlabeled detections;
- use count/density regularization only with care;
- create hard negatives from nearby cells and identity switches.

The official reference training pipeline already masks inactive rows and columns for sparse edge supervision. Build on that rather than replacing it with dense BCE.

### Phase G — Learn division events explicitly

Division Jaccard contributes only 0.1, but divisions can still separate near-tied solutions.

Add a division head or a post-link classifier using:

- one parent and two daughter candidate embeddings;
- daughter separation in physical units;
- parent-to-daughter displacement;
- daughter intensity/shape change;
- local motion direction;
- whether the two branches remain distinct for one or two later frames.

Train division examples with synthetic temporal shifts and missing detections. Score divisions with the official directed local evaluator, not weak connected-component overlap.

### Phase H — Real ensemble

Use diversity that changes errors:

- 3–5 embryo-disjoint folds;
- different XY resolutions;
- different seeds;
- checkpoints from different training epochs;
- detector/linker variants;
- TTA only after model diversity is established.

Ensemble procedure:

1. average detection logits in physical coordinate space;
2. merge detections with anisotropic NMS;
3. average or rank-normalize edge logits;
4. solve one final global graph;
5. calibrate the final node count.

Running the same `split_0` checkpoint multiple times is not an ensemble.

### Phase I — Submission and rule compliance

The rules prohibit using hand labeling or human prediction of validation/test records. The rules allow publicly available, reasonably accessible external data and tools, but a winning submission must be reproducible and winners may be required to provide training code, inference code, environment details, hyperparameters, and a repository link. The competition also requires MIT licensing of winning submission code, while the competition data itself is CC0.

Do not use any hidden test labels, manual corrections, or private data. Treat the metric quirks as engineering knowledge, not permission to fabricate biological records.

### Phase J — Final notebook engineering

The final notebook must:

- install only offline attached wheels;
- avoid internet access;
- run comfortably below 12 hours;
- process all test datasets;
- write exactly `submission.csv`;
- use consecutive `id` values;
- include every test dataset;
- use unique node IDs per dataset;
- ensure every edge references nodes in the same dataset;
- ensure node coordinates and times are valid;
- print counts and timing for every stage;
- save no hidden dependency on a local filesystem path.

## 10. Submission contract

Required columns:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

Node row:

```text
row_type   = node
node_id    = unique integer
t,z,y,x    = integer node coordinates
source_id  = -1
target_id  = -1
```

Edge row:

```text
row_type   = edge
node_id    = -1
t,z,y,x    = -1
source_id  = valid source node ID
target_id  = valid target node ID
```

Before submission, run these checks:

```python
required = [
    "id", "dataset", "row_type", "node_id", "t",
    "z", "y", "x", "source_id", "target_id",
]

assert list(df.columns) == required
assert df["id"].tolist() == list(range(len(df)))
assert set(df["row_type"].unique()) <= {"node", "edge"}
assert df["dataset"].nunique() == len(test_dataset_names)
assert set(test_dataset_names) == set(df["dataset"].unique())

nodes = df[df.row_type == "node"]
edges = df[df.row_type == "edge"]

for dataset, group in df.groupby("dataset", sort=False):
    group_nodes = group[group.row_type == "node"]
    group_edges = group[group.row_type == "edge"]
    assert group_nodes.node_id.is_unique
    assert not group_edges[["source_id", "target_id"]].duplicated().any()
    assert (group_nodes.t >= 0).all()
    assert (group_nodes.z >= 0).all()
    assert (group_nodes.y >= 0).all()
    assert (group_nodes.x >= 0).all()
    assert set(group_edges.source_id).issubset(set(group_nodes.node_id))
    assert set(group_edges.target_id).issubset(set(group_nodes.node_id))

    node_time = dict(zip(group_nodes.node_id, group_nodes.t))
    assert all(
        node_time[source] + 1 == node_time[target]
        for source, target in group_edges[
            ["source_id", "target_id"]
        ].itertuples(index=False, name=None)
    )

    assert not group_edges.target_id.duplicated().any()
    assert group_edges.groupby("source_id").size().max() <= 2
```

Validate per dataset, not only globally. The official format requires edge references to resolve within their dataset. Node IDs only need to be unambiguous inside each dataset when all tooling consistently uses `(dataset, node_id)` as the key.

## 11. Public/private leaderboard risk

The public leaderboard is based on a representative test subset; the private leaderboard determines final placement. A post-processing rule that improves the public score by a few thousandths may be exploiting the visible subset rather than improving generalization. The test set is swapped for hidden test data during notebook reruns, so every candidate must be robust to a new hidden test set.

The competition currently allows up to five submissions per day and up to two final submissions. Reserve submissions for hypotheses that already improve embryo-disjoint local validation. As of this audit, the first-place public score is 0.970.

## 12. Recommended experiment ledger

Every experiment should record:

```text
experiment_id
git_commit
checkpoint
fold
embryos_train
embryos_valid
point_threshold
pool_kernel_um
edge_activation
edge_threshold
top_k
candidate_radius_um
ilp_weights
gap_settings
component_filter
predicted_nodes
estimated_nodes
node_recall
edge_tp
edge_fp
edge_fn
edge_jaccard
adjusted_edge_jaccard
division_tp
division_fp
division_fn
division_jaccard
final_score
runtime
memory
notes
```

The public leaderboard should be used only after local validation has identified a meaningful candidate. With five submissions per day and only two final selections, random tuning is expensive.

## 13. Suggested first three experiments

### Experiment 1 — Metric-faithful reproduction

Use the current `biohub-solution-0.966.ipynb` model without fake negative-time augmentation.

Compare:

```text
ILP off
ILP on
gap closing off
gap closing on
short-component filtering off
short-component filtering on
```

This tells us whether the reported gain comes from the learned graph or from post-processing.

### Experiment 2 — Count calibration

Sweep `POINT_THRESHOLD` and `pool_kernel_um`. Keep only settings whose predicted node count is close to the GEFF estimate on held-out embryos.

Expected outcome: a small loss in raw node recall can produce a larger gain in adjusted edge Jaccard.

### Experiment 3 — Real fold/checkpoint ensemble

Train at least three embryo-disjoint folds and compare:

```text
best single fold
average detector logits
average detector + edge logits
rank-average candidate edge scores
```

Solve the merged candidates with one ILP pass and evaluate all variants locally.

## 14. Current recommendation

Use this as the canonical baseline:

```text
input:
    Zarr, quantile normalization, [1,4,4] subsampling

model:
    TemporalUNet3D + point head + SimpleNodeTransformer

inference:
    XY 8-way TTA
    physical 3 µm peak suppression
    softmax over possible parents
    candidate top-k parents
    physical candidate radius
    tracksdata ILP solver

post-processing:
    no fake hub
    no negative-time nodes
    conservative one-frame gap closing only after validation
    conservative component filtering

selection:
    official evaluator
    adjusted edge Jaccard first
    final score second
```

The most likely path from the current reported 0.966 level to the current public 0.970 target is:

1. proper embryo-disjoint validation;
2. detection-count calibration;
3. true fold/checkpoint ensemble;
4. improved physical candidate/link scoring;
5. conservative, validated gap repair;
6. explicit division handling.

## 15. Source notes

### Primary sources

- Kaggle competition pages and rules.
- `royerlab/kaggle-cell-tracking-competition`.
- `metrics.md`.
- `src/tracking_cellmot/metrics.py`.
- `src/tracking_cellmot/division_metrics.py`.
- `scripts/train_unet_transformer.py`.
- `scripts/predict_unet_transformer.py`.

### Secondary source

The public `pomagrenate/biohub-cell-tracking` repository was also inspected for alternative architecture ideas such as AC-Net, anisotropic preprocessing, and graph-flow concepts. Those ideas are proposals, not verified competition facts and are not currently implemented in this workspace.

### Workspace evidence

The notebook-specific runtime, node counts, edge counts, and configuration values in this README were extracted from:

- `biohub-competition-solution-0.966-2.ipynb`
- `biohub-solution-0.966.ipynb`
- `biohub-v6-ultra-best-0.964.ipynb`
- `improved-metric-hack-last-call_0.966.ipynb`
- `demo_test.ipynb`
