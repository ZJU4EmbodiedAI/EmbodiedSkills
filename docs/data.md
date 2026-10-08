# Feature and supervision format

The training entry point accepts separate `.npz` files for training and validation. Arrays use ordinary NumPy numeric or Unicode dtypes; loading disables pickle. Every row corresponds to a decision anchor and its five policy proposals. Candidate zero is the frozen policy's direct proposal in every candidate-indexed array.

## Current inputs

| Field | Spatial shape | Continuous shape | Meaning |
| :--- | :--- | :--- | :--- |
| `visual` | `[N,64,1024]` | `[N,1024]` | Current frozen visual features |
| `state` | `[N,4]` | `[N,S]` | Deployment-visible state summary |
| `language` | `[N,384]` | `[N,384]` | Frozen MiniLM instruction embedding |
| `progress` | `[N,1]` | `[N,1]` | Consumed execution budget divided by maximum budget |
| `actions` | `[N,5,20]` | `[N,5,H,A]` | Candidate actions in their original units |
| `mask` | — | `[N,5,H]` | Binary validity mask for chunk timesteps |

The spatial preset uses CLIPort's fused RGB view, height-map summary and native pick/place action representation. The continuous preset uses state dimension `S=4`, execution horizon `H=8` and action dimension `A=4`. These continuous dimensions are configurable. Current and future pooled visual features are L2-normalized. Spatial token features retain the encoder's output scale.

State, language and action statistics for continuous models are fitted on training data and applied inside the runtime. Supply physical inputs to both training archives and inference. The spatial model normalizes state and actions in its candidate-context encoder. Language embeddings follow MiniLM's attention-mask pooling and L2-normalization convention.

## Supervision and identity

| Field | Shape | Purpose |
| :--- | :--- | :--- |
| `future_visual` | `[N,5,64,1024]` or `[N,5,1024]` | Visual representation after each candidate's execution horizon |
| `success` | `[N,5]` | Binary branch outcomes under the chosen outcome horizon |
| `return` | `[N,5]` | Continuous-model branch returns |
| `expert_utility` | `[N,5]` | Continuous-model expert-alignment supervision |
| `task` | `[N]` Unicode strings | Task balancing and per-task evaluation |
| `seed` | `[N]` integers | Episode identity |
| `anchor` | `[N]` integers | Decision identity within an episode |

`return` and `expert_utility` are required by the continuous training preset. They are supervision fields. The model's forward inputs are the current tensors in the first table. Task names identify sampling groups and return-normalization statistics; the network receives the instruction embedding.

Collect the five branches from the same saved simulator state, preserving proposal order. Record future features immediately after the candidate executes. Record success according to the collection protocol: a short action horizon and a complete episode continuation represent different outcome targets and should be kept consistent within a dataset. For CLIPort first-step branches, the frozen policy continues after the selected initial primitive. For continuous control, each row can describe a candidate chunk at a later decision anchor.

All anchors from a `(task, seed)` pair belong to the same partition. The trainer rejects overlap between training and validation episodes. Keep evaluation episodes separate from both. Checkpoint selection uses the supplied validation archive. Cached branch metrics have one row per anchor; a multi-anchor archive reports anchor-level branch outcomes. Whole-episode success is measured by the environment during closed-loop evaluation.

## Extracting frozen features

To encode a raw archive, provide `rgb` as `[N,256,256,3]` uint8, `instruction` as `[N]` Unicode strings, and the remaining state/action/identity fields. Training archives can include `future_rgb` as `[N,5,256,256,3]` uint8 along with supervision labels.

```bash
embodiedskills encode \
  --family spatial \
  --input data/raw_train.npz \
  --output data/spatial_train.npz \
  --visual-model /path/to/vjepa2 \
  --language-model /path/to/all-MiniLM-L6-v2 \
  --device cuda:0
```

Encoding replaces the RGB and instruction fields with visual and language features. The same preprocessing implementation is available to online adapters through `FrozenPublicVisualEncoder` and `LanguageEncoder`. Use identical encoder weights, image preprocessing and candidate ordering when collecting training labels and running the selector.

The supplied model dimensions expect 1024-dimensional visual features and 384-dimensional language features. The optional DINOv2 encoder interface accepts a backbone with a matching feature width; changing visual representations requires corresponding WAM training and statistics.
