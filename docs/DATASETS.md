# Training data: sources and licences

The model is trained on about 2,850 images merged from three Roboflow Universe datasets,
remapped to exactly two classes: `fire` and `smoke`. The merge is reproducible with
[`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py), and the `SOURCES` list in
that file is the source of truth.

## Selected

| Source | Licence | Images used | Imagery | Class mapping |
|---|---|---|---|---|
| [master-candidate / forest-fire-and-smoke-rf4pd](https://universe.roboflow.com/master-candidate/forest-fire-and-smoke-rf4pd), v23 | CC BY 4.0 | 1,500 | Mostly aerial and oblique drone views of forest fires | `fire`→fire, `smoke`→smoke |
| [D-Fire](https://github.com/gaiasd/DFireDataset), via the Universe copy [fire-detection-nlrcv / 9.5ai-d-fire](https://universe.roboflow.com/fire-detection-nlrcv/9.5ai-d-fire), v1 | CC0 1.0 upstream; CC BY 4.0 on Universe | 1,200 | Ground-level and surveillance cameras | `Fire`→fire, `Smoke`→smoke |
| [tim-4ijf0 / drone-fire-detection-byija](https://universe.roboflow.com/tim-4ijf0/drone-fire-detection-byija), v1 | CC BY 4.0 | 152 (all) | True top-down drone, small fires | `fire`→fire |

How the merge works:

- **Removing augmented copies:** rf4pd v23 is exported with 3× augmentation, so the script
  keeps only one image per original file. Augmentation is applied once, in our own Roboflow
  version.
- **Avoiding overlap with D-Fire:** rf4pd contains some D-Fire images (files starting
  `WEB` or `AoF`). Those are dropped from rf4pd so the same image can't appear twice.
- **Images without boxes:** any image left with no `fire` or `smoke` box after remapping is
  skipped.
- **Splits:** each image keeps the train/valid/test split its source assigned.
- **Random seed:** fixed at 26.

D-Fire credit: Pedro Vinícius Almeida Borges de Venâncio et al., *"An automatic fire detection
system based on deep convolutional neural networks for low-power, resource-constrained
devices"*, Neural Computing and Applications, 2022. The D-Fire README asks users to respect
the original providers' terms for images named `PublicDataset*`.

## Considered and rejected

| Dataset | Why not |
|---|---|
| hephaestus-drone-project / fire-and-smoke-detection-we2wg | Empty project (0 images) |
| ai-faogz / aerial-fire-detection | Appears to be a fork of tim-4ijf0's set, with a broken class name |
| fire-ebsui / wildfires-from-uav-perspective | Frames taken from YouTube videos with watermarks, so the CC BY licence is doubtful; fire only |
| auebmsc / forest-fire-and-smoke-detection-using-uav-imaging_2 | Good aerial fire and smoke (CC BY 4.0), but it has no dataset version, so it can only be used by forking it in the web UI. Kept as an optional top-up. |
| yangin-pjx4c / drone-fire-smoke-o58av, athrva-cbsa8 / drone-vision-system-rnc2d | Ground-level despite "drone" in the name; mostly D-Fire inside |
| rf20-vl / wildfire-smoke-myxt | MIT licence, but smoke only from fixed mountaintop cameras, likely overlapping D-Fire |
| forestfiresmoke / d-fire-v6exb, vasiliss-workspace / d-fire-aqheb | Other D-Fire copies (an extra `object` class, or no version) |

The shortlist was reviewed in November 2026 from the public Universe pages. "Aerial" and
"ground-level" were judged from a sample of thumbnails, not a full audit.
