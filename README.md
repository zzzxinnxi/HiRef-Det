# HiRef-Det

**HiRef-Det: relation-aware high-resolution feature refinement for small-object detection in unmanned aerial vehicle imagery**

Source, paper evaluators, selected training records and recovered checkpoints for the VisDrone2019-DET and SeaDronesSee ODv2 experiments.

## Download

- **[Complete source and checkpoints — v0.2](https://github.com/zzzxinnxi/HiRef-Det/releases/download/v0.2/HiRef-Det_complete_with_weights.zip)**: the recommended package, including the modified runtime, 15 model configurations, 34 selected training-record groups, 33 unique checkpoints, paper evaluators and recovered plotting materials.
- [Source-only package](https://github.com/zzzxinnxi/HiRef-Det/raw/refs/heads/main/HiRef-Det_source.zip): the same packaged source and records, without checkpoint files.
- [Release page](https://github.com/zzzxinnxi/HiRef-Det/releases/tag/v0.2).

Download and extract the **complete package**, then run commands from its `HiRef-Det/` directory. Checkpoints are already under `weights/`; no additional weight download is needed. Dataset images and annotations must be obtained separately.

The complete code and record tree is preserved inside the packages. Only part of the implementation is currently exposed as browsable files on GitHub. The automatically generated GitHub “Source code (zip)” attachment is a snapshot of the repository; use the named complete package above for the assembled source and checkpoints.

Complete-package SHA-256:

```text
75abc1ade292590186ec824ba8d60d2dbf257772fb5923e5b64908a2af7d7fee
```

## Environment

The archived YOLO training environment used Python 3.12, PyTorch 2.3.0, torchvision 0.18.0, CUDA 12.1 and a modified Ultralytics 8.4.26 runtime. Start with an isolated environment:

```bash
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Choose an appropriate CPU build for CPU inference. Entry scripts select `vendor/ultralytics`; an unmodified Ultralytics installation is insufficient for the custom modules. LCDA retains the serialized implementation name `LDCA`; the manuscript's LCDA uses `sum` mode. External comparison frameworks require separate environments and their own dependencies/licenses. The requirements file is a suggested dependency set, not an exact environment lock.

## Prepare data and train

Place the VisDrone2019-DET train/validation split directories under `datasets/VisDrone/`, alongside its YAML. Each original split contains `images/` and `annotations/`.

```bash
python scripts/visdrone2yolo.py --dataset-root datasets/VisDrone
python scripts/train_canonical_yolo11n_baseline.py
python scripts/train_generic.py --config yolo11n-p2-ldca-sum-csgrf-p2.yaml --name hiref_det --seed 42 --epochs 100 --batch 16
```

The model YAMLs include P2, LCDA branch/placement and CSGRF relation/sharing/placement/initialization ablations. Exact historical settings and training curves are in `paper/training_records/` and its `index.json`. These records retain their original training paths. Different seeds and environments need not reproduce a recorded value exactly.

For SeaDronesSee ODv2, use the official image/annotation archives:

```bash
python scripts/seadronessee_prepare.py --images-zip /path/to/images.zip --annotations-zip /path/to/annotations.zip --dataset-root datasets/SeaDronesSee_ODv2
python scripts/seadronessee_train.py train baseline full
```

## Evaluation protocols

Native Ultralytics validation:

```bash
python scripts/evaluate_yolo_ultralytics_paper.py --model weights/hiref_det_visdrone.pt --data datasets/VisDrone/visdrone.yaml --output-dir outputs/native_hiref --device 0
```

Unified paper COCO evaluation:

```bash
python scripts/visdrone_yolo_to_coco.py --dataset-root datasets/VisDrone --output-dir datasets/VisDrone/coco
python scripts/evaluate_yolo_coco_paper.py --model HiRefDet=weights/hiref_det_visdrone.pt --images-dir datasets/VisDrone/VisDrone2019-DET-val/images --gt-json datasets/VisDrone/coco/instances_val.json --output-dir outputs/coco_hiref --category-offset 0 --device 0
```

The unified protocol uses original-image box areas, IoU 0.50:0.95, image size 640, confidence 0.001, NMS IoU 0.7, `rect=False` and `maxDets=[1,10,300]`. AP is read from the accumulated precision tensor at maxDets=300. The standard summary's hard-coded AP100 entry must not be substituted. The recovered evaluators use **pycocotools**.

For SeaDronesSee, run `scripts/seadronessee_yolo_to_coco.py --dataset-root datasets/SeaDronesSee_ODv2`; then use its `coco_annotations/instances_val.json`, the `images/val` directory and `--category-offset 1`. Use `--device cpu` for CPU inference.

To score existing COCO-format predictions, including external-framework predictions:

```bash
python scripts/evaluate_coco_predictions_paper.py --gt-json /path/to/instances_val.json --predictions /path/to/predictions.json --output outputs/metrics.json --log outputs/cocoeval.txt
```

Native mAP, unified COCO AP and class-wise AP are distinct quantities. Use `scripts/profile_yolo_paper.py` for the recovered FP32 profiling procedure; see its `--help` for arguments.

## Records, figures and validation

After extraction:

- `weights/manifest.json` lists checkpoint roles, byte sizes and SHA-256 hashes. Checkpoint tensors are unchanged.
- `paper/training_records/` and `paper/evaluation_records/` contain selected training curves, compact metrics and resolved comparison configs.
- `paper/README.md` maps materials to experiment families; `paper/VALIDATION.md` records executed checks.
- `archives/figure_generation.zip` contains recovered plotting scripts and selected-image/confusion-matrix inputs. `archives/final_figure_edits.zip` preserves later Figure 6/7/8 edits and Figure 8 replay coordinates. Historical paths and fonts require adaptation, and dataset images are not redistributed.
- External comparison source snapshots retain their available upstream license files. Original module/parser variants are archived for traceability.

Five experiment instances were independently scanned and recovered files were checked against SHA-256 manifests. A sixth drawing-only instance was excluded based on the prior handoff and was not independently rescanned. Failed architecture experiments, cloud orchestration/shutdown helpers and oversized prediction JSON files are excluded.

All 15 released model configurations constructed and completed synthetic CPU forward passes using the packaged runtime. The final detector and a shared-calibration checkpoint loaded and completed CPU forward passes. Both COCO evaluator paths correctly scored 120 perfect small-object detections with AP300=1, APS300=1 and AR300=1. Source syntax and archive integrity were checked. Full training, clean-environment installation, full-dataset accuracy and all external-framework runs have not been rerun during packaging.

Not all non-YOLO comparison checkpoints are included. NanoDet's custom config/profiler were recovered, but its original complete framework snapshot and exact source revision were not verified. External snapshots may need native build assets from upstream. Literature-only comparison rows refer to cited published results rather than locally reproduced experiments.

## License

The modified Ultralytics-based project is distributed under [AGPL-3.0](LICENSE). External frameworks retain their own licenses and notices. See the packaged `THIRD_PARTY_NOTICES.md` and `archives/framework_index.json` for the recovered third-party materials.
