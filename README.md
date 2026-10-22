# HiRef-Det

**HiRef-Det: relation-aware high-resolution feature refinement for small-object detection in unmanned aerial vehicle imagery**

This initial release contains the model source, configurations, training records and checkpoints recovered for the **VisDrone2019-DET** experiments.

## Downloads

- [Source package](HiRef-Det_source.zip): LCDA/CSGRF implementations, modified Ultralytics 8.4.26 runtime, model configurations, scripts, and training records.
- [HiRef-Det checkpoint](hiref_det_visdrone.pt)
- [YOLO11n baseline checkpoint](yolo11n_baseline_visdrone.pt)
- [P2 checkpoint](yolo11n_p2_visdrone.pt)
- [P2 + LCDA checkpoint](yolo11n_p2_lcda_visdrone.pt)
- [YOLO11n pretrained initialization](yolo11n_pretrained.pt)
- [Checkpoint sizes and SHA-256 hashes](weights_manifest.json)

Download and extract `HiRef-Det_source.zip`. Put the five `.pt` files into the extracted `HiRef-Det/weights/` directory. Source code is currently distributed as a ZIP archive; the folder structure is preserved inside it.

## Setup and use

Run commands from the extracted `HiRef-Det` directory. The archived training environment used Python 3.12, PyTorch 2.3.0 with CUDA 12.1, torchvision 0.18.0 and modified Ultralytics 8.4.26. Start with a separate Python environment:

```bash
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Use a suitable PyTorch CPU build if no CUDA GPU is available. Entry scripts select the bundled modified runtime automatically. Do not replace it with an unmodified Ultralytics installation.

Obtain VisDrone2019-DET separately and place its split directories under `datasets/VisDrone/`, beside `visdrone.yaml`. Each split contains `images/` and original `annotations/`; convert annotations before training:

```bash
python scripts/visdrone2yolo.py --dataset-root datasets/VisDrone
python scripts/train_generic.py --config yolo11n-p2-ldca-sum-csgrf-p2.yaml --name hiref_det --seed 42 --epochs 100 --batch 16
```

The other model configurations are `configs/yolo11n-p2.yaml` and `configs/yolo11n-p2-ldca-sum.yaml`. Train the baseline with:

```bash
python scripts/train_yolo_baseline.py --model weights/yolo11n_pretrained.pt --name yolo11n_baseline --seed 42
```

Prediction and native Ultralytics validation:

```bash
python scripts/predict.py --model weights/hiref_det_visdrone.pt --source path/to/images --device 0
python scripts/evaluate_yolo_ultralytics_paper.py --model weights/hiref_det_visdrone.pt --data datasets/VisDrone/visdrone.yaml --output-dir outputs/native_hiref --device 0
```

Use `--device cpu` for CPU prediction or validation. The implementation class `LDCA` in **`sum` mode** corresponds to the LCDA module in the manuscript. Its serialized name is preserved for checkpoint compatibility. Original `args.yaml` and `results.csv` files for the four selected training runs are in `runs/detect/`.

## Current scope and validation

Checkpoint hashes, source syntax and ZIP integrity have been checked. This packaged copy has **not yet been tested in a clean environment** for training or inference.

The validation script above reports **native Ultralytics metrics**. The original unified COCO-style evaluator used for the manuscript has not yet been located and is not included; native metrics should not be equated with the manuscript's unified COCO-style AP. SeaDronesSee ODv2 materials are stored on a separate instance and are not included in this release.

See `provenance/RELEASE_NOTES.md` inside the source package for packaging changes and archived environment details. No dataset images or failed-experiment files are included.

## License

The source bundle includes modified Ultralytics code and is distributed under [AGPL-3.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).
