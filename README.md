# Does monocular depth help few-shot 2D object detection? A controlled pilot

**Question.** When a detector adapts to new classes from only K labelled examples, does adding geometric
information derived from an *estimated* depth map (RGB only, no depth sensor) improve 2D box detection of the
novel classes?

This is a controlled empirical study / college project, framed as perception for robotics. It does **not**
claim a new state of the art, does not use 3D boxes, LiDAR or RGB-D input, and was not deployed on a robot.
Related work (e.g. FI3Det) already combines 3D cues with few-shot detection, so the contribution here is the
controlled comparison, not novelty of the idea.

## Pipeline
RGB image -> Depth Anything V2-Small (monocular depth, cached) -> per-RoI geometric descriptor (16-d, from the
depth map and box geometry only) -> depth-conditioned FiLM + gate on the Faster R-CNN box-head features ->
2D boxes.

- Detector: torchvision Faster R-CNN, ResNet-50 FPN, cosine classifier.
- Split: PASCAL VOC07+12 trainval / VOC07 test, split 1 (novel: bird, bus, cow, motorbike, sofa).
- K in {1, 5, 10}; seeds 0-2.

## Method (DGA-Det)
Geometry stage: starting from the base model, train only the geometry modules (~144K params) and the base
box predictor on base-class data. Few-shot stage: identical to the baseline fine-tuning (only the predictor is
trained), with depth maps passed through the detector at train and test time.
At initialisation the model is exactly the baseline (FiLM is zero-initialised).

## Variants and controls (all use the same schedule and seeds)
| Label | What it is | What it tests |
|---|---|---|
| A0 | Base model + standard few-shot fine-tuning | reference |
| OFF | A0 + the extra geometry-stage training, no geometry branch | fairness: is any gain just extra training? |
| DEPTH | Real estimated depth | the method |
| SHUFFLE | Depth map of a different image | does the *right* depth matter? |
| CONST | Constant depth map, box geometry only | does the depth content matter? |
| GRAY | Grayscale RGB | does the effect need colour, or is it depth? |

A negative result (depth does not help, or hurts) is a valid outcome and will be reported as such.

## Results (novel AP50, mean ± std over seeds; full VOC07 test, 4952 images)
Only real runs are listed. Empty cells mean the run has not been done.

| K | A0 | OFF | DEPTH | SHUFFLE | CONST |
|---|---|---|---|---|---|
| 1 | 26.23 ± 2.59 (n=3) | | | | |
| 5 | 40.28 ± 5.05 (n=3) | | | | |
| 10 | 52.47 ± 3.42 (n=3) | | | | |

Base model: AP50 79.42 on the 15 base classes. Tables are produced by `scripts/aggregate_results.py`
(per-seed paired differences are reported, since the seed variance is large).

## Limitations (read before citing any number)
- Reduced base schedule (24k iterations), ResNet-50 instead of ResNet-101: absolute numbers are below published
  TFA numbers and are not comparable to them.
- K-shot sets come from our own seeded sampler, not the official TFA support files.
- Depth is estimated once by one model (Depth Anything V2-Small); one reference depth cache is used everywhere.
- Single geometry-stage seed; few-shot seeds 0-2 only. Few seeds -> differences smaller than the seed std are
  not evidence.
- Geometry-stage hyper-parameters are untuned first guesses, identical across variants.
- The geometric-prototype logit term and contrastive geometry loss from the original plan are not implemented.
- Depth-estimation cost is not included in ms/image; it is reported separately.
- Split 1 and VOC only; no COCO, no other splits.

## Reproduce
    python scripts/download_voc.py --out data
    python scripts/check_voc.py --root data/VOCdevkit
    python scripts/make_fewshot.py
    python scripts/validate_project.py
    python scripts/precompute_depth.py ...                       # depth cache
    python scripts/train_geo.py --config configs/geo_split1.yaml --base-ckpt <base.pth> --out-dir runs/geo_depth --mode depth
    python scripts/run_fewshot_sweep.py --config configs/fewshot_dga.yaml --base-ckpt <base.pth> \
        --geo-ckpt runs/geo_depth/geo_final.pth --out-dir runs/dga_depth --seeds 0 1 2 --shots 1 5 10 \
        --set experiment=dga_depth_fs_split1
    python scripts/aggregate_results.py A0=... OFF=... DEPTH=...

## Layout
configs/ fsod_data/ models/{fsod,heads,geo}/ preprocessing/ training/ evaluation/ scripts/

## Note: the first geometry-stage attempt was inactive (documented, fixed)

The first geometry stage trained the zero-initialised FiLM layer with the same SGD as the box predictor.
Training loss looked normal, but `scripts/check_geo_effect.py` showed the module was barely used:

| Check | First run (SGD) | After fix (AdamW for geometry modules) |
|---|---|---|
| Largest FiLM weight | 0.00025 | 0.153 |
| Feature change from FiLM | 0.002 % | 3.9 % |
| Real vs flat depth, output change | 0.0001 % | 0.41 % |

Cause: the base model is already converged, so gradients into a zero-started module are tiny and SGD steps shrink with them.
Fix: the geometry modules use AdamW (`train.geo_lr: 0.0003` in `configs/geo_split1.yaml`); the box predictor keeps SGD.
The same fix applies to every geometry mode (depth, shuffle, const), so comparisons stay fair.
Always run `check_geo_effect.py` after a geometry stage and confirm it prints "active".

## Aggregating results

    python scripts/aggregate_results.py --out results_tables.md \
      A0=runs/a0/results.csv OFF=runs/dga_off/results.csv DEPTH=runs/dga_depth/results.csv \
      SHUFFLE=runs/dga_shuffle/results.csv CONST=runs/dga_const/results.csv

Results are mean ± std over seeds 0–2, with paired per-seed differences. Cells stay empty until real runs exist.
