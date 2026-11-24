# IWT–Schur–SVD: final reproducible comparison

One proposal: **IWT–Schur–SVD + FSDS–Givens–HMAC–BCH**. Three paper-derived baselines: Luo 2021, Ganic–Eskicioglu 2005, Liu–Tan 2002. No alternate proposed methods in the primary comparison. Seven optional ablations are separately labeled. Read REFERENCES.md before using results in a paper.

## Setup (Ubuntu / Kaggle)

```bash
python -m pip install -r requirements.txt
python validate.py --data data.zip --host athens --size 64
```

`data.zip` is the user-supplied dataset, bundled unchanged. No downloads of images are required. Pillow must have OpenJPEG support (`python -c 'from PIL import features; assert features.check("jpg_2000")'`). For GPU machines this code still runs on CPU. Each process uses one BLAS thread. Set `--workers` according to CPU resources.

## Run the complete experiment

```bash
# 1. Calibrate baseline strengths on the five training hosts only.
python benchmark.py --data data.zip --phase calibrate --out results/calibration --sizes 64,128,256
# 2. Locked evaluation on the seven test hosts, with the full 45 attacks + clean.
python benchmark.py --data data.zip --phase benchmark --config results/calibration/locked_config.json --split test --out results/test --workers 4
# 3. Report train results separately using the identical locked config.
python benchmark.py --data data.zip --phase benchmark --config results/calibration/locked_config.json --split train --out results/train --workers 4
# 4. Optional ablations on the same held-out test cases.
python benchmark.py --data data.zip --phase benchmark --config results/calibration/locked_config.json --split test --ablation-only --out results/ablation_test --workers 4
```

Run all four primary methods on both splits in one command with `--phase all --out results/full`; add `--ablation` to include controls. The calibration sweep and exact Schur extraction can be computationally expensive. Each completed host/watermark/size case is atomically written. Resume evaluation with the same command, `--resume`, and the same frozen config and output directory. Use a new output directory if you change settings. A fast pipeline smoke check without calibration is:

```bash
python benchmark.py --data data.zip --phase benchmark --smoke --sizes 64 --split test --out results/smoke_test --workers 1
```

`--phase benchmark` without `--config` uses the explicitly declared reference/default settings, **not** calibrated parameters. Ganic defaults are the published strengths; Luo defaults are an initial pair within the published search grid and an assumed T, Liu defaults are size-adjusted experimental strengths. The full recommended workflow is train-only calibration followed by locked evaluation. Proposed alpha=.12/.06/.03 (64/128/256) and signature cap=.4, tau=40 are frozen from previous development, not tuned on this 45-attack test. All 12 images have appeared in earlier development sessions; do not describe the seven evaluation hosts as never observed across the project. A genuinely untouched dataset is still needed to claim independent generalization.

## Fixed protocol

- Train: airplane, baboon, house, pepper, Girl. Test: athens, butfly, manhatan, milkdrop, safari, sailboat, tiffany. Two watermarks (logo alpha composited onto white and wm), sizes 64/128/256 by bicubic resizing. The benchmark treats watermark gray values as [0,255].
- 45 classical attacks from the supplied LaTeX table plus clean. Common deterministic noise seed 1234 across methods; Gaussian and speckle variances in normalized [0,1] units; salt/pepper pixelwise. JPEG Pillow; JP2 rate target via OpenJPEG; median OpenCV; average reflect; lowpass normalized 5x5 binomial; Pillow Gaussian blur radius and sharpening factor 1+listed strength; geometric interpolation and fill follow attacks.py. Center occlusion 25% area; crop-resize 50% retains center 50% area; translations/shear black-fill. Combined attacks applied in listed order. Where manuscript omits numerical conventions these are implementation assumptions, not claimed identical to its unpublished code. Additional WAVES-style tests are not included because severity levels were unspecified.
- Train-only calibration: for Luo, 28 alpha pairs (.03–.06 and .002–.008) minimize the paper objective 1/PSNR + 1/SSIM + 1−mean NC on a common seven-attack subset. Other paper baselines are swept for maximum NC with mean PSNR ≥43 dB where feasible (fallback highest PSNR explicitly indicated in calibration.csv). Proposed parameters remain frozen. The objective and search ranges are visible in benchmark.py.
- Primary comparison: proposed, three actual paper-derived algorithms. All use the same hosts, watermarks and attacks. No ground-truth attack type or angle is supplied to a decoder. NC of grayscale watermark is measured raw before clipping; clipped NC and RMSE also saved. Four-band methods use average of all four NCs and export every band individually. No oracle selection of best band. ASR is exact tag match; unsigned methods have blank ASR. Diagnostic NC is computed even for rejected images, but CLI returns watermark only after valid HMAC. `authorized_nc` is defined only for authenticated cases.
- Authentication payload 128 bits, 255 BCH carriers. BCH(255,131), three padding zeros, designed distance 37, t=18. The verifier processes **every** hypothesis −5°:0.5°:+5°, including clean images. No attack-specific decoder or early-exit search. Exact HMAC equality only. Large rotations, crop-resize, occlusion and strong noise may fail. No universal 100% claim.
- Main benchmark includes 12×2×3×46×4 = 13,248 rows across both splits; test only 7,728 rows. Optional ablations separately produce 23,184 rows across both splits. Each method/size/host is retained, even if authentication fails. Any execution error stops the run visibly rather than silently discarding it.

## Proposed and ablations

`core.py` preserves DOCX YCbCr → one-level integer Haar IWT → LL real Schur → SVD → partition S11 → Arnold (5 iterations) and additive watermark → second SVD → inverse SVD/Schur → inverse IWT → RGB. Extraction uses the original non-blind saved matrices, actual Schur, and inverse Arnold. `run.py` adds the two-level orthonormal Haar DWT, 4x4 blocks, eligible-block filter (interior, gap≥40, radius≥.25, rho≤.25), 40% low-cost pool enlarged to capacity if necessary, descending standard deviation, fixed permutation, Givens update, margin max(.03,40/(sigma1*r)), BCH and HMAC. `proposed.py` exposes authenticated extraction, and `cli.py` allows file-level embedding and extraction. No original payload or ground-truth codeword is given to the decoder. A public reproducibility key is used only in benchmark/validation. The deployment CLI requires an explicit secret key of at least 16 bytes; use a securely random 32-byte key, never the benchmark key.

Ablations: unsigned DOCX; corrected 8-bit Luo-style OAM on DOCX (adaptation, not paper baseline); no cost filter (other eligibility conditions retained); fixed rho=.03; no sync (same embedded image, only angle zero); no ECC (128 physical carriers); direct coefficient update instead of Givens with same selected positions, target coefficients and bit sequence. Ablations are not independent literature baselines. Parameters and metadata format are documented in code.

## Embed/extract own image

```bash
export WATERMARK_KEY_HEX=$(python -c 'import secrets; print(secrets.token_hex(32))')
python cli.py embed --image host.bmp --watermark wm.png --size 64 --record asset-001 --metadata asset.npz --output signed.png
python cli.py extract --image signed.png --metadata asset.npz --output recovered.png
```

Preserve the secret key and metadata. Do not publish the key. The NPZ file contains auxiliary matrices and ordered carrier positions, not the HMAC tag or key. Extraction returns exit status 2 on authentication rejection. The HMAC verifies the metadata/record binding; this alone does not prove content integrity or prevent all transplant/ownership attacks.

## Results and validation status

- `results/validation.log`: completed real-image checks on test host athens (64x64 logo); BCH 100/100 with 18 flipped bits; clean proposed recovery and wrong-key/record/unwatermarked rejection; all three baseline recovery paths; matched direct/Givens positions; all 46 attack operators output correct RGB shape/dtype.
- `results/preliminary_smoke/`: 150 completed rows from the initial integrated smoke run on airplane at all three sizes (10 methods before adding the matched direct ablation, 5 conditions each). This is partial development data, NOT a complete held-out comparison and not from calibrated settings. The original run hit an execution time limit on later hosts. No unexecuted results are filled in.
- The full new 3-paper-baseline, 45-attack, 3-size comparison is provided as runnable source and is NOT claimed complete in this delivery. Generate `all_results.csv`, `summary.csv`, `summary_by_size.csv`, `summary_by_attack.csv`, `summary_by_host.csv` from the commands above. The runner includes separate calibration CSV and config. Do not mix with old comparison CSVs where Luo was only a DOCX layer adaptation.
