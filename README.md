# ONN Online Training

Code and configuration for online training experiments with multi-channel optical neural networks.

## Contents

- `config_loader.py` loads YAML experiment configs, applies runtime overrides, resolves random seeds, and writes reproducible config snapshots.
- `dataloader_patch.py` builds datasets and dataloaders for image, patchified, face-keypoint, notMNIST, VLM captioning, and synthetic experiments.
- `exp_model_v1.py` and `exp_model_v2.py` provide physical optical experiment helpers for SLM mask generation, camera acquisition, and live display.
- `fine_tunning.py` buffers optical-layer measurements and periodically fine-tunes the surrogate model during ONN training.
- `models.py` defines the surrogate optical U-Net and related residual, attention, upsampling, and correction modules.
- `optical_bridge.py` connects the physical optical forward pass with surrogate-based gradients through a custom autograd bridge.
- `phase_system.py` manages trainable optical phase masks, including initialization, normalization, smoothing, and conversion to valid phase units.
- `readout.py` provides ROI/tile mask generation, cached readout masks, tile scoring, scaling, and energy-margin losses.
- `surrogate_model_training.py` trains the surrogate optical model against physical optical measurements.
- `training_viz.py` and `visualization_utils.py` provide notebook-oriented live visualization utilities.

## Training Entry Points

- `optical_onn_training.py`: baseline multi-layer optical classification without channel mixing.
- `optical_onn_training_mix.py`: mixed-channel optical classification with learnable/digital channel mixing.
- `optical_onn_training_binary_decision.py`: binary-decision/ECOC optical classification.
- `optical_onn_training_mix_face_linear.py`: mixed-channel facial keypoint regression.
- `optical_onn_training_mix_VLM.py`: optical encoder plus transformer-style decoder for captioning.

## Result Scripts

- `result.py`: baseline ONN plots and diagnostics.
- `result_binary_decision.py`: binary-decision/ECOC metrics and diagnostics.
- `result_mix.py`: mixed-channel classification curves.
- `result_mix_face.py`: facial keypoint regression metrics.
- `result_mix_VLM.py`: captioning metrics and generated-caption history.

## Configurations

- `config.yaml`: default supervised forward-forward style ONN configuration.
- `config_binary_decision.yaml`: binary-decision/ECOC configuration.
- `config_mix.yaml`: mixed-channel classification configuration.
- `config_mix_face.yaml`: facial keypoint regression configuration.
- `config_mix_VLM.yaml`: VLM/captioning configuration.

## Notebooks

- `Calibration.ipynb`: hardware initialization and calibration workflow.
- `onn_online_training.ipynb`: main notebook for launching surrogate and ONN training runs.
- `result.ipynb`: interactive result plotting notebook.

## Large Artifacts

The `pre_trained_model_save/` directory contains checkpoints and experiment outputs. It is ignored by Git by default because it is large; use Git LFS, GitHub Releases, or an external storage location if these artifacts need to be shared.

