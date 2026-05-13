# FrameSkip: Learning from Fewer but More Informative Frames in VLA Training

[![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/ZGC-EmbodyAI/FrameSkip)
[![Hugging Face](https://img.shields.io/badge/Checkpoints-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/collections/VLyb/frameskip)

**FrameSkip** is a training-time frame selection framework for Vision-Language-Action (VLA) models. Instead of treating every frame in a dense robot demonstration trajectory as equally useful supervision, FrameSkip scores trajectory frames with lightweight cues and trains primarily from fewer but more informative frames.

FrameSkip is designed as a data-layer intervention: it changes which frames are exposed during training while leaving the VLA architecture, action head, loss function, and inference procedure unchanged.

## Highlights

- **Frame-level supervision allocation** for VLA training.
- **Architecture-agnostic dataloader integration** with no change to model inference.
- **Importance-guided frame retention** using action variation, visual-action coherence, task-progress priors, and gripper-transition preservation.
- **Released checkpoints** on [Hugging Face](https://huggingface.co/collections/VLyb/frameskip).

## Checkpoints

### Download Checkpoints

Model checkpoints are hosted in the Hugging Face collection:

**[VLyb/frameskip](https://huggingface.co/collections/VLyb/frameskip)**

You can download checkpoints with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download <checkpoint-repo-name> --local-dir checkpoints/<checkpoint-name>
```

Replace `<checkpoint-repo-name>` with the checkpoint repository listed in the collection.

### Load and Use Checkpoints

FrameSkip is built on the [starVLA](https://github.com/starVLA/starVLA) training and evaluation stack. The released checkpoints follow the standard starVLA checkpoint format and can be loaded in the same way as starVLA VLA policies.

For simulation evaluation, please refer to the model loading and evaluation workflow of the QwenGR00T architecture in starVLA, and replace the checkpoint path with the downloaded FrameSkip checkpoint.

## Quick Start

The code and model checkpoints have been released. The current FrameSkip implementation is located under `./starVLA/starVLA/frameskip`. A cleaner and more user-friendly version is being organized; at this stage, **we recommend using AI-assisted code reading to navigate the implementation details.**

> The ./starVLA directory is a full copy of the starVLA project, with additional implementations for FrameSkip-related functionality.

## Method Overview

FrameSkip follows a three-stage pipeline:

1. **Score frames** in each demonstration trajectory with trajectory-level cues.
2. **Retain high-importance frames** under a target retention ratio.
3. **Remap dataloader queries** to the retained frame indices during VLA training.

The policy is trained with the standard VLA objective, and inference is unchanged.

## Resources

- Code: [https://github.com/ZGC-EmbodyAI/FrameSkip](https://github.com/ZGC-EmbodyAI/FrameSkip)
- Checkpoints: [https://huggingface.co/collections/VLyb/frameskip](https://huggingface.co/collections/VLyb/frameskip)

## Citation

If you find FrameSkip useful, please cite:

```bibtex
TODO
```
