# LongConsist

LongConsist is a subject-consistent I2V generation framework built on LongCat-Video, enabling both standard and extended video generation with identity preservation.

## Hardware Requirements

- **Used**: 1*H200 141GB
- **Minimum**: 1*A100 40GB

## Installation

First, navigate to the LongConsist directory:

```bash
cd LongConsist
```

Then install dependencies:

```bash
pip install -e .
```

This will install all required packages including PyTorch, transformers, diffusers, and other dependencies.

## Download LoRA Weights

Before running the generation scripts, you need to download the LoRA weights. Run the following command:

```bash
python -c "from modelscope.hub.file_download import model_file_download; model_file_download('wwwwww345/LongConsist-Lora', 'lora.safetensors', local_dir='models/train/lora')"
```

## Usage

### Subject-Consistent Video Generation

Generate videos with subject consistency from reference images:

```bash
bash scripts/i2v_subject_consistency.sh
```

This script performs standard subject-consistent image-to-video generation, preserving the identity and characteristics of the subject from the input image.

### Subject-Consistent Long Video Generation

Generate extended videos with temporal continuity:

```bash
bash scripts/i2v_subject_consistency_long.sh
```

This script leverages LongCat-Video's autoregressive continuation capability to generate longer videos while maintaining subject consistency across segments. The long video generation process uses overlapping frames between segments to ensure smooth transitions and temporal coherence.

## Configuration

Both scripts support various parameters for customization:

- `--id_images`: Path(s) to reference image(s) for subject consistency
- `--prompt`: Text prompt describing the desired video content
- `--num_frames`: Number of frames to generate (standard mode)
- `--frames_per_segment` and `--total_segments`: Control long video generation
- `--height` and `--width`: Output video resolution
- `--cfg_scale`: Classifier-free guidance scale
- `--num_inference_steps`: Number of diffusion steps

Modify the shell scripts to adjust these parameters according to your requirements.

## Notes

- The long video generation mode uses an autoregressive continuation mechanism, generating videos segment by segment with overlapping frames to maintain temporal consistency.
- Model checkpoints will be automatically downloaded from ModelScope on first run.

## Acknowledgement

We thank the **DiffSynth-Studio** team for providing the base code for model fine-tuning and the **LongCat-Video** team for the base model.
