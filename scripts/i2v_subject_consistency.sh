export CUDA_VISIBLE_DEVICES=0

python inference/i2v_subject_consistency.py \
  --id_images "test_data/old_woman.jpg" \
  --prompt "An elderly man standing in front of a volcano, holding a microphone as he explains the principles of volcanic eruptions. The scene is set outdoors, with the towering volcano in the background emitting light smoke, adding a sense of realism to the environment." \
  --negative_prompt "static, motionless, still image, frozen, no movement, blurry, low quality, worst quality, jpeg artifacts, deformed, bad anatomy, disfigured, poorly drawn hands, poorly drawn face, extra limbs, missing limbs, fused fingers, too many fingers, text, watermark, signature, messy background, The subject in the input image and the generated video are inconsistent" \
  --lora_path "models/train/lora/lora.safetensors" \
  --lora_alpha 1.0 \
  --seed 42 \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --cfg_scale 5.0 \
  --sigma_shift 1.0 \
  --num_inference_steps 200 \
  --output_path "output/video.mp4" \
  --fps 24 \
  --quality 10