export CUDA_VISIBLE_DEVICES=0

python inference/i2v_subject_consistency_long.py \
  --id_images "test_data/old_woman.jpg" \
  --prompt "An elderly woman standing in front of a volcano, holding a microphone as she explains the principles of volcanic eruptions. The scene is set outdoors, with the towering volcano in the background emitting light smoke, adding a sense of realism to the environment. The woman is dressed in outdoor clothing suitable for fieldwork, and her posture and expression suggest she is confidently delivering an educational presentation. The lighting reflects natural daylight, highlighting both the volcanic landscape and the woman as she speaks." \
  --negative_prompt "static, motionless, still image, frozen, no movement, blurry, low quality, worst quality, jpeg artifacts, deformed, bad anatomy, disfigured, poorly drawn hands, poorly drawn face, extra limbs, missing limbs, fused fingers, too many fingers, text, watermark, signature, messy background, The subject in the input image and the generated video are inconsistent" \
  --lora_path "models/train/lora/lora.safetensors" \
  --lora_alpha 1.0 \
  --seed 42 \
  --height 480 \
  --width 832 \
  --frames_per_segment 81 \
  --total_segments 3 \
  --overlap_frames 8 \
  --cfg_scale 5.0 \
  --sigma_shift 1.0 \
  --num_inference_steps 200 \
  --num_inference_steps_continue 200 \
  --output_prefix "video_consistent_long" \
  --save_dir "output/consistent_long_video" \
  --fps 24 \
  --quality 10