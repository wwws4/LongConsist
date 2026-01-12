import torch, torchvision
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import argparse
import os

class ImageCropAndResize():
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


def parse_args():
    parser = argparse.ArgumentParser(description='Generate consistent video from image')
    parser.add_argument('--id_images', type=str, nargs='+', default=["test_data/old_woman.jpg"],
                        help='Paths to reference images')
    parser.add_argument('--prompt', type=str, 
                        default="An elderly woman standing in front of a volcano, holding a microphone as she explains the principles of volcanic eruptions. The scene is set outdoors, with the towering volcano in the background emitting light smoke, adding a sense of realism to the environment. The woman is dressed in outdoor clothing suitable for fieldwork, and her posture and expression suggest she is confidently delivering an educational presentation. The lighting reflects natural daylight, highlighting both the volcanic landscape and the woman as she speaks.",
                        help='Text prompt for video generation')
    parser.add_argument('--negative_prompt', type=str,
                        default="static, motionless, still image, frozen, no movement, blurry, low quality, worst quality, jpeg artifacts, deformed, bad anatomy, disfigured, poorly drawn hands, poorly drawn face, extra limbs, missing limbs, fused fingers, too many fingers, text, watermark, signature, messy background, The subject in the input image and the generated video are inconsistent",
                        help='Negative prompt')
    parser.add_argument('--lora_path', type=str, default="models/train/lora/lora.safetensors",
                        help='Path to LoRA weights')
    parser.add_argument('--lora_alpha', type=float, default=1.0,
                        help='LoRA alpha value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_frames', type=int, default=81,
                        help='Number of frames to generate')
    parser.add_argument('--height', type=int, default=480,
                        help='Video height')
    parser.add_argument('--width', type=int, default=832,
                        help='Video width')
    parser.add_argument('--cfg_scale', type=float, default=5.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--sigma_shift', type=float, default=1.0,
                        help='Sigma shift value')
    parser.add_argument('--num_inference_steps', type=int, default=200,
                        help='Number of inference steps')
    parser.add_argument('--output_path', type=str, default="output/video.mp4",
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=24,
                        help='Output video FPS')
    parser.add_argument('--quality', type=int, default=10,
                        help='Output video quality')
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("LongCat Video Generation")
    print("=" * 60)
    
    print(f"\nLoading reference images: {args.id_images}")
    id_images = [Image.open(path).convert("RGB") for path in args.id_images]

    imagecropandresize = ImageCropAndResize(args.height, args.width, 1280*720, 16, 16)
    id_images = [imagecropandresize(id_image) for id_image in id_images]
    
    print(f"Successfully loaded {len(id_images)} reference image(s)")
    
    print("\nLoading model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="meituan-longcat/LongCat-Video", origin_file_pattern="dit/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    
    if os.path.exists(args.lora_path):
        print(f"\nLoading LoRA: {args.lora_path}")
        pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
    else:
        print(f"\nWarning: LoRA path {args.lora_path} not found, skipping LoRA loading")
    
    pipe.enable_vram_management()
    print("Model loading completed!")

    print("\n" + "-" * 40)
    print(f"Generating video with {len(id_images)} reference image(s)!")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        id_images=id_images,
        seed=args.seed,
        tiled=True,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
    )
    
    save_video(video, args.output_path, fps=args.fps, quality=args.quality)
    
    print("\n" + "=" * 60)
    print(f"Video saved to: {args.output_path}")
    print("=" * 60)
    
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()