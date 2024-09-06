import os
import sys
import argparse
import time
import torch
import torchvision
from pathlib import Path
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from pytorch_lightning import seed_everything
from einops import repeat
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
sys.path.insert(0, "scripts/evaluation")
from funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    get_latent_z,
    save_videos
)


# Function to run inference and generate the video
def infer(image1, prompt, result, width=256, height=256, steps=50, cfg_scale=7.5, eta=1.0, fs=0, seed=123, interp=False, image2=None, ckpt_dir="checkpoints"):
    if not fs:
        if width==256:
            fs=3
        elif width==512:
            fs=24         
        elif width==1024:
            fs=10

    suffix=""
    if interp:
        suffix="_Interp"
        height=320
        width=512        
    # Load the model
    ckpt_path = f'{ckpt_dir}/dynamicrafter_{width}{suffix}_v1/model.ckpt'
    config_file = f'configs/inference_{width}_v1.0.yaml'
    config = OmegaConf.load(config_file)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    model = model.cuda()

    if width==256:
        width = 256 if image1.width==imag21.height else 512
        height=256 if image1.width==imag21.height else 320

    # Set the frames per second (FPS) for the output video
    save_fps = 8

    # Set the seed for reproducibility
    seed_everything(seed)

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.CenterCrop((height, width)),
    ])

    torch.cuda.empty_cache()
    print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # Ensure the number of steps does not exceed 60
    if steps > 60:
        steps = 60 

    # Model parameters
    batch_size = 1
    channels = model.model.diffusion_model.out_channels
    frames = model.temporal_length
    h, w = height // 8, width // 8
    noise_shape = [batch_size, channels, frames, h, w]

    # Load and preprocess the image
    img_tensor = torchvision.io.read_image(image1).float().to(model.device)  # Assuming 3xHxW image
    img_tensor = (img_tensor / 255. - 0.5) * 2  # Normalize to [-1, 1]
   
    if img_tensor.size(0) != 3:
        img_tensor = img_tensor[:3, :, :]

    # Resize and crop the image
    image_tensor_resized = transform(img_tensor)  # Resize and crop to the desired resolution
    videos = image_tensor_resized.unsqueeze(0)  # Add batch dimension

    # Get the latent representation of the image
    z = get_latent_z(model, videos.unsqueeze(2))  # Add temporal dimension

    if image2 is not None:
        img_tensor2 = torchvision.io.read_image(image2).float().to(model.device)  # Assuming 3xHxW image
        img_tensor2 = (img_tensor2 / 255. - 0.5) * 2

        if img_tensor2.size(0) != 3:
            img_tensor2 = img_tensor2[:3, :, :]

        image_tensor_resized2 = transform(img_tensor2) #3,h,w
        videos2 = image_tensor_resized2.unsqueeze(0) # bchw
        
        z2 = get_latent_z(model, videos2.unsqueeze(2)) #bc,1,hw
        
    # If interpolation is enabled
    if interp:
        img_tensor_repeat = torch.zeros_like(repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames))
        img_tensor_repeat[:, :, :1, :, :] = z  # Set the first frame
        if image2 is not None:
            img_tensor_repeat[:, :, -1:, :, :] = z2
        else:
            img_tensor_repeat[:, :, -1:, :, :] = z  # Set the last frame
    else:
        img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

    # Get text embedding for the prompt
    text_emb = model.get_learned_conditioning([prompt])

    # Image conditioning
    cond_images = model.embedder(img_tensor.unsqueeze(0))  # Get image embedding
    img_emb = model.image_proj_model(cond_images)

    # Concatenate the image and text embeddings
    imtext_cond = torch.cat([text_emb, img_emb], dim=1)

    fs = torch.tensor([fs], dtype=torch.long, device=model.device)
    cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}

    # Run inference
    batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)

    # Save the generated video
    out_vid_nm = Path(result).stem
    result_dir = Path(result).parent
    save_videos(batch_samples, result_dir, filenames=[out_vid_nm], fps=save_fps)

    model = model.cpu()  # Move model back to CPU to free GPU memory
    return result


# Main function to parse input arguments and run inference
def main():
    parser = argparse.ArgumentParser(description="Image to Video Animation using DynamiCrafter")
    parser.add_argument('--image', type=str, required=True, help='Path to the start image')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt describing the animation')
    parser.add_argument("--result", type=str, default="results/video.mp4", help="Path to output video")
    parser.add_argument('--width', type=int, default=512, help='Width of the output video (default: 512)')
    parser.add_argument('--height', type=int, default=512, help='Height of the output video (default: 512)')
    parser.add_argument('--steps', type=int, default=50, help='Number of sampling steps (max 60)')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale (default: 7.5)')
    parser.add_argument('--eta', type=float, default=1.0, help='ETA for DDIM sampling (default: 1.0)')
    parser.add_argument('--fs', type=int, default=0, help='Motion magnitude (default: 3)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed (default: 123)')
    parser.add_argument('--interp', action='store_true', help="Enable interpolation between frames")
    parser.add_argument('--image2', type=str, default=None, help='Path to the end image')
    parser.add_argument("--model", type=str, default="checkpoints", help="Path to checkpoints folder")

    args = parser.parse_args()

    # Call the inference function
    video_path = infer(args.image, args.prompt, args.result, args.width, args.height, args.steps, args.cfg_scale, args.eta, args.fs, args.seed, args.interp, args.image2, args.model)

    print(f"Video generated: {video_path}")

if __name__ == "__main__":
    main()
