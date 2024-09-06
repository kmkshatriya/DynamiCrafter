import os
import sys
import argparse
import time
import torch
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

# Function to download the model from Hugging Face
def download_model():
    REPO_ID = 'Doubiiu/DynamiCrafter'
    filename_list = ['model.ckpt']
    if not os.path.exists('./checkpoints/dynamicrafter_256_v1/'):
        os.makedirs('./checkpoints/dynamicrafter_256_v1/')
    for filename in filename_list:
        local_file = os.path.join('./checkpoints/dynamicrafter_256_v1/', filename)
        if not os.path.exists(local_file):
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/dynamicrafter_256_v1/', force_download=True)

# Function to run inference and generate the video
def infer(image_path, prompt, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123, resolution=256):
    download_model()
    ckpt_path = 'checkpoints/dynamicrafter_256_v1/model.ckpt'
    config_file = 'configs/inference_256_v1.0.yaml'
    config = OmegaConf.load(config_file)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    model = model.cuda()
    save_fps = 8

    seed_everything(seed)
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
    ])
    torch.cuda.empty_cache()
    print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    
    # Ensure steps don't exceed the limit
    if steps > 60:
        steps = 60 

    batch_size = 1
    channels = model.model.diffusion_model.out_channels
    frames = model.temporal_length
    h, w = resolution // 8, resolution // 8
    noise_shape = [batch_size, channels, frames, h, w]

    # Load image and convert it into a tensor
    img_tensor = torchvision.io.read_image(image_path).float().to(model.device)  # Assuming 3xHxW image
    img_tensor = (img_tensor / 255. - 0.5) * 2  # Normalize

    image_tensor_resized = transform(img_tensor)  # Resize and crop
    videos = image_tensor_resized.unsqueeze(0)  # Add batch dimension

    # Get latent representation of the image
    z = get_latent_z(model, videos.unsqueeze(2))  # Add temporal dimension

    img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

    # Get text embedding for the prompt
    text_emb = model.get_learned_conditioning([prompt])

    # Image conditioning
    cond_images = model.embedder(img_tensor.unsqueeze(0))  # Embedding for image
    img_emb = model.image_proj_model(cond_images)

    # Concatenate the image and text embeddings
    imtext_cond = torch.cat([text_emb, img_emb], dim=1)

    fs = torch.tensor([fs], dtype=torch.long, device=model.device)
    cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}

    # Run inference
    batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)

    # Save the generated video
    video_path = './output.mp4'
    save_videos(batch_samples, './', filenames=['output'], fps=save_fps)
    model = model.cpu()
    return video_path

# Main function to parse input arguments and run inference
def main():
    parser = argparse.ArgumentParser(description="Image to Video Animation using DynamiCrafter")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt describing the animation')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the output video')
    parser.add_argument('--steps', type=int, default=50, help='Number of sampling steps (max 60)')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale (default: 7.5)')
    parser.add_argument('--eta', type=float, default=1.0, help='ETA for DDIM sampling (default: 1.0)')
    parser.add_argument('--fs', type=int, default=3, help='Motion magnitude (default: 3)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed (default: 123)')
    
    args = parser.parse_args()

    # Call the inference function
    video_path = infer(args.image, args.prompt, args.steps, args.cfg_scale, args.eta, args.fs, args.seed, args.resolution)
    print(f"Video generated: {video_path}")

if __name__ == "__main__":
    main()
