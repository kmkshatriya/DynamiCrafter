import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import time
from omegaconf import OmegaConf
import torch
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling, get_latent_z
from utils.utils import instantiate_from_config
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything


class Image2Video():
    def __init__(self, resolution='320_512', is_interp=False, gpu_num=1) -> None:
        self.suffix=""
        self.is_interp=is_interp
        if is_interp:
            self.suffix="_Interp"
        
        self.resolution = (int(resolution.split('_')[0]), int(resolution.split('_')[1])) #hw

        self.download_model()
        ckpt_path='checkpoints/dynamicrafter_'+resolution.split('_')[1]+self.suffix+'_v1/model.ckpt'
        # ckpt_path='checkpoints/dynamicrafter_512_interp_fp16_pruned.safetensors'
        # ckpt_path='checkpoints/dynamicrafter_512_fp16_pruned.safetensors'        
        # ckpt_path='checkpoints/dynamicrafter_1024_fp16_pruned.safetensors'  

        config_file='configs/inference_'+resolution.split('_')[1]+'_v1.0.yaml'
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint']=False   
        model_list = []
        for gpu_id in range(gpu_num):
            model = instantiate_from_config(model_config)
            # model = model.cuda(gpu_id)
            assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
            model = load_model_checkpoint(model, ckpt_path)
            model.eval()
            model_list.append(model)
        self.model_list = model_list
        self.save_fps = 8

    def get_image(self, image_path, prompt, result, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123):
        seed_everything(seed)
        transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
            ])
        torch.cuda.empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        start = time.time()
        gpu_id=0
        if steps > 60:
            steps = 60 
        model = self.model_list[gpu_id]
        model = model.cuda()
        batch_size=1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # text cond
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_emb = model.get_learned_conditioning([prompt])

            # img cond
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
            img_tensor = (img_tensor / 255. - 0.5) * 2

            image_tensor_resized = transform(img_tensor) #3,h,w
            videos = image_tensor_resized.unsqueeze(0) # bchw
            
            z = get_latent_z(model, videos.unsqueeze(2)) #bc,1,hw
            
            img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

            if self.is_interp:
                img_tensor_repeat = torch.zeros_like(img_tensor_repeat)
                img_tensor_repeat[:,:,:1,:,:] = z
                img_tensor_repeat[:,:,-1:,:,:] = z

            cond_images = model.embedder(img_tensor.unsqueeze(0)) ## blc
            img_emb = model.image_proj_model(cond_images)

            imtext_cond = torch.cat([text_emb, img_emb], dim=1)

            fs = torch.tensor([fs], dtype=torch.long, device=model.device)
            cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
            
            ## inference
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)
            ## b,samples,c,t,h,w
            if self.is_interp:
                batch_samples = batch_samples[:,:,:,:-1,...]

            out_vid_nm = Path(result).stem
            result_dir = Path(result).parent

        save_videos(batch_samples, result_dir, filenames=[out_vid_nm], fps=self.save_fps)
        print(f"Saved as {out_vid_nm}. Time used: {(time.time() - start):.2f} seconds")
        model = model.cpu()
        return os.path.join(result_dir, out_vid_nm)
    
    def download_model(self):
        REPO_ID = 'Doubiiu/DynamiCrafter_'+str(self.resolution[1])+self.suffix if self.resolution[1]!=256 else 'Doubiiu/DynamiCrafter'
        filename_list = ['model.ckpt']
        if not os.path.exists('./checkpoints/dynamicrafter_'+str(self.resolution[1])+self.suffix+'_v1/'):
            os.makedirs('./checkpoints/dynamicrafter_'+str(self.resolution[1])+self.suffix+'_v1/')
        for filename in filename_list:
            local_file = os.path.join('./checkpoints/dynamicrafter_'+str(self.resolution[1])+self.suffix+'_v1/', filename)
            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/dynamicrafter_'+str(self.resolution[1])+self.suffix+'_v1/', local_dir_use_symlinks=False)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="prompts/512/girl08.png", help="Path to input image")
    parser.add_argument("--prompt", type=str, default="a woman looking out in the rain", help="Text prompt for the video")
    parser.add_argument("--result", type=str, default="results/video.mp4", help="Path to output video")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument('--interp', action='store_true', help="Enable interpolation or not")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    i2v = Image2Video(f"{args.height}_{args.width}", args.interp )
    video_path = i2v.get_image(args.image, args.prompt, args.result)
    print('done', video_path)