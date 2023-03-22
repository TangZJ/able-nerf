import hydra
from omegaconf import DictConfig, OmegaConf
from pdb import set_trace as st
from setuptools import setup
import torch
import numpy as np
import random
import os
from config import Systemcfg
from hydra.core.config_store import ConfigStore

from torch.utils.data import DataLoader


from tqdm import tqdm

from data_utils import dataset_dict
from models.vis import save_images, save_rgb_images

from data_utils.datasets import Rays_keys, Rays
from models.utils import rearrange_render_image
from models.metrics import eval_errors

from ablenerf_litsystem import LitSystem



cs = ConfigStore.instance()
cs.store(name='systemconfig', node=Systemcfg )


@hydra.main(version_base='1.1', config_path="conf", config_name="defaults")
def main(cfg: Systemcfg) -> None:
    main_dir = hydra.utils.get_original_cwd()
    ckpt_path = os.path.join(main_dir, cfg.checkpoint.resume_path)
    exp_name = cfg.expt_settings.test_name
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = LitSystem(cfg).load_from_checkpoint(ckpt_path).to(device).eval()

    dataset = dataset_dict[cfg.expt_settings.dataset]
    test_dataset = dataset(data_dir=cfg.dataset.data_dir,split='test',white_bkgd=cfg.val.white_bkgd,batch_type=cfg.val.batch_type, factor=cfg.dataset.factor) 

    test_loader = DataLoader(test_dataset, 
        shuffle=False, 
        num_workers=cfg.val.num_workers, 
        batch_size=cfg.val.im_batch_size, 
        pin_memory=True)

    save_path = os.path.join(main_dir, 'test', exp_name)
    os.makedirs(save_path, exist_ok= True)

    psnr_values = []
    ssim_values = []
    n = -1
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            n += 1
            rays, rgbs = batch
            rays = Rays(*[getattr(rays, name).to(device) for name in Rays_keys])
            rgbs = rgbs.to(device)
            _, height, width, _ = rgbs.shape
            single_image_rays, val_mask = rearrange_render_image(rays, cfg.val.batch_size)
            distances, accs = [], []
            coarse_rgb, fine_rgb = [], []
            with torch.no_grad():
                for batch_rays in single_image_rays:
                    output_coarse, output_fine, weights = model(batch_rays)
                    fine_rgb.append(output_fine)
            
            fine_rgb = torch.cat(fine_rgb, dim=0)
            fine_rgb = fine_rgb.reshape(1, height, width, fine_rgb.shape[-1])  # N H W C
            psnr_val, ssim_val = eval_errors(fine_rgb, rgbs)
            psnr_values.append(psnr_val.cpu().item())
            ssim_values.append(ssim_val.cpu().item())

            save_rgb_images(fine_rgb, save_path, idx)

        with open(os.path.join(save_path, 'psnrs.txt'), 'w') as f:
            f.write(' '.join([str(v) for v in psnr_values]))
        with open(os.path.join(save_path, 'ssim.txt'), 'w') as f:
            f.write(' '.join([str(v) for v in ssim_values]))         
    return 

if __name__ == '__main__':

    main()
