import pytorch_lightning as pl
from config import Systemcfg
from models.utils import *

from data_utils import dataset_dict
from models import model_dict

from torch.utils.data import DataLoader
from pdb import set_trace as st

from torchvision.utils import save_image


class LitSystem(pl.LightningModule):
    def __init__(self, cfg: Systemcfg):
        super(LitSystem, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = model_dict[self.cfg.model.version](
            cfg=cfg,
            dim=self.cfg.model.dim_in, 
            ff_ratio=cfg.model.ff_ratio,
            dropout=cfg.model.dropout,
            L_bands=cfg.ray_param.L_bands)
        self.loss = torch.nn.MSELoss()
    
    def forward(self, batch_rays):
        output = self.model(batch_rays)
        return output

    def setup(self, stage):
        dataset = dataset_dict[self.cfg.expt_settings.dataset]
        self.train_dataset = dataset(data_dir=self.cfg.dataset.data_dir, split='train', white_bkgd=self.cfg.train.white_bkgd, batch_type=self.cfg.train.batch_type, factor=self.cfg.dataset.factor)
        self.val_dataset = dataset(data_dir=self.cfg.dataset.data_dir, split='val', white_bkgd=self.cfg.val.white_bkgd, batch_type=self.cfg.val.batch_type, factor=self.cfg.dataset.factor)
        self.test_dataset = dataset(data_dir=self.cfg.dataset.data_dir, split='test', white_bkgd=self.cfg.val.white_bkgd, batch_type=self.cfg.val.batch_type, factor=self.cfg.dataset.factor)


    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, shuffle=True, num_workers=self.cfg.train.num_workers, batch_size=self.cfg.train.batch_size, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, shuffle=False, num_workers=self.cfg.val.num_workers, batch_size=self.cfg.val.im_batch_size, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, shuffle=False, num_workers=self.cfg.val.num_workers, batch_size=self.cfg.val.im_batch_size, pin_memory=True)
        return test_dl

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr_init)
        scheduler = MipLRDecay(optimizer=optimizer, 
            lr_init=self.cfg.optimizer.lr_init,
            lr_final=self.cfg.optimizer.lr_final,
            max_steps=self.cfg.optimizer.max_steps,
            lr_delay_steps=self.cfg.optimizer.lr_delay_steps,
            lr_delay_mult=self.cfg.optimizer.lr_delay_mult)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


    def training_step(self, batch, batch_nb):
        rays, rgbs = batch
        output_coarse, output_fine, attn_weights,  = self(rays)
        loss_coarse = self.loss(output_coarse, rgbs)
        loss_fine = self.loss(output_fine, rgbs)

        loss =  self.cfg.optimizer.loss_coarse * loss_coarse + self.cfg.optimizer.loss_fine * loss_fine


        with torch.no_grad():
            psnr_fine = calc_psnr(output_fine, rgbs)
            psnr_coarse = calc_psnr(output_coarse, rgbs)

        
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])
        self.log("train/loss", loss)
        self.log("train/psnr_fine", psnr_fine, prog_bar=True)
        self.log("train/psnr_coarse", psnr_coarse, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        _, rgbs = batch
        rgb_gt = rgbs[..., :3]
        coarse_rgb, fine_rgb = self.render_image(batch)
        with torch.no_grad():
            loss_coarse = self.loss(coarse_rgb, rgb_gt)
            loss_fine = self.loss(fine_rgb,rgb_gt )
            val_loss = self.cfg.optimizer.loss_coarse  * loss_coarse + self.cfg.optimizer.loss_fine * loss_fine
            val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)

        log = {'val/loss': val_loss, 'val/psnr': val_psnr_fine}
        stack = stack_rgb(rgb_gt, coarse_rgb, fine_rgb)  # (3, 3, H, W)
        if batch_nb == 0:
            self.logger.experiment.add_images('val/GT_coarse_fine',
                                            stack, self.current_epoch)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


    def render_image(self, batch):
        rays, rgbs = batch
        _, height, width, _ = rgbs.shape  # N H W C
        single_image_rays, val_mask = rearrange_render_image(rays, self.cfg.val.batch_size)
        coarse_rgb, fine_rgb = [], []
        vol_coarse, vol_fine = [], []
        with torch.no_grad():
            for batch_rays in single_image_rays:

                output_coarse, output_fine, weights = self(batch_rays)

                coarse_rgb.append(output_coarse)
                fine_rgb.append(output_fine)


        coarse_rgb = torch.cat(coarse_rgb, dim=0)
        fine_rgb = torch.cat(fine_rgb, dim=0)


        coarse_rgb = coarse_rgb.reshape(1, height, width, coarse_rgb.shape[-1])  # N H W C
        fine_rgb = fine_rgb.reshape(1, height, width, fine_rgb.shape[-1])  # N H W C

        return coarse_rgb, fine_rgb


    def test_step(self, batch, batch_nb):
        _, rgbs = batch
        rgb_gt = rgbs[..., :3]
        coarse_rgb, fine_rgb = self.render_image(batch)
        with torch.no_grad():
            test_loss =  self.cfg.optimizer.loss_coarse * self.loss(coarse_rgb, rgb_gt) + self.cfg.optimizer.loss_fine* self.loss(fine_rgb,rgb_gt )
            test_psnr_fine = calc_psnr(fine_rgb, rgb_gt)

        log = {'test/loss': test_loss, 'test/psnr': test_psnr_fine}

        output_image = fine_rgb[0].permute(2,0,1)
        im_name = 'out_'+str(batch_nb).zfill(3)+'.png'
        save_image(output_image, im_name)

        return log

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([x['test/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['test/psnr'] for x in outputs]).mean()

        self.log('test/loss', mean_loss)
        self.log('test/psnr', mean_psnr, prog_bar=True)