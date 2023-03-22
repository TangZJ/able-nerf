import torch
from torch import nn
from pytorch_lightning import LightningModule

from models.utils import generate_ipe_ray_samples, resampled_generate_ipe_ray_samples

from pdb import set_trace as st
from models.utils import Embedding

from models.transformers import *

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        #m.bias.data.fill_(0.01)

def linear_to_srgb(linear, eps):
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.max(eps,linear)**(5/12) - 11) / 200
    return torch.where(linear<=0.0031308, srgb0, srgb1)

def get_attn_mask(num_points):

    mask = torch.ones(1, num_points+1, num_points+1).triu(diagonal=1)
    mask[:,0, 1:] = 0     
    return mask


class LETransformer(LightningModule):
    def __init__(self, dim, ff_ratio=2, dropout=0.0, lp_attn_layer=2):
        super(LETransformer, self).__init__()
        num_heads = dim // 64
        self.cross_attn = AttentionEncoder(input_dim=dim, num_heads=num_heads, ff_ratio=ff_ratio, dropout_p=dropout)
        self.self_attn = nn.ModuleList(
            [AttentionEncoder(input_dim=dim, num_heads=num_heads, ff_ratio=ff_ratio, dropout_p=dropout) \
                for _ in range(lp_attn_layer)])
        self.decode = AttentionEncoder(input_dim=dim, num_heads=num_heads, ff_ratio=ff_ratio, dropout_p=dropout)

    def forward(self, tokens, light_probes, class_token):
        
        stored_lighting, _ = self.cross_attn(light_probes, tokens)

        for layer in self.self_attn:
            stored_lighting, weights = layer(stored_lighting, stored_lighting)
        #processed_lighting,_ = self.self_attn(stored_lighting, stored_lighting)
        decoded_spec,_ = self.decode(class_token.unsqueeze(1), stored_lighting)

        return decoded_spec


class ableNeRF(LightningModule):
    
    def __init__(self, cfg, dim=256, ff_ratio=2, dropout=0.0, L_bands=16):
        super(ableNeRF, self).__init__()

        self.cfg = cfg
    
        point_coarse_layers = self.cfg.model.coarse_layers
        point_fine_layers = self.cfg.model.fine_layers
        self.dim = dim
        self.in_channels_xyz = L_bands * 3 * 2
        self.ff_ratio = self.cfg.model.ff_ratio
        self.dropout = dropout
        lp_attn_layer = self.cfg.model.lp_layers
        
        self.num_light_probes = cfg.model.num_lp
        self.masks = get_attn_mask(cfg.ray_param.num_samples)


        if self.num_light_probes != 0:
            self.light_probes = nn.Parameter(torch.randn(1,self.num_light_probes,dim))

        dir_L_bands = 4
        self.fourier_dir_emb = Embedding(3,dir_L_bands)
        self.in_channels_dir = dir_L_bands * 3 * 2 + 3 


        self.color_token = nn.Parameter(torch.randn(1,1, dim))

        self.view_token = nn.Sequential(
            nn.Linear(self.in_channels_dir , dim)
        )
        
        self.dir_query = nn.Sequential(
            nn.Linear(self.in_channels_dir, dim),
        )


        self.point_embedding = nn.Sequential(
            nn.Linear(self.in_channels_xyz, dim),
            nn.ReLU(True),
            nn.Linear(dim,dim),
            nn.ReLU(True),
            nn.Linear(dim,dim),
            nn.ReLU(True),
            nn.Linear(dim,dim), 
        )


        self.coarse = TransformerClassDecoderV2(layers=point_coarse_layers, dim=dim,ff_ratio=ff_ratio,dropout=dropout)
        self.fine = TransformerClassDecoderV2(layers=point_fine_layers, dim=dim,ff_ratio=ff_ratio,dropout=dropout)

        self.rgb_c_diff = nn.Sequential(
            nn.Linear(dim, 3), 
        )

        self.rgb_f_diff = nn.Sequential(
            nn.Linear(dim, 3), 
        )

        self.c_spec_colour = LETransformer(dim=dim, ff_ratio=ff_ratio, dropout=dropout, lp_attn_layer=lp_attn_layer)
        self.c_spec_rgb = nn.Sequential(
            nn.Linear(dim + self.in_channels_dir , dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 3),
            nn.Sigmoid()
        )

        self.f_spec_colour = LETransformer(dim=dim, ff_ratio=ff_ratio, dropout=dropout, lp_attn_layer=lp_attn_layer)
        self.f_spec_rgb = nn.Sequential(
            nn.Linear(dim + self.in_channels_dir, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 3),
            nn.Sigmoid()
        )

        self.sig = nn.Sigmoid()
        self.eps = torch.Tensor([1e-7])

        init_weights(self.color_token)
        init_weights(self.point_embedding)
        init_weights(self.dir_query)
        init_weights(self.rgb_f_diff)
        init_weights(self.rgb_c_diff)
        init_weights(self.c_spec_rgb)
        init_weights(self.f_spec_rgb)

    def forward(self, rays):

        t_vals_coarse, t_coarse_emb, ipe_features = generate_ipe_ray_samples(
                    origins=rays.origins,
                    directions=rays.directions,
                    radii=rays.radii,
                    num_samples_per_ray=self.cfg.ray_param.num_samples,
                    near_bound=rays.near,
                    far_bound=rays.far,
                    bool_randomized=self.cfg.train.randomized,
                    bool_disparity=self.cfg.ray_param.disparity,
                    L_bands=self.cfg.ray_param.L_bands,
                    ray_shape=self.cfg.ray_param.shape,
                    class_token=True
                )
        num_batch, num_points, _ = ipe_features.shape
        attn_masks = self.masks.to(ipe_features.device)
        eps = self.eps.to(ipe_features.device)

        """
        One time generation of colour token for diffuse and direction  for specular
        """        

        dir_emb = self.fourier_dir_emb(rays.directions)

        color_token = self.color_token.expand(num_batch, -1, -1)
        view_token = self.view_token( dir_emb)

        """ Tokenizer for color and cones
        RAY TOKENS = | COLOR TOKEN | CONIC TOKENS |  
        """

        conic_tokens = self.point_embedding(ipe_features) # encodes xyz to dim size
        ray_tokens = torch.cat((color_token, conic_tokens), 1)
        
        """ Add Light Probes
        | COLOR TOKEN | CONIC TOKENS | LIGHT TOKENS |
        """

        light_tokens = self.light_probes.expand(num_batch,-1,-1) 
        
        """
        Transformer Encoding
        """

        ray_tokens, attn_weights = self.coarse(ray_tokens, attn_masks)
        coarse_color_token, conic_tokens = ray_tokens[:,0,:], ray_tokens[:,1:,:]

        rgb_raw_coarse_diff = self.rgb_c_diff(coarse_color_token)
        rgb_coarse_diff = self.sig(rgb_raw_coarse_diff - 1.0986122886681098)

        rgb_coarse_spec_fea = self.c_spec_colour(conic_tokens, light_tokens, view_token)

        rgb_coarse_spec_fea = torch.cat((rgb_coarse_spec_fea, dir_emb.unsqueeze(1)), 2)

        rgb_coarse_spec_rgb = self.c_spec_rgb(rgb_coarse_spec_fea.squeeze(1)) * 0.5
    
        rgb_coarse = torch.clip(linear_to_srgb(rgb_coarse_spec_rgb + rgb_coarse_diff, eps),0.0,1.0)

        coarse_weights = attn_weights[:,0,1:] # 1 because discard self attn token weight        

        t_vals_fine, t_fine_emb, ipe_features= resampled_generate_ipe_ray_samples(
            origins=rays.origins,
            directions=rays.directions,
            radii=rays.radii,
            t_samples=t_vals_coarse,
            weights=coarse_weights,
            bool_randomized=self.cfg.train.randomized,
            bool_disparity=self.cfg.ray_param.disparity,
            ray_shape=self.cfg.ray_param.shape,
            bool_stop_resample_grad=True,
            resampled_padding=self.cfg.ray_param.resampled_padding,
            L_bands=self.cfg.ray_param.L_bands,
            num_samples= int(num_points * self.cfg.ray_param.fine_sampling_multiplier),
            class_token=True
            )

        """
        Tokenizer
        """

        conic_tokens = self.point_embedding(ipe_features) # encodes xyz to dim size 
        ray_tokens = torch.cat((coarse_color_token.unsqueeze(1),conic_tokens), 1)
        
        """
        Transformer Encoding
        """

        ray_tokens, attn_weights = self.fine(ray_tokens, attn_masks)

        fine_color_token, conic_tokens = ray_tokens[:,0,:], ray_tokens[:,1:, :]

        rgb_raw_fine_diff = self.rgb_f_diff(fine_color_token)
        rgb_fine_diff = self.sig(rgb_raw_fine_diff -  1.0986122886681098)
        
        rgb_fine_spec_fea = self.f_spec_colour(conic_tokens, light_tokens, view_token)

        rgb_fine_spec_fea = torch.cat((rgb_fine_spec_fea, dir_emb.unsqueeze(1)), 2)


        rgb_fine_spec_rgb = self.f_spec_rgb(rgb_fine_spec_fea.squeeze(1)) * 0.5
    
        rgb_fine = torch.clip(linear_to_srgb(rgb_fine_spec_rgb + rgb_fine_diff, eps ), 0.0, 1.0)


        return rgb_coarse, rgb_fine, attn_weights[:,0,1:]

