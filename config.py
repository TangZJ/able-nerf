from dataclasses import dataclass

@dataclass
class Expt_settings:
    seed: int
    num_gpu: int
    exp_name: str
    test_name: str

@dataclass
class Dataset:
    type: str
    name: str
    data_dir: str
    factor: int


@dataclass
class Train:
    batch_size: int
    batch_type: str
    num_workers: int
    randomized: bool
    white_bkgd: bool

@dataclass
class Val: 
    im_batch_size: int
    batch_type: str
    batch_size: int
    num_workers: int
    check_interval: int
    randomized: bool
    white_bkgd: bool
    limit_batch_size: int

@dataclass
class Ray_param:
    num_samples: int
    fine_sampling_multiplier: float
    perturb: float
    noise_std: int
    L_bands: int
    disparity: bool
    randomized: bool
    shape: str
    resampled_padding: float

@dataclass
class Model:
    coarse_layers: int
    fine_layers: int
    ff_ratio: int
    dropout: float
    dim_in: int
    version: str
    num_lp: int
    lp_layers: int

@dataclass
class Optimizer:
    lr_init: float
    lr_final: float
    lr_delay_steps: int
    lr_delay_mult: float
    max_steps: float
    loss_coarse: float
    loss_fine: float

@dataclass
class Checkpoint:
    resume_path: str

@dataclass
class Systemcfg:
    expt_settings: Expt_settings
    dataset: Dataset
    train: Train
    val: Val
    ray_param: Ray_param
    model: Model
    optimizer: Optimizer
    checkpoint: Checkpoint