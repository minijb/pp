from .dataset import MVTecDataset
from torch.utils.data import DataLoader


def build_dataset(datadir:str, texturedir:str, target:str, train:bool=True, to_memory:bool=False):
    dataset = MVTecDataset(
        datadir                = datadir,
        target                 = target, 
        train                  = train,
        to_memory              = to_memory,
        resize                 = (256, 256),
        texture_source_dir     = texturedir,
        structure_grid_size    = 8,
        transparency_range     = [0.15, 1.0],
        perlin_scale           = 6, 
        min_perlin_scale       = 0, 
        perlin_noise_threshold = 0.5
    )
    return dataset

def build_dataLoader(dataset, train: bool, batch_size: int = 4, num_workers: int = 1):
    dataloader = DataLoader(
        dataset,
        shuffle     = train,
        batch_size  = batch_size,
        num_workers = num_workers
    )

    return dataloader
