from torch.utils.data import DataLoader
from .dynamic import DNeRFDataset

def parse_dataloader(cfg) -> None:
    if cfg.name == "DNeRFDataset":
        train_dataset = DNeRFDataset(cfg, split="train")
        val_dataset = DNeRFDataset(cfg, split="test")
        render_dataset = DNeRFDataset(cfg, split="render")
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers, 
        )
    
    return {
        "train" : train_dataloader,
        "val" : val_dataset,
        "render" : render_dataset,
    }