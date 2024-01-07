from torch.utils.data import DataLoader
from .dynamic import DNeRFDataset
from .static import NeRFDataset

def parse_dataloader(
    dataset_name, 
    batch_size, 
    num_workers, 
    cfg
) -> None:
    if dataset_name == "DNeRFDataset":
        train_dataset = DNeRFDataset(cfg, split="train")
        val_dataset = DNeRFDataset(cfg, split="test")
        render_dataset = DNeRFDataset(cfg, split="render")
        
    elif dataset_name == "NeRFDataset":
        train_dataset = NeRFDataset(cfg, split="train")
        val_dataset = NeRFDataset(cfg, split="test")
        render_dataset = NeRFDataset(cfg, split="render")
        
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=list
    )
    
    return {
        "train" : train_dataloader,
        "val" : val_dataset,
        "render" : render_dataset,
    }