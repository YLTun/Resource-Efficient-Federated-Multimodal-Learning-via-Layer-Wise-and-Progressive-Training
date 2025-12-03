from dataclasses import dataclass
import torch


@dataclass
class Config:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir: str = '../datasets/coco_ratio/train2017'                                     # Change according to your path.
    train_anno_dir: str = '../datasets/coco_ratio/annotations/captions_train2017.json'      # Change according to your path.
    val_dir: str = '../datasets/coco_ratio/val2017'                                         # Change according to your path.
    val_anno_dir: str = '../datasets/coco_ratio/annotations/captions_val2017.json'          # Change according to your path.

    img_size: int = 224                # 64, 96, 128
    batch_size: int = 32               # 32, 64, 128, 256
    num_workers: int = 2
    prefetch_factor: int = 2
    save_epoch = 15

    img_enc_name: str = 'WinKawaks/vit-tiny-patch16-224'
    img_feat_dim: int = 192                                 # base: 768, tiny: 192
    interpolate_pos_encoding: bool = False                  # False
    
    text_enc_name: str = 'distilbert-base-uncased'
    text_feat_dim: int = 768
    max_length: int = 40                                   # 200
    
    hidden_dim: int = 4096                                 # 256, 512, 1024, 2048, 4096
    projection_dim: int = 256                              # 256, 512, 1024, 2048
    dropout: float = 0.1
    num_layers: int = 3                                    # 2, 3, 4 # projection head layers.

    base_head_lr: float = 1e-4                             # 1e-4, 1e-3, 1e-2
    base_img_enc_lr: float = 1e-7                          # 1e-8, 5e-7, 1e-7, 1e-6, 1e-5, 1e-4
    base_text_enc_lr: float = 1e-5                         # 1e-6, 1e-5, 3e-5, 1e-4
    
    head_lr: float = base_head_lr * batch_size / 256              # lr scaling.
    img_enc_lr: float = base_img_enc_lr * batch_size / 256          
    text_enc_lr: float = base_text_enc_lr * batch_size / 256          
    weight_decay: float = 1e-6         # 1e-5
    temperature: float = 0.05          # 0.01, 0.05, 0.07, 0.1, 0.5, 0.9

    num_local_epochs: int = 3
    num_clients: int = 10              # 10, 40
    active_num_clients: int = None     # 10, 20, None
    valid_client_ratio: float = None   # 0.8, 0.7, 0.5, 0.3, None

    set_num: str = '90_uni_lr' 
    num_stages: int = 6
    num_epochs: tuple[int] = (15, 15, 15, 15, 15, 15)
    num_img_layers: tuple[int] = (2, 2, 2, 2, 2, 2)
    num_text_layers: tuple[int] = (1, 1, 1, 1, 1, 1)
    stage_lr: bool = True
    stage_img_lr: tuple[float] = (1e-7, 1e-7, 1e-7, 5e-6, 3e-5, 5e-5)
    stage_text_lr: tuple[float] = (1e-7, 1e-7, 1e-7, 1e-5, 3e-5, 5e-5)
    stage_head_lr: tuple[float] = (3e-4, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4)   