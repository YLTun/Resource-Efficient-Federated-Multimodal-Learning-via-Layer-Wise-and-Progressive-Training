from dataclasses import dataclass
import torch


@dataclass
class Config:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_img_dir: str = '../datasets/ADVANCE/train/vision'     # Change according to your path.
    train_audio_dir: str = '../datasets/ADVANCE/train/sound'    # Change according to your path.
    
    val_img_dir: str = '../datasets/ADVANCE/test/vision'        # Change according to your path.
    val_audio_dir: str = '../datasets/ADVANCE/test/sound'       # Change according to your path.

    img_size: int = 224                 # 64, 96, 128, 224
    batch_size: int = 16                # 16 
    num_workers: int = 2
    prefetch_factor: int = 2
    save_epoch = 15

    img_enc_name: str = 'WinKawaks/vit-tiny-patch16-224'
    img_feat_dim: int = 192                                # base: 768, tiny: 192

    audio_enc_name: str = 'bookbot/distil-ast-audioset'
    audio_feat_dim: int = 768

    # Projection head.
    hidden_dim: int = 4096                                 # 256, 512, 1024, 2048, 4096
    projection_dim: int = 256                              # 256, 512, 1024, 2048
    dropout: float = 0.1
    num_layers: int = 3                                    # 2, 3, 4  # projection head layers.

    # Prediction head.
    pred_hidden_dim: int = 256
    pred_num_layers: int = 2
    pred_dropout: float = 0.1

    base_head_lr: float = 1e-4
    base_img_enc_lr: float = 1e-7
    base_audio_enc_lr: float = 1e-5
    base_pred_head_lr: float = 1e-4

    img_enc_lr: float = base_img_enc_lr * batch_size / 256
    audio_enc_lr: float = base_audio_enc_lr * batch_size / 256
    head_lr: float = base_head_lr * batch_size / 256               # lr scaling.
    pred_head_lr: float = base_pred_head_lr * batch_size / 256

    weight_decay: float = 1e-6
    temperature: float = 0.05

    num_local_epochs: int = 3
    num_clients: int = 40
    active_num_clients: int = None     # 10, None
    beta: float = 0.5

    set_num: str = '90_lh10_lr'
    num_stages: int = 6
    num_epochs: tuple[int] = (0, 0, 0, 10, 30, 50)
    num_img_layers: tuple[int] = (2, 2, 2, 2, 2, 2)
    num_audio_layers: tuple[int] = (1, 1, 1, 1, 1, 1)
    stage_lr: bool = True
    stage_img_lr: tuple[float] = (1e-7, 1e-7, 1e-7, 5e-7, 1e-6, 5e-6)
    stage_audio_lr: tuple[float] = (1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 3e-5)
    stage_head_lr: tuple[float] = (1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)
    stage_pred_head_lr: tuple[float] = (1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)

    # set_num: str = '90_uni_lr' 
    # num_stages: int = 6
    # num_epochs: tuple[int] = (15, 15, 15, 15, 15, 15)
    # num_img_layers: tuple[int] = (2, 2, 2, 2, 2, 2)
    # num_audio_layers: tuple[int] = (1, 1, 1, 1, 1, 1)
    # stage_lr: bool = True
    # stage_img_lr: tuple[float] = (1e-7, 1e-7, 1e-7, 5e-7, 1e-6, 5e-6)
    # stage_audio_lr: tuple[float] = (1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 3e-5)
    # stage_head_lr: tuple[float] = (1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)
    # stage_pred_head_lr: tuple[float] = (1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)
    

   

