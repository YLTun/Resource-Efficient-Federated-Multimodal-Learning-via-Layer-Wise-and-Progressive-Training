import torch
import torch.nn as nn
from transformers import DistilBertModel, ViTModel, ASTModel, ResNetModel, MobileBertModel
import os
import torch.nn.functional as F

class ReshapeLayer(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        dim1 = x.shape[1]
        dim2 = x.shape[2]
        return x.view((batch_size, dim2, dim1))

class PredictionHead(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.,):
        super().__init__()
        
        mlp = []
        for layer_idx in range(num_layers):
            dim_1 = embedding_dim if layer_idx == 0 else hidden_dim
            dim_2 = num_classes if layer_idx == (num_layers - 1) else hidden_dim

            mlp.append(nn.Linear(dim_1, dim_2))
            if layer_idx < num_layers - 1:
                mlp.append(nn.Dropout(dropout))
                mlp.append(nn.ReLU(inplace=True))

        self.head = nn.Sequential(*mlp)

    def forward(self, x):
        return self.head(x)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=4096, projection_dim=256, num_layers=3, dropout=0., last_bn=True):
        super().__init__()

        mlp = []
        for layer_idx in range(num_layers):
            dim_1 = embedding_dim if layer_idx == 0 else hidden_dim
            dim_2 = projection_dim if layer_idx == (num_layers - 1) else hidden_dim

            mlp.append(nn.Linear(dim_1, dim_2, bias=False))
            if layer_idx < num_layers - 1:
                mlp.append(nn.Dropout(dropout))
                mlp.append(nn.BatchNorm1d(dim_2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim_2, affine=False))

        self.head = nn.Sequential(*mlp)

    def forward(self, x):
        return self.head(x)


class ImageEncoder(nn.Module):

    def __init__(self, config=None, model_name=None, is_trainable=True):
        super().__init__()

        if (config is not None) and (model_name is not None):
            raise ValueError('Both config and model_name cannot be given.')

        if config is not None:
            self.model = ViTModel(config=config)
        elif model_name is not None:
            self.model = ViTModel.from_pretrained(model_name)
        else:
            raise ValueError('config or model_name required.')
            
        for p in self.model.parameters():
            p.requires_grad = is_trainable

        # We are using the CLS token hidden representation as the sentence's embedding.
        self.target_token_idx = 0

    def forward(self, x, interpolate_pos_encoding=False):
        output = self.model(x, interpolate_pos_encoding=interpolate_pos_encoding)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class TextEncoder(nn.Module):
    def __init__(self, config=None, model_name=None, is_trainable=True):
        super().__init__()

        if (config is not None) and (model_name is not None):
            raise ValueError('Both config and model_name cannot be given.')

        if config is not None:
            self.model = DistilBertModel(config=config)
        elif model_name is not None:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            raise ValueError('config or model_name required.')

        for p in self.model.parameters():
            p.requires_grad = is_trainable

        # We are using the CLS token hidden representation as the sentence's embedding.
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class AudioEncoder(nn.Module):

    def __init__(self, config=None, model_name=None, is_trainable=True):
        super().__init__()

        if (config is not None) and (model_name is not None):
            raise ValueError('Both config and model_name cannot be given.')

        if config is not None:
            self.model = ASTModel(config=config)
        elif model_name is not None:
            self.model = ASTModel.from_pretrained(model_name)
        else:
            raise ValueError('config or model_name required.')
            
        for p in self.model.parameters():
            p.requires_grad = is_trainable

        # We are using the CLS token hidden representation as the sentence's embedding.
        self.target_token_idx = 0

    def forward(self, x):
        output = self.model(x)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ClipCoco(nn.Module):
    def __init__(
        self,
        img_encoder,
        text_encoder,
        img_proj_head,
        text_proj_head,
    ):
        super().__init__()

        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.img_proj_head = img_proj_head
        self.text_proj_head = text_proj_head

    def forward(self, img, input_ids, attention_mask, inter_features=False, interpolate_pos_encoding=False):
        img_features = self.img_encoder(
            img, 
            interpolate_pos_encoding=interpolate_pos_encoding
        )
        
        text_features = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )

        img_proj = self.img_proj_head(img_features)
        text_proj = self.text_proj_head(text_features)

        if inter_features:
            return (img_features, text_features), (img_proj, text_proj)
        return (img_proj, text_proj)


# Prepare text model for layer-wise learning.
def prep_text_model(text_model, num_layers, layer_class, weights_dir=None):

    # Freeze current layers.
    if len(text_model.model.transformer.layer) > 0:
        for param in text_model.parameters():
            param.requires_grad = False
    else:
        if weights_dir is not None:
            # At the first stage, load pretrained weights for embedding layer .
            layer_weights = torch.load(os.path.join(weights_dir, 'distilbert_embeddings.pth'))
            text_model.model.embeddings.load_state_dict(layer_weights)

    
    for i in range(num_layers):
        # Add a new layer.  
        text_model.model.transformer.layer.append(layer_class(text_model.model.config))
        text_model.model.config.num_hidden_layers += 1 

        if weights_dir is not None:
            # Load pre-trained weights.
            layer_idx = text_model.model.config.num_hidden_layers - 1
            layer_weights = torch.load(os.path.join(weights_dir, 'distilbert_layer_{}.pth'.format(layer_idx)))
            text_model.model.transformer.layer[layer_idx].load_state_dict(layer_weights)

    return text_model


# Prepare image model for layer-wise learning.
def prep_img_model(img_model, num_layers, layer_class, weights_dir=None):

    # Freeze current layers.
    if len(img_model.model.encoder.layer) > 0:
        for param in img_model.parameters():
            param.requires_grad = False

        # Unfreeze last norm layer in ViT.
        for param in img_model.model.layernorm.parameters():
            param.requires_grad = True
    else:
        if weights_dir is not None:
            # At the first stage, load pretrained weights for embedding layer and layernorm.
            layer_weights = torch.load(os.path.join(weights_dir, 'vit_embeddings.pth'))
            img_model.model.embeddings.load_state_dict(layer_weights)

            # layer_weights = torch.load(os.path.join(weights_dir, 'vit_layernorm.pth'))
            # img_model.model.layernorm.load_state_dict(layer_weights)

    for i in range(num_layers):
        # Add a new layer. 
        img_model.model.encoder.layer.append(layer_class(img_model.model.config))
        img_model.model.config.num_hidden_layers += 1 

        if weights_dir is not None:
            # Load pre-trained weights.
            layer_idx = img_model.model.config.num_hidden_layers - 1
            layer_weights = torch.load(os.path.join(weights_dir, 'vit_layer_{}.pth'.format(layer_idx)))
            img_model.model.encoder.layer[layer_idx].load_state_dict(layer_weights)

    return img_model


# Prepare audio model for layer-wise learning.
def prep_audio_model(audio_model, num_layers, layer_class, weights_dir=None):

    # Freeze current layers.
    if len(audio_model.model.encoder.layer) > 0:
        for param in audio_model.parameters():
            param.requires_grad = False

        # Unfreeze last norm layer in ViT.
        for param in audio_model.model.layernorm.parameters():
            param.requires_grad = True
    else:
        if weights_dir is not None:
            # At the first stage, load pretrained weights for embedding layer and layernorm.
            layer_weights = torch.load(os.path.join(weights_dir, 'ast_embeddings.pth'))
            audio_model.model.embeddings.load_state_dict(layer_weights)
    
            # layer_weights = torch.load(os.path.join(weights_dir, 'vit_layernorm.pth'))
            # img_model.model.layernorm.load_state_dict(layer_weights)

    for i in range(num_layers):
        # Add a new layer. 
        audio_model.model.encoder.layer.append(layer_class(audio_model.model.config))
        audio_model.model.config.num_hidden_layers += 1 

        if weights_dir is not None:
            # Load pre-trained weights.
            layer_idx = audio_model.model.config.num_hidden_layers - 1
            layer_weights = torch.load(os.path.join(weights_dir, 'ast_layer_{}.pth'.format(layer_idx)))
            audio_model.model.encoder.layer[layer_idx].load_state_dict(layer_weights)

    return audio_model


# Prepare text model for progressive learning.
def prep_text_model_prog(text_model, num_layers, layer_class, weights_dir=None):

    # At the first stage, load pretrained weights for embedding layer.
    if len(text_model.model.transformer.layer) < 1 and weights_dir is not None:
        layer_weights = torch.load(os.path.join(weights_dir, 'distilbert_embeddings.pth'))
        text_model.model.embeddings.load_state_dict(layer_weights)

    for i in range(num_layers):
        # Add a new layer.  
        text_model.model.transformer.layer.append(layer_class(text_model.model.config))
        text_model.model.config.num_hidden_layers += 1 

        if weights_dir is not None:
            # Load pre-trained weights.
            layer_idx = text_model.model.config.num_hidden_layers - 1
            layer_weights = torch.load(os.path.join(weights_dir, 'distilbert_layer_{}.pth'.format(layer_idx)))
            text_model.model.transformer.layer[layer_idx].load_state_dict(layer_weights)

    return text_model


# Prepare image model for progressive learning.
def prep_img_model_prog(img_model, num_layers, layer_class, weights_dir=None):

    # At the first stage, load pretrained weights for embedding layer and layernorm.
    if len(img_model.model.encoder.layer) < 1 and weights_dir is not None:
        layer_weights = torch.load(os.path.join(weights_dir, 'vit_embeddings.pth'))
        img_model.model.embeddings.load_state_dict(layer_weights)

        # layer_weights = torch.load(os.path.join(weights_dir, 'vit_layernorm.pth'))
        # img_model.model.layernorm.load_state_dict(layer_weights)

    for i in range(num_layers):
        # Add a new layer. 
        img_model.model.encoder.layer.append(layer_class(img_model.model.config))
        img_model.model.config.num_hidden_layers += 1 

        if weights_dir is not None:
            # Load pre-trained weights.
            layer_idx = img_model.model.config.num_hidden_layers - 1
            layer_weights = torch.load(os.path.join(weights_dir, 'vit_layer_{}.pth'.format(layer_idx)))
            img_model.model.encoder.layer[layer_idx].load_state_dict(layer_weights)

    return img_model


# Prepare audio model for progressive learning.
def prep_audio_model_prog(audio_model, num_layers, layer_class, weights_dir=None):

    # At the first stage, load pretrained weights for embedding layer and layernorm.
    if len(audio_model.model.encoder.layer) < 1 and weights_dir is not None:
        layer_weights = torch.load(os.path.join(weights_dir, 'ast_embeddings.pth'))
        audio_model.model.embeddings.load_state_dict(layer_weights)

        # layer_weights = torch.load(os.path.join(weights_dir, 'vit_layernorm.pth'))
        # img_model.model.layernorm.load_state_dict(layer_weights)

    for i in range(num_layers):
        # Add a new layer. 
        audio_model.model.encoder.layer.append(layer_class(audio_model.model.config))
        audio_model.model.config.num_hidden_layers += 1 

        if weights_dir is not None:
            # Load pre-trained weights.
            layer_idx = audio_model.model.config.num_hidden_layers - 1
            layer_weights = torch.load(os.path.join(weights_dir, 'ast_layer_{}.pth'.format(layer_idx)))
            audio_model.model.encoder.layer[layer_idx].load_state_dict(layer_weights)

    return audio_model


class SupAdvance(nn.Module):
    def __init__(
        self,
        img_encoder,
        audio_encoder,
        img_proj_head,
        audio_proj_head,
        pred_head,
    ):
        super().__init__()

        self.img_encoder = img_encoder
        self.audio_encoder = audio_encoder
        self.img_proj_head = img_proj_head
        self.audio_proj_head = audio_proj_head
        self.pred_head = pred_head

    def forward(self, img, audio, inter_features=False, interpolate_pos_encoding=False):
        img_features = self.img_encoder(img, interpolate_pos_encoding=interpolate_pos_encoding)
        audio_features = self.audio_encoder(audio)

        img_proj = self.img_proj_head(img_features)
        audio_proj = self.audio_proj_head(audio_features)
        
        concat_proj = torch.cat((img_proj, audio_proj), dim=-1)

        label_pred = self.pred_head(concat_proj)

        if inter_features:
            return (img_features, audio_features), (img_proj, audio_proj), label_pred
        return label_pred


class SupAdvanceFuse(nn.Module):
    def __init__(
        self,
        img_encoder,
        audio_encoder,
        img_proj_head,
        audio_proj_head,
        fusion_head,
        pred_head,
    ):
        super().__init__()

        self.img_encoder = img_encoder
        self.audio_encoder = audio_encoder
        self.img_proj_head = img_proj_head
        self.audio_proj_head = audio_proj_head
        self.fusion_head = fusion_head
        self.pred_head = pred_head

    def forward(self, img, audio, inter_features=False, interpolate_pos_encoding=False):
        img_features = self.img_encoder(img, interpolate_pos_encoding=interpolate_pos_encoding)
        audio_features = self.audio_encoder(audio)

        img_proj = self.img_proj_head(img_features)
        audio_proj = self.audio_proj_head(audio_features)
        
        fused_proj = self.fusion_head(img_proj, audio_proj)

        label_pred = self.pred_head(fused_proj)

        if inter_features:
            return (img_features, audio_features), (img_proj, audio_proj), label_pred
        return label_pred


class SupUCIHARFuse(nn.Module):
    def __init__(
        self,
        acc_encoder,
        gyro_encoder,
        acc_proj_head,
        gyro_proj_head,
        fusion_head,
        pred_head,
    ):
        super().__init__()

        self.acc_encoder = acc_encoder
        self.gyro_encoder = gyro_encoder
        self.acc_proj_head = acc_proj_head
        self.gyro_proj_head = gyro_proj_head
        self.fusion_head = fusion_head
        self.pred_head = pred_head

    def forward(self, acc, gyro, inter_features=False):
        acc_features = self.acc_encoder(acc)
        gyro_features = self.gyro_encoder(gyro)

        acc_proj = self.acc_proj_head(acc_features)
        gyro_proj = self.gyro_proj_head(gyro_features)
        
        fused_proj = self.fusion_head(acc_proj, gyro_proj)

        label_pred = self.pred_head(fused_proj)

        if inter_features:
            return (acc_features, gyro_features), (acc_proj, gyro_proj), label_pred
        return label_pred



class FusionHeadConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_proj, audio_proj):
        fused_proj = torch.cat((img_proj, audio_proj), dim=-1)
        return fused_proj


class FusionHeadAdd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_proj, audio_proj):
        concat_proj = torch.cat((img_proj, audio_proj), dim=-1)
        fused_proj = torch.add(img_proj, audio_proj)
        fused_proj = torch.cat((concat_proj, fused_proj), dim=-1)
        return fused_proj


# https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/fusions/attention_fusion.py
class FusionHeadAtten(nn.Module):
    def __init__(self, projection_dim):
        super().__init__()

        attn_in_dim = 2 * projection_dim 
        head_in_dim = projection_dim
        num_channels = 2   # Number of modalities.
        
        self.attention = nn.Sequential(
            nn.Linear(attn_in_dim, head_in_dim),
            nn.Softmax(-1),
        )
                     
        self.encoding_projection = nn.ModuleList([
            nn.Linear(head_in_dim, projection_dim) for i in range(10)
        ])
        

    def forward(self, img_proj, audio_proj):
        concat_proj = torch.cat((img_proj, audio_proj), dim=-1)
        
        embeddings = [img_proj, audio_proj]
        concatenated_in = torch.cat(
            [embedding for embedding in embeddings], dim=-1
        )

        attention_weights = self.attention(concatenated_in)
        projected_embeddings = []
        for embedding, projection in zip(embeddings, self.encoding_projection):
            projected_embedding = projection(embedding)
            projected_embeddings.append(projected_embedding)
        
        for i in range(len(projected_embeddings)):
            projected_embeddings[i] = (
                attention_weights[:, i].unsqueeze(-1) * projected_embeddings[i]
            )

        fused_proj = torch.sum(torch.stack(projected_embeddings), dim=0)
        fused_proj = torch.cat((concat_proj, fused_proj), dim=-1)
        return fused_proj


# https://github.com/atulkum/co-attention/blob/master/code/model.py
class FusionHeadCoAtten(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, img_proj, audio_proj):
        concat_proj = torch.cat((img_proj, audio_proj), dim=-1)
        # Project.
        img_proj = torch.tanh(self.linear(img_proj.view(-1, self.hidden_dim))).view(img_proj.size()) #B x n + 1 x l
        img_proj = img_proj.unsqueeze(dim=-1)
        
        # Co-attention
        audio_proj = audio_proj.unsqueeze(dim=-1)
        audio_proj_t = torch.transpose(audio_proj, 1, 2)  #B x l x m + 1
        L = torch.bmm(img_proj, audio_proj_t) # L = B x n + 1 x m + 1

        A_img_proj_ = F.softmax(L, dim=1) # B x n + 1 x m + 1
        A_img_proj = torch.transpose(A_img_proj_, 1, 2) # B x m + 1 x n + 1
        C_img_proj = torch.bmm(audio_proj_t, A_img_proj)
        C_img_proj = C_img_proj.squeeze(dim=1)

        img_proj_t = torch.transpose(img_proj, 1, 2)  # B x l x n + 1
        A_audio_proj = F.softmax(L, dim=2)  # B x n + 1 x m + 1
        C_audio_proj = torch.bmm(img_proj_t, A_audio_proj)
        C_audio_proj = C_audio_proj.squeeze(dim=1)

        fused_proj = torch.cat((C_img_proj, C_audio_proj), dim=-1)
        fused_proj = torch.cat((concat_proj, fused_proj), dim=-1)
        return fused_proj

        
class SupAdvanceHGB(nn.Module):
    def __init__(
        self,
        img_encoder,
        audio_encoder,
        img_proj_head,
        audio_proj_head,
        img_pred_head,
        audio_pred_head,
        pred_head,
    ):
        super().__init__()

        self.img_encoder = img_encoder
        self.audio_encoder = audio_encoder
        self.img_proj_head = img_proj_head
        self.audio_proj_head = audio_proj_head
        self.img_pred_head = img_pred_head
        self.audio_pred_head = audio_pred_head
        self.pred_head = pred_head

    def forward(self, img, audio, inter_features=False, interpolate_pos_encoding=False):
        img_features = self.img_encoder(img, interpolate_pos_encoding=interpolate_pos_encoding)
        audio_features = self.audio_encoder(audio)

        img_proj = self.img_proj_head(img_features)
        audio_proj = self.audio_proj_head(audio_features)

        # Modality specific prediction.
        img_pred = self.img_pred_head(img_proj)
        audio_pred = self.audio_pred_head(audio_proj)

        # Combined prediction.
        concat_proj = torch.cat((img_proj, audio_proj), dim=-1)
        label_pred = self.pred_head(concat_proj)

        if inter_features:
            return (img_features, audio_features), (img_proj, audio_proj), label_pred, img_pred, audio_pred
        
        return label_pred


class ClipCocoAdapt(nn.Module):
    def __init__(
        self,
        img_encoder,
        text_encoder,
        img_proj_head,
        text_proj_head,
        img_adapter, 
        text_adapter,
    ):
        super().__init__()

        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.img_proj_head = img_proj_head
        self.text_proj_head = text_proj_head
        self.img_adapter = img_adapter
        self.text_adapter = text_adapter

    def forward(self, img, input_ids, attention_mask, inter_features=False, interpolate_pos_encoding=False):
        # interpolate_pos_encoding in the input is not used.
        # It is there to be consistent and not break the code.
        img_features = self.img_encoder(img)
        
        text_features = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )

        img_adapt = self.img_adapter(img_features)
        text_adapt = self.text_adapter(text_features)

        img_proj = self.img_proj_head(img_adapt)
        text_proj = self.text_proj_head(text_adapt)

        if inter_features:
            return (img_features, text_features), (img_proj, text_proj)
        return (img_proj, text_proj)


class ImageEncoderResNet(nn.Module):

    def __init__(self, config=None, model_name=None, is_trainable=True):
        super().__init__()

        if (config is not None) and (model_name is not None):
            raise ValueError('Both config and model_name cannot be given.')

        if config is not None:
            self.model = ResNetModel(config=config)
        elif model_name is not None:
            self.model = ResNetModel.from_pretrained(model_name)
        else:
            raise ValueError('config or model_name required.')
            
        for p in self.model.parameters():
            p.requires_grad = is_trainable

        self.flatten = nn.Flatten()

    def forward(self, x, interpolate_pos_encoding=False):
        # interpolate_pos_encoding is not required for resnet.
        # It is there to be consistent and not break the code.
        output = self.model(x)
        return self.flatten(output[1])


class TextEncoderMobileBert(nn.Module):
    def __init__(self, config=None, model_name=None, is_trainable=True):
        super().__init__()

        if (config is not None) and (model_name is not None):
            raise ValueError('Both config and model_name cannot be given.')

        if config is not None:
            self.model = MobileBertModel(config=config)
        elif model_name is not None:
            self.model = MobileBertModel.from_pretrained(model_name)
        else:
            raise ValueError('config or model_name required.')

        for p in self.model.parameters():
            p.requires_grad = is_trainable

        # We are using the CLS token hidden representation as the sentence's embedding.
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


def get_adapter(dim_1, dim_2):
    adapter = nn.Identity()
    if dim_1 != dim_2:
        adapter = nn.Linear(dim_1, dim_2)
    return adapter


# Prepare text model for layer-wise learning.
def prep_text_model_mobilebert(text_model, num_layers, layer_class, weights_dir):

    # Freeze current layers.
    if len(text_model.model.encoder.layer) > 0:
        for param in text_model.parameters():
            param.requires_grad = False
    else:
        # At the first stage, load pretrained weights for embedding layer .
        layer_weights = torch.load(os.path.join(weights_dir, 'mobilebert_embeddings.pth'))
        text_model.model.embeddings.load_state_dict(layer_weights)

    
    for i in range(num_layers):
        # Add a new layer.  
        text_model.model.encoder.layer.append(layer_class(text_model.model.config))
        text_model.model.config.num_hidden_layers += 1 

        # Load pre-trained weights.
        layer_idx = text_model.model.config.num_hidden_layers - 1
        layer_weights = torch.load(os.path.join(weights_dir, 'mobilebert_layer_{}.pth'.format(layer_idx)))
        text_model.model.encoder.layer[layer_idx].load_state_dict(layer_weights)

    return text_model


# Prepare image model for layer-wise learning.
def prep_img_model_resnet(img_model, num_layers, layer_class, weights_dir):

    config = img_model.model.config
    
    # Freeze current layers.
    if len(img_model.model.encoder.stages) > 0:
        for param in img_model.parameters():
            param.requires_grad = False
    else:
        # At the first stage, load pretrained weights for embedding layer.
        layer_weights = torch.load(os.path.join(weights_dir, 'embedder.pth'))
        img_model.model.embedder.load_state_dict(layer_weights)

        stage = layer_class(
            config,
            config.embedding_size,
            config.hidden_sizes[0],
            stride=2 if config.downsample_in_first_stage else 1,
            depth=config.depths[0],
        )
        img_model.model.encoder.stages.append(stage)
        # Load pre-trained weights.
        layer_idx = len(img_model.model.encoder.stages) - 1
        layer_weights = torch.load(os.path.join(weights_dir, 'stage_{}.pth'.format(layer_idx)))
        img_model.model.encoder.stages[layer_idx].load_state_dict(layer_weights)
        
        num_layers -= 1
    
    in_out_channels = list(zip(config.hidden_sizes, config.hidden_sizes[1:]))
    depths = config.depths[1:]
    
    for i in range(num_layers):

        layer_idx = len(img_model.model.encoder.stages) - 1
        (in_channels, out_channels) = in_out_channels[layer_idx]
        depth = depths[layer_idx]
        stage = layer_class(config, in_channels, out_channels, depth=depth)
        
        # Add a new layer. 
        img_model.model.encoder.stages.append(stage)

        # Load pre-trained weights.
        layer_weights = torch.load(os.path.join(weights_dir, 'stage_{}.pth'.format(layer_idx)))
        img_model.model.encoder.stages[layer_idx].load_state_dict(layer_weights)

    return img_model


# Prepare text model for progressive learning.
def prep_text_model_prog_mobilebert(text_model, num_layers, layer_class, weights_dir=None):

    # At the first stage, load pretrained weights for embedding layer.
    if len(text_model.model.encoder.layer) < 1 and weights_dir is not None:
        layer_weights = torch.load(os.path.join(weights_dir, 'mobilebert_embeddings.pth'))
        text_model.model.embeddings.load_state_dict(layer_weights)

    for i in range(num_layers):
        # Add a new layer.  
        text_model.model.encoder.layer.append(layer_class(text_model.model.config))
        text_model.model.config.num_hidden_layers += 1 

        if weights_dir is not None:
            # Load pre-trained weights.
            layer_idx = text_model.model.config.num_hidden_layers - 1
            layer_weights = torch.load(os.path.join(weights_dir, 'mobilebert_layer_{}.pth'.format(layer_idx)))
            text_model.model.encoder.layer[layer_idx].load_state_dict(layer_weights)

    return text_model


# Prepare image model for progressive learning.
def prep_img_model_prog_resnet(img_model, num_layers, layer_class, weights_dir=None):
    
    config = img_model.model.config
    
    # At the first stage, load pretrained weights for embedding layer and layernorm.
    if len(img_model.model.encoder.stages) < 1 and weights_dir is not None:
        layer_weights = torch.load(os.path.join(weights_dir, 'embedder.pth'))
        img_model.model.embedder.load_state_dict(layer_weights)

        stage = layer_class(
            config,
            config.embedding_size,
            config.hidden_sizes[0],
            stride=2 if config.downsample_in_first_stage else 1,
            depth=config.depths[0],
        )
        img_model.model.encoder.stages.append(stage)
        # Load pre-trained weights.
        layer_idx = len(img_model.model.encoder.stages) - 1
        layer_weights = torch.load(os.path.join(weights_dir, 'stage_{}.pth'.format(layer_idx)))
        img_model.model.encoder.stages[layer_idx].load_state_dict(layer_weights)
        
        num_layers -= 1

    in_out_channels = list(zip(config.hidden_sizes, config.hidden_sizes[1:]))
    depths = config.depths[1:]
    
    for i in range(num_layers):

        layer_idx = len(img_model.model.encoder.stages) - 1
        (in_channels, out_channels) = in_out_channels[layer_idx]
        depth = depths[layer_idx]
        stage = layer_class(config, in_channels, out_channels, depth=depth)
        
        # Add a new layer. 
        img_model.model.encoder.stages.append(stage)

        if weights_dir is not None:
            # Load pre-trained weights.
            layer_weights = torch.load(os.path.join(weights_dir, 'stage_{}.pth'.format(layer_idx)))
            img_model.model.encoder.stages[layer_idx].load_state_dict(layer_weights)

    return img_model


class SupAdvanceFuseAdapt(nn.Module):
    def __init__(
        self,
        img_encoder,
        audio_encoder,
        img_proj_head,
        audio_proj_head,
        img_adapter, 
        audio_adapter,
        fusion_head,
        pred_head,
    ):
        super().__init__()

        self.img_encoder = img_encoder
        self.audio_encoder = audio_encoder
        self.img_adapter = img_adapter
        self.audio_adapter = audio_adapter
        self.img_proj_head = img_proj_head
        self.audio_proj_head = audio_proj_head
        self.fusion_head = fusion_head
        self.pred_head = pred_head

    def forward(self, img, audio, inter_features=False, interpolate_pos_encoding=False):
        img_features = self.img_encoder(img, interpolate_pos_encoding=interpolate_pos_encoding)
        audio_features = self.audio_encoder(audio)

        img_adapt = self.img_adapter(img_features)
        audio_adapt = self.audio_adapter(audio_features)

        img_proj = self.img_proj_head(img_adapt)
        audio_proj = self.audio_proj_head(audio_adapt)
        
        fused_proj = self.fusion_head(img_proj, audio_proj)

        label_pred = self.pred_head(fused_proj)

        if inter_features:
            return (img_features, audio_features), (img_proj, audio_proj), label_pred
        return label_pred
        