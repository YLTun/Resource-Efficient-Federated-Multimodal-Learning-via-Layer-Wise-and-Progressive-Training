import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from tqdm import tqdm
import numpy as np
import subprocess
import os


def gpu_mem_usage_pid(cur_pid=None):
    # ! nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    # ! nvidia-smi --help-query-compute-apps

    if cur_pid == None:
        cur_pid = os.getpid()
    cur_gpu_mem = None

    sp = subprocess.Popen(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )

    out = sp.communicate()
    out = out[0].split(b'\n')
    for proc in out[:-1]:
        pid, mem = proc.split(b',')
        _, mem, _ = mem.split(b' ')

        if cur_pid == int(pid):
            cur_gpu_mem = int(mem)
            break

    return cur_gpu_mem


def elapsed_time(func, *args):
    gpu_stime = torch.cuda.Event(enable_timing=True)
    gpu_etime = torch.cuda.Event(enable_timing=True)

    gpu_stime.record()
    results = func(*args)
    gpu_etime.record()
    torch.cuda.synchronize()

    gpu_time = gpu_stime.elapsed_time(gpu_etime)      # milliseconds
    gpu_time = gpu_time / 1000                        # convert into seconds

    return gpu_time, results


def comp_cost(gpu_index, func, *args):
    gpu_time, results = elapsed_time(func, *args)
    gpu_mem = gpu_mem_usage_pid()

    return gpu_time, gpu_mem, results


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / 1024**2
    return total_size_mb


def clip_loss(proj_i, proj_j, temperature, loss_split=False): 
    device = proj_i.device

    proj_i = F.normalize(proj_i, dim=-1)
    proj_j = F.normalize(proj_j, dim=-1)
    logits = torch.mm(proj_i, proj_j.T) / temperature

    N = logits.shape[0]           # Batch size.
    labels = torch.arange(N, dtype=torch.long, device=device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    loss = (loss_i + loss_j) / 2.0
    
    if loss_split:
        return loss_i, loss_j, loss
    return loss


# Train one epoch.
def train_clip_coco(clip_model, data_loader, tokenizer, optimizer, CFG, scheduler=None):
    device = next(clip_model.parameters()).device
    loss_metric = utils.MeanMetric()
    result_dict = {}
    
    for img, cap in tqdm(data_loader):
        encoded_caption = tokenizer(
            cap, 
            padding=True,
            truncation=True, 
            max_length=CFG.max_length,
        )
    
        input_ids = encoded_caption['input_ids']
        attention_mask = encoded_caption['attention_mask']
    
        img = img.to(device)
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)

        # Training.
        clip_model.zero_grad(set_to_none=True)
    
        img_proj, text_proj = clip_model(img, input_ids, attention_mask, interpolate_pos_encoding=CFG.interpolate_pos_encoding)
        loss = clip_loss(img_proj, text_proj, CFG.temperature)
    
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Log for lr.
        for param_group in optimizer.param_groups:
            result_dict.setdefault(param_group['name'], list()).append(param_group['lr'])
    
        loss_metric.update_state(loss.item())

    result_dict['loss'] = loss_metric.result()
    loss_metric.reset_state()
    return result_dict


@torch.no_grad()
def val_clip_coco(clip_model, data_loader, tokenizer, CFG):
    device = next(clip_model.parameters()).device
    loss_metric = utils.MeanMetric()
    
    with utils.eval_mode(clip_model):
        for img, cap in tqdm(data_loader):
            encoded_caption = tokenizer(
                cap, 
                padding=True,
                truncation=True, 
                max_length=CFG.max_length,
            )
        
            input_ids = encoded_caption['input_ids']
            attention_mask = encoded_caption['attention_mask']
        
            img = img.to(device)
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)

            img_proj, text_proj = clip_model(img, input_ids, attention_mask, interpolate_pos_encoding=CFG.interpolate_pos_encoding)
            loss = clip_loss(img_proj, text_proj, CFG.temperature)

            loss_metric.update_state(loss.item())

    result_dict = {
        'loss': loss_metric.result()
    }
    loss_metric.reset_state()
    return result_dict


# Train one epoch.
def train_sup_advance(sup_model, data_loader, optimizer, CFG, scheduler=None):
    device = next(sup_model.parameters()).device
    loss_metric = utils.MeanMetric()
    acc_metric = utils.MeanMetric()
    criterion = nn.CrossEntropyLoss()

    result_dict = {}

    for image, audio, label in tqdm(data_loader):

        image = image.to(device)
        audio = audio.to(device)
        label = label.to(device)

        # Training.
        sup_model.zero_grad(set_to_none=True)
        label_pred = sup_model(image, audio)

        loss = criterion(label_pred, label)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Log for lr.
        for param_group in optimizer.param_groups:
            result_dict.setdefault(param_group['name'], list()).append(param_group['lr'])

        loss_metric.update_state(loss.item())
        acc_metric.update_state(utils.compute_accuracy(label, label_pred))

    result_dict['loss'] = loss_metric.result()
    result_dict['acc'] = acc_metric.result()
    loss_metric.reset_state()
    acc_metric.reset_state()
    return result_dict


@torch.no_grad()
def val_sup_advance(sup_model, data_loader, CFG):
    device = next(sup_model.parameters()).device
    loss_metric = utils.MeanMetric()
    acc_metric = utils.MeanMetric()
    criterion = nn.CrossEntropyLoss()

    with utils.eval_mode(sup_model):
        for image, audio, label in tqdm(data_loader):

            image = image.to(device)
            audio = audio.to(device)
            label = label.to(device)

            label_pred = sup_model(image, audio)
            loss = criterion(label_pred, label)

            loss_metric.update_state(loss.item())
            acc_metric.update_state(utils.compute_accuracy(label, label_pred))

    result_dict = {
        'loss': loss_metric.result(),
        'acc': acc_metric.result()
    }
    loss_metric.reset_state()
    acc_metric.reset_state()
    return result_dict


@torch.no_grad()
def val_sup_advance_metrics(sup_model, data_loader, CFG):
    device = next(sup_model.parameters()).device
    loss_metric = utils.MeanMetric()
    acc_metric = utils.MeanMetric()
    precision_metric = utils.MeanMetric()
    recall_metric = utils.MeanMetric()
    f1_metric = utils.MeanMetric()

    criterion = nn.CrossEntropyLoss()
    
    with utils.eval_mode(sup_model):
        for image, audio, label in tqdm(data_loader):
            
            image = image.to(device)
            audio = audio.to(device)
            label = label.to(device)

            label_pred = sup_model(image, audio)
            loss = criterion(label_pred, label)
            loss_metric.update_state(loss.item())
            acc_metric.update_state(utils.compute_accuracy(label, label_pred))
            precision, recall, f1 = utils.compute_precision_recall_f1(label, label_pred)
            precision_metric.update_state(precision)
            recall_metric.update_state(recall)
            f1_metric.update_state(f1)

    result_dict = {
        'loss': loss_metric.result(),
        'precision': precision_metric.result(),
        'recall': recall_metric.result(),
        'f1': f1_metric.result(),
        'acc': acc_metric.result(),
    }
    loss_metric.reset_state()
    precision_metric.reset_state()
    recall_metric.reset_state()
    f1_metric.reset_state()
    acc_metric.reset_state()
    return result_dict


# Train one epoch.
def train_sup_advance_hgb(sup_model_hgb, train_loader, val_loader, optimizer, z_img, z_audio, CFG, scheduler=None):
    
    device = next(sup_model_hgb.parameters()).device
    loss_metric = utils.MeanMetric()
    acc_metric = utils.MeanMetric()
    criterion = nn.CrossEntropyLoss()

    result_dict = {}
    for image, audio, label in tqdm(train_loader):

        image = image.to(device)
        audio = audio.to(device)
        label = label.to(device)

        # Training.
        sup_model_hgb.zero_grad(set_to_none=True)
        _, _, label_pred, img_pred, audio_pred = sup_model_hgb(image, audio, inter_features=True)

        img_loss = criterion(img_pred, label)
        audio_loss = criterion(audio_pred, label)
        img_audio_loss = criterion(label_pred, label)

        loss = img_audio_loss + (z_img * img_loss) + (z_audio * audio_loss)
        
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Log for lr.
        for param_group in optimizer.param_groups:
            result_dict.setdefault(param_group['name'], list()).append(param_group['lr'])

        loss_metric.update_state(loss.item())
        acc_metric.update_state(utils.compute_accuracy(label, label_pred))

    result_dict['loss'] = loss_metric.result()
    result_dict['acc'] = acc_metric.result()
    loss_metric.reset_state()
    acc_metric.reset_state()

    return result_dict


@torch.no_grad()
def modality_loss_hgb(sup_model_hgb, data_loader, CFG, subset_ratio=1.0):
    device = next(sup_model_hgb.parameters()).device
    img_loss_metric = utils.MeanMetric()
    audio_loss_metric = utils.MeanMetric()
    criterion = nn.CrossEntropyLoss()

    # HGB. Using only a subset of dataset.
    num_steps = len(data_loader)
    stop_step = int(num_steps * subset_ratio)
    
    with utils.eval_mode(sup_model_hgb):
        for step, (image, audio, label) in enumerate(tqdm(data_loader)):

            image = image.to(device)
            audio = audio.to(device)
            label = label.to(device)

            _, _, _, img_pred, audio_pred = sup_model_hgb(image, audio, inter_features=True)

            img_loss = criterion(img_pred, label)
            audio_loss = criterion(audio_pred, label)

            img_loss_metric.update_state(img_loss.item())
            audio_loss_metric.update_state(audio_loss.item())

            if step >= stop_step:
                 break
    
    return img_loss_metric.result(), audio_loss_metric.result()


# Train one epoch.
def train_sup_advance_pa(sup_model, glob_img_proj_head, glob_audio_proj_head, data_loader, optimizer, CFG, scheduler=None):
    device = next(sup_model.parameters()).device
    loss_metric = utils.MeanMetric()
    acc_metric = utils.MeanMetric()
    criterion = nn.CrossEntropyLoss()

    result_dict = {}

    for image, audio, label in tqdm(data_loader):

        image = image.to(device)
        audio = audio.to(device)
        label = label.to(device)

        # Training.
        sup_model.zero_grad(set_to_none=True)
        (img_features, audio_features), (img_proj, audio_proj), label_pred = sup_model(image, audio, inter_features=True)

        # =========================================================================================
        with torch.no_grad():
            glob_img_proj = glob_img_proj_head(img_features)
            glob_img_proj = glob_img_proj.detach()
            glob_audio_proj = glob_audio_proj_head(audio_features)
            glob_audio_proj = glob_audio_proj.detach()

        # img_loss_pa, _, _ = clip_loss(img_proj, glob_img_proj, CFG.temperature, loss_split=True)
        # audio_loss_pa, _, _ = clip_loss(audio_proj, glob_audio_proj, CFG.temperature, loss_split=True)

        # Cosine.
        img_loss_pa = 1 - torch.mean(torch.nn.functional.cosine_similarity(img_proj, glob_img_proj, dim=-1))
        audio_loss_pa = 1 - torch.mean(torch.nn.functional.cosine_similarity(audio_proj, glob_audio_proj, dim=-1))

        # MSE.
        # img_loss_pa = torch.nn.functional.mse_loss(img_proj, glob_img_proj)
        # audio_loss_pa = torch.nn.functional.mse_loss(audio_proj, glob_audio_proj)

        loss = criterion(label_pred, label) + CFG.aux_weight * (img_loss_pa + audio_loss_pa)
        # =========================================================================================
        
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Log for lr.
        for param_group in optimizer.param_groups:
            result_dict.setdefault(param_group['name'], list()).append(param_group['lr'])

        loss_metric.update_state(loss.item())
        acc_metric.update_state(utils.compute_accuracy(label, label_pred))

    result_dict['loss'] = loss_metric.result()
    result_dict['acc'] = acc_metric.result()
    loss_metric.reset_state()
    acc_metric.reset_state()
    return result_dict
    

@torch.no_grad()
def get_img_embed(clip_model, data_loader, CFG):

    device = next(clip_model.parameters()).device
    img_proj_list = []
    with utils.eval_mode(clip_model):
        for img, _ in tqdm(data_loader):
            img = img.to(device)
            img_features = clip_model.img_encoder(img, interpolate_pos_encoding=CFG.interpolate_pos_encoding)
            img_proj = clip_model.img_proj_head(img_features)
            img_proj_list.append(img_proj)
    return torch.cat(img_proj_list)


@torch.no_grad()
def get_similar_img_idxes(query, img_proj, tokenizer, clip_model, n=9):
    device = next(clip_model.parameters()).device
    
    encoded_query = tokenizer([query])
    input_ids = torch.tensor(encoded_query['input_ids']).to(device)
    attention_mask = torch.tensor(encoded_query['attention_mask']).to(device)

    with utils.eval_mode(clip_model):
        text_features = clip_model.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_proj = clip_model.text_proj_head(text_features)

    img_proj_n = F.normalize(img_proj, p=2, dim=-1)
    text_proj_n = F.normalize(text_proj, p=2, dim=-1)
    dot_similarity = text_proj_n @ img_proj_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n)

    return values, indices


@torch.no_grad()
def zero_shot_acc(tokenizer, clip_model, data_loader, CFG, tqdm_desc=None):
    device = next(clip_model.parameters()).device
    acc_metric = utils.MeanMetric()

    # Prompt engineering.
    classes = list(data_loader.dataset.classes)
    text = [(f'a photo of a {c}') for c in classes]

    # Get text features for different classes.
    encoded_text = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=CFG.max_length,
    )
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)

    with utils.eval_mode(clip_model):

        # Text features.
        text_features = clip_model.text_encoder(input_ids, attention_mask)
        text_proj = clip_model.text_proj_head(text_features)
        text_proj /= text_proj.norm(dim=-1, keepdim=True)

        for (img, label) in tqdm(data_loader, desc=tqdm_desc):
            img = img.to(device)
            label = label.to(device)

            # Image features.
            img_features = clip_model.img_encoder(img, interpolate_pos_encoding=CFG.interpolate_pos_encoding)
            img_proj = clip_model.img_proj_head(img_features)
            img_proj /= img_proj.norm(dim=-1, keepdim=True)

            similarity = (100.0 * img_proj @ text_proj.T).softmax(dim=-1)
            acc_metric.update_state(utils.compute_accuracy(label, similarity))

    return acc_metric.result()


# Encodes all text and images in a dataset
@torch.no_grad()
def encode_dataset(clip_model, retrieval_data_loader, CFG):

    device = next(clip_model.parameters()).device
    
    # image_to_text_map[i] gives the corresponding text indices for the i-th image.
    # (as there are multiple pieces of text for each image)
    image_to_text_map = []

    # text_to_image_map[i] gives the corresponding image index for the i-th text.
    text_to_image_map = []

    image_encodings = []
    text_encodings = []

    text_index = 0
    image_index = 0

    for img, (input_ids, attention_mask) in tqdm(retrieval_data_loader):
        img = img.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # text has shape B x 5 x 60
        batch_size, captions_per_image, _ = input_ids.shape
        # Update text_to_image_map and image_to_text_map for this batch
        for i in range(batch_size):
            # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
            text_indices = list(range(text_index, text_index + captions_per_image))
            image_to_text_map.append(text_indices)
            text_index += captions_per_image
        
            # Each of the next captions_per_image text captions correspond to the same image
            text_to_image_map += [image_index] * captions_per_image
            image_index += 1

        # B x 5 x 77 -> (B*5) x 77
        input_ids = torch.flatten(input_ids, start_dim=0, end_dim=1)
        attention_mask = torch.flatten(attention_mask, start_dim=0, end_dim=1)
        
        with utils.eval_mode(clip_model):
            img_proj, text_proj = clip_model(img, input_ids, attention_mask, interpolate_pos_encoding=CFG.interpolate_pos_encoding)
        image_encodings.append(img_proj)
        text_encodings.append(text_proj)
    
    image_encodings = torch.cat(image_encodings)
    text_encodings = torch.cat(text_encodings)
    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

    # Normalise encodings.
    image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
    text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

    return image_encodings, text_encodings, text_to_image_map, image_to_text_map


def recall_at_k(image_encodings, text_encodings, text_to_image_map, image_to_text_map, k_vals):
    device = image_encodings.device
    
    num_text = text_encodings.shape[0]
    num_img = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    # Text-to-Image recall.
    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text
    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB).
    # torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting.
    # dist_matrix = dist_matrix.cpu()
    
    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)
    
    text_to_image_recall = []
    
    for k in tqdm(k_vals):
        # Extract top k indices only
        topk = inds[:, :k]
    
        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
    
        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)


    # Image-to-Text recall.
    # dist_matrix = image_encodings @ text_encodings.T
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the i-th image.
    
    # Sort in descending order; first is the biggest logit.
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)
    
    image_to_text_recall = []
    
    for k in tqdm(k_vals):
        # Extract top k indices only
        topk = inds[:, :k]
    
        correct = torch.zeros((num_img,), dtype=torch.bool).to(device)
    
        # For each image, check whether one of the 5 relevant captions was retrieved.
        # Check if image matches its i-th caption (for i=0..4).
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)
    
        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_img)

    return text_to_image_recall, image_to_text_recall


def adjust_learning_rate(epoch, base_lr, num_epochs, warmup_epochs=0, mul=1):
    
    """Decays the learning rate with half-cycle cosine after warmup"""
    # Set mul greater than 1 to make minimum lr greater than 0.
    assert mul >= 1
    
    if epoch > num_epochs:
        return 0
    
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs 
    else:
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs * mul - warmup_epochs)))
    return lr


