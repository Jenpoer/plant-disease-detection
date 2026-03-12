import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import pandas as pd

class SupConLoss(nn.Module):
    """
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020

    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
def create_mask(labels, canonical_to_crop, canonical_to_disease, device=None):
    """
    Returns a mask for hierarchical supervised contrastive loss.
    
    Positives are:
        - same canonical label
        - OR same crop
        - OR same disease
    
    Args:
        labels: torch.Tensor of shape [batch_size] with canonical IDs
        canonical_to_crop: dict mapping canonical ID → crop ID
        canonical_to_disease: dict mapping canonical ID → disease ID
        device: torch.device
    
    Returns:
        mask: torch.Tensor of shape [batch_size, batch_size] with 1s for positives, 0s otherwise
    """
    batch_size = labels.shape[0]
    device = device or labels.device

    # Map canonical labels to crop/disease
    crop_labels = torch.tensor([canonical_to_crop[int(l)] for l in labels], device=device)
    disease_labels = torch.tensor([canonical_to_disease[int(l)] for l in labels], device=device)

    # Compare all pairs
    mask = (
        labels[:, None] == labels[None, :]          # same canonical
    ) | (
        crop_labels[:, None] == crop_labels[None, :]  # same crop
    ) | (
        disease_labels[:, None] == disease_labels[None, :]  # same disease
    )

    mask = mask.float()
    
    # mask out self-comparison if desired (optional, SupCon usually masks anchor itself)
    mask.fill_diagonal_(0)

    return mask

def get_label_mappings(label_csv_path="label_space.csv"):
    """
    Creates all mappings needed for multi-task hierarchical learning.

    Returns:
        crop_to_id
        disease_to_id
        canonical_to_crop
        canonical_to_disease
        crop_disease_to_canonical
        valid_diseases_per_crop
    """

    class_df = pd.read_csv(label_csv_path)  # canonical_id, canonical_label

    crop_to_id = {}
    disease_to_id = {}
    crop_ids = []
    disease_ids = []

    # ------------------------------------
    # Build crop and disease ID mappings
    # ------------------------------------
    for _, row in class_df.iterrows():
        crop, disease = row["canonical_label"].split("__")

        if crop not in crop_to_id:
            crop_to_id[crop] = len(crop_to_id)

        if disease not in disease_to_id:
            disease_to_id[disease] = len(disease_to_id)

        crop_ids.append(crop_to_id[crop])
        disease_ids.append(disease_to_id[disease])

    # ------------------------------------
    # Canonical → crop/disease
    # ------------------------------------
    canonical_to_crop = dict(zip(class_df["canonical_id"], crop_ids))
    canonical_to_disease = dict(zip(class_df["canonical_id"], disease_ids))

    # ------------------------------------
    # (crop, disease) → canonical
    # ------------------------------------
    crop_disease_to_canonical = {}

    for cid in canonical_to_crop:
        crop = canonical_to_crop[cid]
        disease = canonical_to_disease[cid]
        crop_disease_to_canonical[(crop, disease)] = cid

    # ------------------------------------
    # Valid diseases per crop (for masking)
    # ------------------------------------
    valid_diseases_per_crop = {crop_id: set() for crop_id in crop_to_id.values()}

    for cid in canonical_to_crop:
        crop = canonical_to_crop[cid]
        disease = canonical_to_disease[cid]
        valid_diseases_per_crop[crop].add(disease)

    return (
        crop_to_id,
        disease_to_id,
        canonical_to_crop,
        canonical_to_disease,
        crop_disease_to_canonical,
        valid_diseases_per_crop,
    )
    
class SupConViT(nn.Module):
    def __init__(self, backbone, backbone_name, proj_dim=128, num_classes=26):
        super().__init__()

        self.backbone = backbone  # pretrained ViT
        self.backbone_name = backbone_name

        # embed_dim = backbone.embed_dim  # usually 768

        # Projection head (for contrastive learning)
        # self.projection = nn.Sequential(
        #     nn.Linear(embed_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, proj_dim)
        # )

    def forward(self, x):
        features = self.backbone.forward_features(x)  # (B, 197, 768)
        
        pooled = self.backbone.forward_head(features, pre_logits=True)

        proj = F.normalize(pooled, dim=1)

        # Classification logits
        logits = self.backbone.forward_head(features)

        return proj, logits
    
class TwoCropTransform:
    def __init__(self, base_transform):
        self.transform = transforms.Compose(
                            [transforms.RandomResizedCrop(224)] +
                            list(base_transform.transforms)
                        )

    def __call__(self, x):
        return [
            self.transform(x),
            self.transform(x)
        ]