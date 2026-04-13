"""PhysioWave model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from wavelet_modules import SoftGateWaveletDecomp
from transformer_modules import PatchEmbed, PositionEmbedding, TransformerEncoder
from head_modules import ClassificationHead, ReconstructionHead, RegressionHead, LinearHead, MultiLabelClassificationHead


class BERTWaveletTransformer(nn.Module):
    """Wavelet-transformer backbone with task heads."""
    def __init__(self,
                 in_channels=8, 
                 max_level=3,
                 wave_kernel_size=16,
                 wavelet_names=None,
                 use_separate_channel=True,
                 patch_size=(1,20),
                 embed_dim=128,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 rope_dim=None,
                 use_pos_embed=True,
                 pos_embed_type='2d',
                 masking_strategy='frequency_guided',
                 importance_ratio=0.6,
                 mask_ratio=0.15,
                 task_type=None,
                 num_labels=None,
                 num_classes=None,
                 output_dim=None,
                 head_config=None,
                 pooling='mean',
                 use_activation_checkpointing: bool = False
                 ):
        super().__init__()
        
        self.in_channels = in_channels
        self.max_level = max_level
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_pos_embed = use_pos_embed
        self.pos_embed_type = pos_embed_type
        self.masking_strategy = masking_strategy
        self.importance_ratio = importance_ratio
        self.mask_ratio = mask_ratio
        self.task_type = task_type
        self.use_activation_checkpointing = use_activation_checkpointing
        
        self.patch_dim = patch_size[0] * patch_size[1]
        
        self.wavelet_decomp = SoftGateWaveletDecomp(
            in_channels=in_channels,
            max_level=max_level,
            kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_separate_channel,
            ffn_ratio=4.0,
            ffn_kernel_size=3,
            ffn_drop=0.1
        )
        
        self.patch_embed = PatchEmbed(
            input_channels=1,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        if use_pos_embed:
            self.pos_embed = PositionEmbedding(
                embed_dim=embed_dim,
                pos_type=pos_embed_type
            )
        else:
            self.pos_embed = None
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            rope_dim=rope_dim
        )
        
        self.task_heads = nn.ModuleDict()
        
        self.task_heads['pretrain'] = ReconstructionHead(
            embed_dim=embed_dim,
            patch_dim=self.patch_dim,
            hidden_dims=[embed_dim],
            dropout=dropout
        )
        
        if task_type == 'multilabel' and num_labels is not None:
            head_config = head_config or {}
            self.task_heads['multilabel'] = MultiLabelClassificationHead(
                embed_dim=embed_dim,
                num_labels=num_labels,
                hidden_dims=head_config.get('hidden_dims'),
                dropout=head_config.get('dropout', dropout),
                pooling=head_config.get('pooling', pooling),
                label_smoothing=head_config.get('label_smoothing', 0.0),
                use_class_weights=head_config.get('use_class_weights', False),
            )
        elif task_type == 'classification' and num_classes is not None:
            head_config = head_config or {}
            self.task_heads['classification'] = ClassificationHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                hidden_dims=head_config.get('hidden_dims'),
                dropout=head_config.get('dropout', dropout),
                pooling=head_config.get('pooling', pooling)
            )
        
        elif task_type == 'regression' and output_dim is not None:
            head_config = head_config or {}
            self.task_heads['regression'] = RegressionHead(
                embed_dim=embed_dim,
                output_dim=output_dim,
                hidden_dims=head_config.get('hidden_dims'),
                dropout=head_config.get('dropout', dropout),
                pooling=head_config.get('pooling', pooling),
                output_activation=head_config.get('output_activation')
            )
        
        elif task_type == 'linear' and output_dim is not None:
            head_config = head_config or {}
            self.task_heads['linear'] = LinearHead(
                embed_dim=embed_dim,
                output_dim=output_dim,
                pooling=head_config.get('pooling', pooling),
                use_norm=head_config.get('use_norm', False)
            )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Weight initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def initialize_weights(self):
        """Initialize special weights"""
        nn.init.normal_(self.mask_token, std=.02)
    
    def add_task_head(self, task_name, head_module):
        """Dynamically add task head"""
        self.task_heads[task_name] = head_module
    
    def frequency_guided_masking(self, tokens, mask_ratio, importance_ratio=0.6):
        """Frequency-domain importance-based masking strategy"""
        B, L, D = tokens.shape
        num_mask = int(L * mask_ratio)

        tokens_reshaped = tokens.permute(0, 2, 1)
        tokens_fft = torch.abs(torch.fft.rfft(tokens_reshaped, dim=2))
        importance_scores = torch.sum(tokens_fft, dim=1)
        
        importance_full = F.interpolate(
            importance_scores.unsqueeze(1), size=L,
            mode='linear', align_corners=True
        ).squeeze(1)

        random_noise = torch.rand(B, L, device=tokens.device)
        combined_scores = importance_ratio * importance_full + (1 - importance_ratio) * random_noise

        _, mask_indices = torch.topk(combined_scores, num_mask, dim=1)
        
        mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, mask_indices, True)
        
        return mask

    def random_masking(self, tokens, mask_ratio):
        """Random masking strategy"""
        B, L, D = tokens.shape
        num_mask = int(L * mask_ratio)
        
        mask_indices = torch.randperm(L, device=tokens.device)[:num_mask].unsqueeze(0).repeat(B, 1)
        
        mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, mask_indices, True)
        
        return mask

    def apply_masking(self, tokens, mask):
        """Apply masking: replace masked positions with [MASK] token"""
        B, L, D = tokens.shape
        
        masked_tokens = tokens.clone()
        
        mask_token_expanded = self.mask_token.expand(B, L, D)
        masked_tokens[mask] = mask_token_expanded[mask]
        
        return masked_tokens

    def patchify(self, imgs):
        """Convert images to patches"""
        B, C, F, T = imgs.shape
        p_f, p_t = self.patch_size
        assert F % p_f == 0 and T % p_t == 0
        
        f = F // p_f
        t = T // p_t
        
        x = imgs.reshape(shape=(B, C, f, p_f, t, p_t))
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(shape=(B, f * t, p_f * p_t * C))
        return x

    def prepare_tokens(self, x):
        """Prepare tokens and add position encoding"""
        B, C, F, T = x.shape
        
        tokens = self.patch_embed(x)
        _, L, D = tokens.shape
        
        if self.pos_embed is not None:
            if self.pos_embed_type == '2d':
                p_f, p_t = self.patch_size
                patches_per_freq = F // p_f
                patches_per_time = T // p_t
                tokens = self.pos_embed(tokens, freq_size=patches_per_freq, time_size=patches_per_time)
            else:
                tokens = self.pos_embed(tokens)
        
        return tokens

    def _encode_with_optional_ckpt(self, tokens):
        """Run encoder with optional activation checkpointing to save memory."""
        if self.use_activation_checkpointing:
            try:
                return _ckpt(self.encoder, tokens, use_reentrant=False)
            except TypeError:
                return _ckpt(self.encoder, tokens)
        else:
            return self.encoder(tokens)

    def forward_features(self, x):
        """Extract features (encoder part)"""
        wave_spec = self.wavelet_decomp(x)
        wave_2d = wave_spec.unsqueeze(1)
        
        tokens = self.prepare_tokens(wave_2d)
        
        features = self._encode_with_optional_ckpt(tokens)
        
        return features

    def forward_pretrain(self, x, mask_ratio=None):
        """Pretraining forward pass"""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        wave_spec = self.wavelet_decomp(x)
        wave_2d = wave_spec.unsqueeze(1)
        
        tokens = self.prepare_tokens(wave_2d)
        
        target_patches = self.patchify(wave_2d)
        
        if self.masking_strategy == 'frequency_guided':
            mask = self.frequency_guided_masking(tokens, mask_ratio, self.importance_ratio)
        else:  # 'random'
            mask = self.random_masking(tokens, mask_ratio)
        
        masked_tokens = self.apply_masking(tokens, mask)
        
        encoded_tokens = self._encode_with_optional_ckpt(masked_tokens)
        
        pred_patches = self.task_heads['pretrain'](encoded_tokens)
        
        return pred_patches, mask, target_patches

 
    def forward_downstream(self, x, task_name, **head_kwargs):   
        """Downstream task forward pass"""
        if task_name not in self.task_heads:
            raise ValueError(f"Task head '{task_name}' not found. Available: {list(self.task_heads.keys())}")
        
        features = self.forward_features(x)
        

        if task_name == 'multilabel':
            output = self.task_heads[task_name](features, **head_kwargs)
        else:
            output = self.task_heads[task_name](features)

        return output

    def forward(self, x, task='features', mask_ratio=None, task_name=None, **head_kwargs):
        """
        Unified forward pass interface
        
        Args:
            x: [B, C, T] - Input time series signal
            task: 'features', 'pretrain', 'downstream'
            mask_ratio: Masking ratio (for pretraining)
            task_name: Downstream task name
            
        Returns:
            Different results based on task
        """
        if task == 'features':
            return self.forward_features(x)
        elif task == 'pretrain':
            return self.forward_pretrain(x, mask_ratio)
        elif task == 'downstream':
            if task_name is None:
                raise ValueError("task_name must be specified for downstream tasks")
            return self.forward_downstream(x, task_name, **head_kwargs)
        else:
            if task == 'classify' and 'classification' in self.task_heads:
                return self.forward_downstream(x, 'classification')
            elif task in self.task_heads:
                return self.forward_downstream(x, task)
            else:
                raise ValueError(f"Unknown task: {task}")


def create_wavelet_classifier(in_channels=8, max_level=3, embed_dim=256, depth=8, 
                             num_heads=8, num_classes=2, **kwargs):
    """Create wavelet classifier"""
    return BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=max_level,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        task_type='classification',
        num_classes=num_classes,
        **kwargs
    )


def create_wavelet_regressor(in_channels=8, max_level=3, embed_dim=256, depth=8,
                            num_heads=8, output_dim=1, **kwargs):
    """Create wavelet regressor"""
    return BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=max_level,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        task_type='regression',
        output_dim=output_dim,
        **kwargs
    )


def create_wavelet_pretrain_model(in_channels=8, max_level=3, embed_dim=256, depth=8,
                                 num_heads=8, **kwargs):
    """Create pretraining model"""
    return BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=max_level,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        task_type='pretrain',
        **kwargs
    )
