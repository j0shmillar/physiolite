"""Task heads used by PhysioWave."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Classification head with configurable pooling."""
    def __init__(self, embed_dim, num_classes, hidden_dims=None, dropout=0.1, 
                 pooling='mean', activation='gelu', use_norm=True):
        super().__init__()
        self.pooling = pooling
        
        if hidden_dims is None:
            hidden_dims = [embed_dim]
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        layers = []
        prev_dim = embed_dim
        
        if use_norm:
            layers.append(nn.LayerNorm(prev_dim))
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
        
    def pool_features(self, x):
        """
        Feature pooling
        
        Args:
            x: [B, N, C] - Input feature sequence
            
        Returns:
            [B, C] - Pooled features
        """
        if self.pooling == 'mean':
            return x.mean(dim=1)
        elif self.pooling == 'max':
            return x.max(dim=1)[0]
        elif self.pooling == 'first':
            return x[:, 0]
        elif self.pooling == 'last':
            return x[:, -1]
        elif self.pooling == 'cls':
            return x[:, 0]
        elif self.pooling == 'attention':
            attention_weights = torch.softmax(x.mean(dim=-1), dim=1)  # [B, N]
            return torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [B, C]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
            
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input feature sequence
            
        Returns:
            [B, num_classes] - Classification logits
        """
        pooled = self.pool_features(x)
        
        return self.head(pooled)


class MultiLabelClassificationHead(nn.Module):
    """Multi-label classification head."""
    def __init__(self, embed_dim, num_labels, hidden_dims=None, dropout=0.1,
                 pooling='mean', activation='gelu', use_norm=True, 
                 label_smoothing=0.0, use_class_weights=False):
        super().__init__()
        self.pooling = pooling
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.use_class_weights = use_class_weights
        
        if hidden_dims is None:
            hidden_dims = [embed_dim]
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        layers = []
        prev_dim = embed_dim
        
        if use_norm:
            layers.append(nn.LayerNorm(prev_dim))
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_labels))
        
        self.head = nn.Sequential(*layers)
        
        if use_class_weights:
            self.class_weights = nn.Parameter(torch.ones(num_labels))
        else:
            self.register_buffer('class_weights', None)
        
    def pool_features(self, x):
        """Feature pooling (same as ClassificationHead)"""
        if self.pooling == 'mean':
            return x.mean(dim=1)
        elif self.pooling == 'max':
            return x.max(dim=1)[0]
        elif self.pooling == 'first':
            return x[:, 0]
        elif self.pooling == 'last':
            return x[:, -1]
        elif self.pooling == 'cls':
            return x[:, 0]
        elif self.pooling == 'attention':
            attention_weights = torch.softmax(x.mean(dim=-1), dim=1)
            return torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
            
    def forward(self, x, return_logits=False):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input feature sequence
            return_logits: If True, return logits; if False, return probabilities
            
        Returns:
            [B, num_labels] - Multi-label predictions (logits or probabilities)
        """
        pooled = self.pool_features(x)
        
        logits = self.head(pooled)
        
        if self.class_weights is not None:
            logits = logits * self.class_weights.unsqueeze(0)
        
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)
    
    def compute_loss(self, logits, targets):
        """
        Compute multi-label classification loss (Binary Cross Entropy)
        
        Args:
            logits: [B, num_labels] - Model predictions (logits)
            targets: [B, num_labels] - Ground truth binary labels
            
        Returns:
            loss: Scalar loss value
        """
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        
        return loss


class ReconstructionHead(nn.Module):
    """Patch reconstruction head."""
    def __init__(self, embed_dim, patch_dim, hidden_dims=None, dropout=0.1, 
                 activation='gelu', use_norm=True):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [embed_dim]
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        layers = []
        prev_dim = embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            if use_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, patch_dim))
        
        self.head = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Encoder output
            
        Returns:
            [B, N, patch_dim] - Reconstructed patches
        """
        return self.head(x)


class RegressionHead(nn.Module):
    """
    Regression Head
    Used for regression tasks
    """
    def __init__(self, embed_dim, output_dim=1, hidden_dims=None, dropout=0.1,
                 pooling='mean', activation='gelu', use_norm=True, output_activation=None):
        super().__init__()
        self.pooling = pooling
        self.output_activation = output_activation
        
        if hidden_dims is None:
            hidden_dims = [embed_dim]
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        layers = []
        prev_dim = embed_dim
        
        if use_norm:
            layers.append(nn.LayerNorm(prev_dim))
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.head = nn.Sequential(*layers)
        
    def pool_features(self, x):
        """Feature pooling (same as ClassificationHead)"""
        if self.pooling == 'mean':
            return x.mean(dim=1)
        elif self.pooling == 'max':
            return x.max(dim=1)[0]
        elif self.pooling == 'first':
            return x[:, 0]
        elif self.pooling == 'last':
            return x[:, -1]
        elif self.pooling == 'attention':
            attention_weights = torch.softmax(x.mean(dim=-1), dim=1)
            return torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
            
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input feature sequence
            
        Returns:
            [B, output_dim] - Regression output
        """
        pooled = self.pool_features(x)
        
        output = self.head(pooled)
        
        if self.output_activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.output_activation == 'tanh':
            output = torch.tanh(output)
        elif self.output_activation == 'relu':
            output = F.relu(output)
        
        return output


class LinearHead(nn.Module):
    """
    Simple Linear Head
    Most basic output layer containing only a linear transformation
    """
    def __init__(self, embed_dim, output_dim, pooling='mean', bias=True, use_norm=False):
        super().__init__()
        self.pooling = pooling
        
        layers = []
        if use_norm:
            layers.append(nn.LayerNorm(embed_dim))
        layers.append(nn.Linear(embed_dim, output_dim, bias=bias))
        
        self.head = nn.Sequential(*layers)
        
    def pool_features(self, x):
        """Feature pooling"""
        if self.pooling == 'mean':
            return x.mean(dim=1)
        elif self.pooling == 'max':
            return x.max(dim=1)[0]
        elif self.pooling == 'first':
            return x[:, 0]
        elif self.pooling == 'last':
            return x[:, -1]
        elif self.pooling == 'none':
            return x  # No pooling, keep sequence shape
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
            
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input feature sequence
            
        Returns:
            [B, output_dim] if pooling != 'none'
            [B, N, output_dim] if pooling == 'none'
        """
        if self.pooling == 'none':
            return self.head(x)
        else:
            pooled = self.pool_features(x)
            return self.head(pooled)


class MultiTaskHead(nn.Module):
    """
    Multi-Task Head
    Supports simultaneous output for multiple tasks
    """
    def __init__(self, embed_dim, task_configs, shared_hidden_dim=None, dropout=0.1):
        """
        Args:
            embed_dim: Input feature dimension
            task_configs: Task configuration dictionary
                Format: {
                    'task1': {'type': 'classification', 'num_classes': 2, 'pooling': 'mean'},
                    'task2': {'type': 'regression', 'output_dim': 1, 'pooling': 'max'},
                    'task3': {'type': 'multilabel', 'num_labels': 5, 'pooling': 'mean'}
                }
            shared_hidden_dim: Shared hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.task_names = list(task_configs.keys())
        
        if shared_hidden_dim:
            self.shared_transform = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, shared_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            feature_dim = shared_hidden_dim
        else:
            self.shared_transform = nn.Identity()
            feature_dim = embed_dim
        
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            task_type = config['type']
            
            if task_type == 'classification':
                self.task_heads[task_name] = ClassificationHead(
                    embed_dim=feature_dim,
                    num_classes=config['num_classes'],
                    hidden_dims=config.get('hidden_dims'),
                    dropout=config.get('dropout', dropout),
                    pooling=config.get('pooling', 'mean')
                )
            elif task_type == 'multilabel':
                self.task_heads[task_name] = MultiLabelClassificationHead(
                    embed_dim=feature_dim,
                    num_labels=config['num_labels'],
                    hidden_dims=config.get('hidden_dims'),
                    dropout=config.get('dropout', dropout),
                    pooling=config.get('pooling', 'mean'),
                    label_smoothing=config.get('label_smoothing', 0.0)
                )
            elif task_type == 'regression':
                self.task_heads[task_name] = RegressionHead(
                    embed_dim=feature_dim,
                    output_dim=config.get('output_dim', 1),
                    hidden_dims=config.get('hidden_dims'),
                    dropout=config.get('dropout', dropout),
                    pooling=config.get('pooling', 'mean'),
                    output_activation=config.get('output_activation')
                )
            elif task_type == 'linear':
                self.task_heads[task_name] = LinearHead(
                    embed_dim=feature_dim,
                    output_dim=config['output_dim'],
                    pooling=config.get('pooling', 'mean'),
                    use_norm=config.get('use_norm', False)
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input feature sequence
            
        Returns:
            dict: {task_name: task_output} - Outputs for each task
        """
        shared_features = self.shared_transform(x)
        
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_heads[task_name](shared_features)
        
        return outputs


class ContrastiveHead(nn.Module):
    """
    Contrastive Learning Head
    Used for self-supervised contrastive learning tasks
    """
    def __init__(self, embed_dim, projection_dim=128, hidden_dim=None, 
                 pooling='mean', normalize=True):
        super().__init__()
        self.pooling = pooling
        self.normalize = normalize
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def pool_features(self, x):
        """Feature pooling"""
        if self.pooling == 'mean':
            return x.mean(dim=1)
        elif self.pooling == 'max':
            return x.max(dim=1)[0]
        elif self.pooling == 'first':
            return x[:, 0]
        elif self.pooling == 'last':
            return x[:, -1]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input feature sequence
            
        Returns:
            [B, projection_dim] - Projected features
        """
        pooled = self.pool_features(x)
        
        projected = self.projector(pooled)
        
        if self.normalize:
            projected = F.normalize(projected, p=2, dim=1)
        
        return projected
