import torch
import torch.nn as nn
import torch.nn.functional as F

# class SoftTargetCrossEntropy(nn.Module):
#     def __init__(self, class_weights: torch.Tensor = None):
#         super(SoftTargetCrossEntropy, self).__init__()
#         self.class_weights = class_weights

#     def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         log_probs = F.log_softmax(x, dim=-1)
#         if self.class_weights is not None:
#             # Expand class weights to match the batch size and number of classes
#             weights = self.class_weights.unsqueeze(0).expand_as(target)
#             loss = torch.sum(-target * log_probs * weights, dim=-1)
#         else:
#             loss = torch.sum(-target * log_probs, dim=-1)
#         return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, class_weights: torch.Tensor = None, margin: float = 1.0, window_size: int = 3):
        super(SoftTargetCrossEntropy, self).__init__()
        self.class_weights = class_weights
        self.margin = margin
        self.window_size = window_size

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate base soft cross-entropy loss
        log_probs = F.log_softmax(x, dim=-1)
        if self.class_weights is not None:
            # Expand class weights to match the batch size and number of classes
            weights = self.class_weights.unsqueeze(0).expand_as(target)
            base_loss = torch.sum(-target * log_probs * weights, dim=-1)
        else:
            base_loss = torch.sum(-target * log_probs, dim=-1)
        
        # Calculate temporal hinge loss with class weights
        #hinge_loss = self.temporal_hinge_loss(x, target)
        base_loss = base_loss[:, 8]

        # Calculate sequence-level consistency loss with class weights
        consistency_loss = self.sequence_level_consistency_loss(x, target)
        
        # Combine all losses
        total_loss = base_loss.mean() + consistency_loss

        return total_loss

    def temporal_hinge_loss(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize features to calculate cosine similarity
        x_normalized = F.normalize(x, p=2, dim=-1)
        batch_size, time_steps, num_classes = x.shape
        hinge_loss = 0

        # Compute hinge loss for each frame with respect to its window
        for t in range(time_steps):
            start = max(0, t - self.window_size)
            end = min(time_steps, t + self.window_size + 1)
            
            # Compute cosine similarity between x_t and frames within the window
            similarities = torch.sum(x_normalized[:, t, :].unsqueeze(1) * x_normalized[:, start:end, :], dim=-1)
            # Apply hinge loss on the similarities within the window
            loss_in_window = torch.clamp(self.margin - similarities, min=0)
            hinge_loss += loss_in_window.mean()
        
        # Average over batch and time steps
        hinge_loss /= (batch_size * time_steps)
        return hinge_loss

    def sequence_level_consistency_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax over classes and average over the temporal window
        softmax_preds = F.softmax(x, dim=-1)
        window_size = self.window_size
        
        # Collect probabilities within a temporal window
        avg_probs = torch.zeros_like(softmax_preds)
        
        # For each time step, average across the window
        for t in range(softmax_preds.shape[1]):  # assuming x has shape [batch_size, time_steps, num_classes]
            start = max(0, t - window_size)
            end = min(softmax_preds.shape[1], t + window_size + 1)
            avg_probs[:, t, :] = softmax_preds[:, start:end, :].mean(dim=1)
        
        # Calculate consistency loss as the weighted negative log likelihood
        consistency_loss = -torch.sum(target * torch.log(avg_probs + 1e-8), dim=-1)
        
        if self.class_weights is not None:
            weights = torch.sum(target * self.class_weights, dim=-1)
            consistency_loss = consistency_loss * weights
        
        consistency_loss = consistency_loss[:, 8]
        # Average over batch
        return consistency_loss.mean()