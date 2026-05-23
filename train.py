"""
Training script for NeLL-M Latent Reasoning Pipeline.
Uses cached hidden states to save VRAM (no LLM backbone loaded).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import sys
import os
import json
if not os.path.exists("src/models/pipeline.py") and not os.path.exists("/kaggle/working/src"):
    print("Cloning GitHub repository to fetch source code...")
    os.system("git clone https://github.com/uwu4746573-dot/nellm.git /tmp/nellm")
    sys.path.append("/tmp/nellm")
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.pipeline import NeLLMReasoningPipeline

class MathDataset(Dataset):
    """Dataset loading problems and target T-rules."""
    def __init__(self, json_path: str, d_v: int = 2048):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.d_v = d_v

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        problem_text = item['problem']
        t_rules = item.get('extracted_t_rules', [])

        # Convert T-rules into a multi-hot float vector of size D_v
        # Using hash for simplicity to map strings to indices
        target_v_t = torch.zeros(self.d_v)
        if t_rules:
            for rule in t_rules:
                # Deterministic hash to keep index consistent
                rule_idx = hash(rule) % self.d_v
                target_v_t[rule_idx] = 1.0

        # Dummy target values for other losses to keep the training loop working
        target_f_new = torch.randn(self.d_v)
        target_halt = torch.randint(0, 2, (1,)).float()

        return problem_text, target_v_t, target_f_new, target_halt


class NeLLMTrainingWrapper(nn.Module):
    """
    Wrapper around NeLLMReasoningPipeline for training.
    Executes a single reasoning step and exposes intermediate outputs
    required for computing the losses.
    """
    def __init__(self, pipeline: NeLLMReasoningPipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, problem_text):
        # 1. Encode raw text into reasoning space
        # problem_text: list/tuple of strings -> f_1: [B, D_v]
        f_1 = self.pipeline.encoder(problem_text)
        
        # 2. Find relevant transformation rule
        # router returns soft Gumbel-Softmax weights [B, D_v]
        v_t = self.pipeline.router(f_1)
        
        # 3. Apply transformation
        # Synthesizer creates the new latent state
        f_new = self.pipeline.synthesizer(f_1, v_t)
        
        # 4. Evaluate step
        # Critic evaluates process reward / halt probability
        # returns raw halt_logit [B, 1]
        halt_logit = self.pipeline.critic(f_new, f_1)
        
        return v_t, f_new, halt_logit


class NTXentLoss(nn.Module):
    """
    NT-Xent loss to push apart valid and invalid F_new latent states.
    Uses InfoNCE formulation over the batch.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, f_new: torch.Tensor, target_f_new: torch.Tensor) -> torch.Tensor:
        """
        f_new: [B, D_v] (predicted latent states)
        target_f_new: [B, D_v] (target valid latent states)
        """
        # Normalize to unit vectors
        f_new_norm = F.normalize(f_new, p=2, dim=-1)
        target_f_new_norm = F.normalize(target_f_new, p=2, dim=-1)
        
        # Compute cosine similarity matrix [B, B]
        # Diagonal elements are positives (valid), off-diagonals are negatives (invalid)
        logits = torch.matmul(f_new_norm, target_f_new_norm.transpose(0, 1)) / self.temperature
        
        # Labels are the diagonal indices
        labels = torch.arange(f_new.size(0), device=f_new.device)
        return self.cross_entropy(logits, labels)


def main():
    # --- Configuration ---
    B = 2            # Batch size
    D_model = 4096   # Original LLM dimension
    D_v = 2048       # Latent reasoning dimension
    Epochs = 5       # 5 epochs on 2000 samples = ~5-10 minutes
    LearningRate = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset Setup ---
    data_path = os.path.join(os.path.dirname(__file__), "data", "processed_math.json")
    print(f"Loading dataset from {data_path}...")
    full_dataset = MathDataset(data_path, d_v=D_v)
    
    # Subset to 2000 examples for a fast ~5-10 minute test run
    subset_indices = list(range(min(2000, len(full_dataset))))
    dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    # --- Model Setup ---
    pipeline = NeLLMReasoningPipeline(d_model=D_model, d_v=D_v)
    model = NeLLMTrainingWrapper(pipeline)
    
    model.to(device)
    
    # Support for 2x T4 GPUs (Kaggle Environment)
    if torch.cuda.device_count() > 1:
        print(f"Wrapping model in DataParallel for {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=LearningRate)
    
    # --- Loss Functions ---
    # 1. Router Loss: Minimize distance to target T-rules
    criterion_router = nn.MSELoss()
    
    # 2. Contrastive Loss: NT-Xent loss for latent states
    criterion_contrastive = NTXentLoss(temperature=0.1)
    
    # 3. Critic Loss: BCEWithLogitsLoss for halt probabilities
    criterion_critic = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    print("Starting training...")
    model.train()
    
    for epoch in range(Epochs):
        total_epoch_loss = 0.0
        
        for batch_idx, (x_text, tgt_v_t, tgt_f_new, tgt_halt) in enumerate(dataloader):
            # Move to device
            tgt_v_t = tgt_v_t.to(device)
            tgt_f_new = tgt_f_new.to(device)
            tgt_halt = tgt_halt.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (automatically utilizes multiple GPUs if DataParallel is used)
            # x_text is a list/tuple of strings
            v_t, f_new, halt_logit = model(x_text)
            
            # Compute individual losses
            loss_router = criterion_router(v_t, tgt_v_t)
            loss_contrastive = criterion_contrastive(f_new, tgt_f_new)
            loss_critic = criterion_critic(halt_logit, tgt_halt)
            
            # Combine losses
            total_loss = loss_router + loss_contrastive + loss_critic
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            
            print(f"Epoch [{epoch+1}/{Epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {total_loss.item():.4f} "
                  f"(Router: {loss_router.item():.4f}, "
                  f"Contrastive: {loss_contrastive.item():.4f}, "
                  f"Critic: {loss_critic.item():.4f})")
                  
        avg_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
