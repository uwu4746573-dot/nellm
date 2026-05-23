import json
import torch
from torch.utils.data import Dataset
import random
import os

class ReasoningDataset(Dataset):
    def __init__(self, json_path=None, seq_len=128, num_samples=100):
        if json_path is None:
            json_path = os.path.join(os.path.dirname(__file__), 'base_t_db.json')
        
        with open(json_path, 'r') as f:
            self.t_rules = json.load(f)
            
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_rules = len(self.t_rules)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy hidden states: [SeqLen, 4096]
        hidden_states = torch.randn(self.seq_len, 4096)
        
        # Dummy target tree: list of T-rule indices
        # We'll just generate a random sequence of indices representing the reasoning tree/path
        num_steps = random.randint(3, 10)
        target_tree = [random.randint(0, self.num_rules - 1) for _ in range(num_steps)]
        
        return hidden_states, target_tree
