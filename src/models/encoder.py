import torch
import torch.nn as nn
from typing import Union, List, Dict
from transformers import AutoModel, AutoTokenizer

class LatentEncoderLayer(nn.Module):
    """
    Latent Encoder Layer using an LLM encoder.
    Compresses raw text into a latent reasoning fact vector.
    """
    def __init__(self, model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct", d_model: int = 1536, d_v: int = 2048, pooling: str = "last_token"):
        super().__init__()
        self.d_model = d_model
        self.d_v = d_v
        self.pooling = pooling
        
        # Load real tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Linear projection to latent reasoning space
        self.projection = nn.Linear(d_model, d_v)

    def forward(self, inputs: Union[str, List[str], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            inputs: Raw string text, list of strings, or dictionary with tokenized input IDs (input_ids, attention_mask).
        
        Returns:
            f_1 tensor of shape [B, D_v]
        """
        # Determine the device of the module
        device = self.projection.weight.device

        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs, list):
            # Tokenize raw string
            inputs = self.tokenizer(
                inputs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
        elif isinstance(inputs, dict):
            # Move dict inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            raise ValueError("Unsupported input type. Please provide string, list of strings, or tokenized dictionary.")
            
        # Pass through the Qwen2 encoder
        outputs = self.encoder(**inputs)
        
        # Take the last hidden state of shape [B, SeqLen, D_model]
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        if self.pooling == "last_token":
            # Extract the last token (often the EOS token) for causal LMs like Qwen2
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            pooled_output = last_hidden_state[torch.arange(batch_size, device=device), sequence_lengths]
        elif self.pooling == "mean":
            # Perform mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            pooled_output = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Project to D_v
        # f_1 shape: [B, D_v]
        f_1 = self.projection(pooled_output)
        
        return f_1
