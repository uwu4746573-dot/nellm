import sys
import json
import torch
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from nellm.src.models.pipeline import NeLLMReasoningPipeline

def main():
    # Instantiate the pipeline
    # Using realistic dimensions as per the pipeline's defaults but we can keep it small for quick execution
    # or just use defaults to be safe. We'll use smaller ones to avoid memory/time issues if just simulating,
    # but let's use the defaults to be faithful, it's just one forward pass.
    pipeline = NeLLMReasoningPipeline(
        d_model=4096, 
        d_v=2048, 
        d_k=128, 
        n_t=64, 
        beam_width=1, 
        prefix_len=16
    )
    pipeline.eval()
    
    # Load JSON
    db_path = Path(__file__).resolve().parent.parent / "data" / "base_t_db.json"
    with open(db_path, "r", encoding="utf-8") as f:
        t_db = json.load(f)
        
    t_map = {item["id"]: item["name"] for item in t_db}
    
    # Mock input from a base LLM (B=1, SeqLen=10, D_model=4096)
    torch.manual_seed(42)  # For reproducible output
    B = 1
    D_model = 4096
    hidden_states = torch.randn(B, 10, D_model)
    
    # 1. Encode into reasoning space
    f_1 = pipeline.encoder(hidden_states) # [B, D_v]
    f_current = f_1
    
    max_steps = 3
    
    print("[F_1] Initial state")
    
    indent = " "
    
    for step in range(max_steps):
        # We simulate the exact logic of the router to find the chosen T-rule index
        Q = pipeline.router.query_proj(f_current)
        logits = torch.matmul(Q, pipeline.router.K_T.t()) / (pipeline.router.d_k ** 0.5)
        top1_index = torch.argmax(logits, dim=-1).item()
        
        rule_name = t_map.get(top1_index, "Unknown Rule")
        
        # Step the actual pipeline
        v_t = pipeline.router(f_current)
        if not pipeline.training:
            # We take the top branch (greedy path)
            v_t = v_t[:, 0, :]
            
        f_new = pipeline.synthesizer(f_current, v_t)
        
        # Critic evaluates if reasoning is complete
        # Critic returns a raw logit, so we apply sigmoid to display probability
        halt_logit = pipeline.critic(f_new, f_1).item()
        halt_prob = torch.sigmoid(torch.tensor(halt_logit)).item()
        
        f_current = f_new
        
        print(f"{indent}├── T_{top1_index}: {rule_name}")
        
        # Pipeline breaks if logit > 0.5 (which is what pipeline.critic returns directly)
        if halt_logit > 0.5 or step == max_steps - 1:
            print(f"{indent}│    └── [F_{step+2}] -> Critic: Halt ({halt_prob:.2f}) -> Answer")
            break
        else:
            print(f"{indent}│    └── [F_{step+2}]")
            indent += "│         "

if __name__ == "__main__":
    main()
