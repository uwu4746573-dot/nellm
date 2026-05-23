#!/bin/bash
set -e

DIR="/Users/uchebnick/projects/secretcase/repo/nellm/scripts/kaggle_kernel"
rm -rf "$DIR"
mkdir -p "$DIR"

# Copy code
cp -r /Users/uchebnick/projects/secretcase/repo/nellm/src "$DIR/"
cp -r /Users/uchebnick/projects/secretcase/repo/nellm/data "$DIR/"
cp /Users/uchebnick/projects/secretcase/repo/nellm/train.py "$DIR/"

cat << 'JSON' > "$DIR/kernel-metadata.json"
{
  "id": "uwu4746573/nellm-latent-reasoning-v7",
  "title": "nellm-latent-reasoning-v7",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON

# Use the token the user gave us
export KAGGLE_USERNAME="uwu4746573"
export KAGGLE_KEY="KGAT_c1cd2a9585faea90c0731d0244113e0b"

# Deploy!
/Users/uchebnick/Library/Python/3.13/bin/kaggle kernels push -p "$DIR"
