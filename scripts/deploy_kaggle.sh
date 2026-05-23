#!/bin/bash
set -e

DIR="/Users/uchebnick/projects/secretcase/repo/nellm/scripts/kaggle_kernel"
mkdir -p "$DIR"

# Copy code
cp -r /Users/uchebnick/projects/secretcase/repo/nellm/src "$DIR/"
cat << 'JSON' > "$DIR/kernel-metadata.json"
{
  "id": "uchebnick/nellm-latent-router",
  "title": "NELLM Latent Router Layer",
  "code_file": "src/models/router.py",
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

export KAGGLE_USERNAME="uchebnick"
export KAGGLE_KEY="KGAT_c1cd2a9585faea90c0731d0244113e0b"

/Users/uchebnick/Library/Python/3.13/bin/kaggle kernels push -p "$DIR"
