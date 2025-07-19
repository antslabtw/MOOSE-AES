#!/bin/bash


for prompt_id in {1..8}; do
  echo "Running fine-tuning for prompt_id $prompt_id"
  python finetune.py --test_prompt_id="$prompt_id" --tag="moose"
done
