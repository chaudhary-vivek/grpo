# Reasoning with GRPO

Please do not share the code with others. You are NOT AUTHORIZED to resell or distribute this code. 
Main video: https://youtu.be/LZ5xljPCJfA

## Setup

- Create conda environment conda create -n torch_env python=3.12
- Run pip install -r requirements.txt
- Make a huggingface account and set up the HF API access

## A trained model

I finetuned a 135M-Instruct model on the `syllogism` task for you to play with. Download it here:
https://drive.google.com/drive/folders/1heoDiuD-z7te3Ad9i9AW3XQSG07RIMfq?usp=drive_link

## Files shared

- **prompt_utils.py** : Contains system prompt used for training reasoning models
- **utils.py** : Generate scenarios using reasoning-gym to finetune + 
- **grpo_utils.py** : Various utility functions related to the GRPO training!
- **train_playground.ipynb** : Experiment with reasoning models! This is the notebook I showed off in the video. Remember to replace the model name at the top of the notebook with the downloaded model from above.

