
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn.functional as F
from torch.optim import AdamW





# Load model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Store reference model (frozen copy for KL divergence)
reference_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
reference_model.eval()

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
beta = 0.1  # KL penalty coefficient

def generate_responses(model, tokenizer, prompts, max_length=100, num_responses=4):
    """Generate multiple responses for each prompt"""
    all_responses = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        responses = []
        
        for _ in range(num_responses):
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )
            
            # Decode only the generated part
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            responses.append(response)
        
        all_responses.append(responses)
    
    return all_responses

def simple_reward_function(prompt, response):
    """Simple reward: prefer shorter, coherent responses"""
    words = response.split()
    if len(words) == 0:
        return -1.0
    
    # Penalize excessive repetition
    unique_words = len(set(words))
    repetition_penalty = unique_words / len(words)
    
    # Prefer moderate length responses
    length_penalty = max(0, 1 - abs(len(words) - 20) / 50)
    
    # Simple sentiment bonus (prefer positive words)
    positive_words = ['good', 'great', 'excellent', 'wonderful', 'helpful']
    sentiment_bonus = sum(1 for word in words if word.lower() in positive_words) * 0.1
    
    return repetition_penalty + length_penalty + sentiment_bonus

def compute_grpo_loss(model, reference_model, tokenizer, prompts, responses_batch, rewards_batch, beta):
    """Compute GRPO loss for a batch"""
    total_loss = 0
    
    for prompt, responses, rewards in zip(prompts, responses_batch, rewards_batch):
        log_probs = []
        ref_log_probs = []
        
        for response in responses:
            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt", padding=True)
            
            # Get log probabilities from current model
            with torch.enable_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                logits = outputs.logits[0, :-1]  # Remove last token
                targets = inputs.input_ids[0, 1:]  # Shift targets
                
                log_prob = F.log_softmax(logits, dim=-1)
                token_log_probs = log_prob.gather(1, targets.unsqueeze(1)).squeeze()
                
                # Only consider tokens from the response part
                prompt_length = len(tokenizer(prompt)['input_ids'])
                response_log_prob = token_log_probs[prompt_length-1:].sum()
                log_probs.append(response_log_prob)
            
            # Get reference log probabilities
            with torch.no_grad():
                ref_outputs = reference_model(**inputs, labels=inputs.input_ids)
                ref_logits = ref_outputs.logits[0, :-1]
                ref_log_prob = F.log_softmax(ref_logits, dim=-1)
                ref_token_log_probs = ref_log_prob.gather(1, targets.unsqueeze(1)).squeeze()
                ref_response_log_prob = ref_token_log_probs[prompt_length-1:].sum()
                ref_log_probs.append(ref_response_log_prob)
        
        # Convert to tensors
        log_probs = torch.stack(log_probs)
        ref_log_probs = torch.stack(ref_log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # GRPO loss computation
        advantages = rewards - rewards.mean()
        log_ratios = log_probs - ref_log_probs
        kl_penalty = beta * log_ratios
        
        # GRPO objective: maximize advantage-weighted log probability with KL penalty
        grpo_loss = -(advantages * (log_probs - kl_penalty)).mean()
        total_loss += grpo_loss
    
    return total_loss / len(prompts)

def train_step(model, reference_model, tokenizer, optimizer, prompts, beta):
    """Single training step"""
    # Generate responses
    responses_batch = generate_responses(model, tokenizer, prompts)
    
    # Compute rewards
    rewards_batch = []
    for prompt, responses in zip(prompts, responses_batch):
        rewards = [simple_reward_function(prompt, response) for response in responses]
        rewards_batch.append(rewards)
    
    # Compute loss and update
    loss = compute_grpo_loss(model, reference_model, tokenizer, prompts, responses_batch, rewards_batch, beta)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item(), responses_batch, rewards_batch


# %%
training_prompts = ["How can I improve my productivity?",
        "What makes a good friend?",
        "Explain the benefits of reading books."
    ]
    
print("Starting GRPO training...")

for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")
    
    loss, responses, rewards = train_step(
        model, reference_model, tokenizer, optimizer, training_prompts, beta
    )
    
    print(f"Loss: {loss:.4f}")
    
    # Show some examples
    for i, (prompt, resp_list, reward_list) in enumerate(zip(training_prompts[:1], responses[:1], rewards[:1])):
        print(f"\nPrompt: {prompt}")
        for j, (resp, reward) in enumerate(zip(resp_list, reward_list)):
            print(f"Response {j+1} (reward: {reward:.3f}): {resp[:100]}...")

print("\nTraining completed!")


