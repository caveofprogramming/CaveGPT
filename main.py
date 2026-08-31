import os
import json
import model as m
import torch
import torch.nn.functional as F
import re
import string
import sentencepiece as spm

def save_weights(model, path="weights.pt"):
    torch.save(model.state_dict(), path)

def load_weights(model, path="weights.pt"):
    model.load_state_dict(torch.load(path))

def load_config(file):
    with open(file, 'r') as f:
        return json.load(f)

import string

import pandas as pd
import re
def load_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def train(model, loader, context_len, batch_size, steps, lr=3e-4):
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in range(steps):
        x, y = loader.get_batch(context_len, batch_size)
        logits = model(x)
        vocab_size = logits.shape[-1]
        logits = logits.view(-1, vocab_size)
        y = y.view(-1)

        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} loss {loss.item():.4f}")

def generate(model, vocab, prompt, context_len, max_new_tokens=50):
    prompt_ids = vocab.encode(prompt).tolist()
    ids = prompt_ids.copy()

    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-context_len:]])
        logits = model(x)
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)

    return vocab.decode(torch.tensor(ids))


def main():
    config = load_config('config.json')
    text = load_text(config['data'])
    vocab_size=config['vocab_size']

    spm.SentencePieceTrainer.Train(
        input=config['data'],
        model_prefix='cave',
        vocab_size=vocab_size,
        model_type='bpe'
    )

    sp = spm.SentencePieceProcessor()
    sp.load("cave.model")
    
    vocab = m.Vocab(sp)
    dataset = m.TokenDataset(text, vocab)
    loader = m.BatchLoader(dataset)

    context_len = config['context_len']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    num_layers = config['num_layers']

    model = m.TransformerModel(
        vocab_size=vocab_size,
        context_len=context_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )

    steps = config['steps']
    batch_size = config['batch_size']
    lr = config["learning_rate"]

    weights_path = "weights.pt"

    if os.path.exists(weights_path):
        print("Loading saved weights...")
        model.load(weights_path)
    else:
        print("No saved weights found — training...")
        train(model, loader, context_len, batch_size, steps, lr)
        print("Saving weights...")
        model.save(weights_path)

    print("Model ready. Enter text, or type 'quit' to exit.\n")

    while True:
        prompt = input("You: ")
        if prompt.strip().lower() in ("quit", "exit"):
            break

        response = generate(model, vocab, prompt, context_len, max_new_tokens=config['max_new_tokens'])
        print("Model:", response)
        print()

main()