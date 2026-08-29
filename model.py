import torch

class Vocab:
    def __init__(self, text):

        # Sorted list of all characters in the text.
        chars = sorted(set(text))

        # Create a mapping from chars to integer IDs.
        self.stoi = {c:i for i, c in enumerate(chars)}

        # Create a mapping from the int ID back to the original char.
        self.itos = {i:c for c, i in self.stoi.items()}

        # How many unique chars do we have?
        self.vocab_size = len(chars)

    def encode(self, text):
        # Take some text and turn it into a vector of integer IDs,
        # where each ID corresponds to a character.
        return torch.tensor([self.stoi[c] for c in text])
    
    def decode(self, ids):
        # Translate vector (tensor) of IDs to text.
        # Tensor contains rank-0 scaler tensors, so
        # i.item() gets the actual integer.
        return ''.join(self.itos[i.item()] for i in ids)

class CharDataset:
    def __init__(self, text, vocab: Vocab):

        # Store the vocab used to map characters to ints
        self.vocab = vocab

        # Store the entire encoded text
        self.ids = vocab.encode(text)

class BatchLoader:
    def __init__(self, dataset: CharDataset):

        # ids is the entire encoded text, as int IDs instead of characters
        self.ids = dataset.ids

    def get_batch(self, context_length, batch_size):
        ids = self.ids # entire encoded text

        # vector of random ints, of length batch_size.
        # Maximum int is big enough that if we use it as an index into ids 
        # (the encoded text), we can read context_length characters from that point
        ix = torch.randint(len(ids) - context_length, (batch_size, ))

        # x consists of sequences of characters of context_length, 
        # randomly selected using ix.
        x = torch.stack([ids[i : i + context_length] for i in ix])

        # y contains same as corresponding x sequences, but shifted along 1 character
        y = torch.stack([ids[i + 1 : i + context_length + 1] for i in ix])
        
        return x, y
    
class TokenEmbedding:
    def __init__(self, vocab_size, embed_dim):
        # Divide weights by square root of embed_dim to keep constant magnitude of vector
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, embed_dim) / embed_dim ** 0.5)

    def __call__(self, x):
        return self.weight[x]

    def parameters(self):
        return [self.weight]

class PositionalEmbedding:
    def __init__(self, context_length, embed_dim):
        # Divide weights by square root of embed_dim to keep constant magnitude of vector
        self.weight = torch.nn.Parameter(torch.randn(context_length, embed_dim) / embed_dim ** 0.5)

    def parameters(self):
        return [self.weight]

    def __call__(self, x):
        return self.weight[x]

class AttentionHead:
    def __init__(self, embed_dim, head_dim):
        self.Wq = torch.nn.Parameter(torch.randn(embed_dim, head_dim) / embed_dim ** 0.5)
        self.Wk = torch.nn.Parameter(torch.randn(embed_dim, head_dim) / embed_dim ** 0.5)
        self.Wv = torch.nn.Parameter(torch.randn(embed_dim, head_dim) / embed_dim ** 0.5)

    def parameters(self):
        return [self.Wq, self.Wk, self.Wv]

    def __call__(self, x):
        # x is a tensor where each row represents a character in a sequence
        # of characters extracted from the text.
        # Each character is represented as a vector of floats of embed_dim length.

        # Imagine we only have one token. The output matrices below would only 
        # contain one row. Each row is a different projection of the original
        # token, representing different things.

        Q = x @ self.Wq # Q = query, what this token wants
        K = x @ self.Wk # K = key, what this token offers
        V = x @ self.Wv # V = value, what this token gives if someone attends to it

        # This is computing a matrix where each row is the dot product
        # of a row in Q with a row in K.
        # So we're computing similarity scores between all possible 
        # combinations of 2 rows.
        # These grow in proportion to the square of the number of columns
        # hence the normalising factor.
        scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)

        # Causal mask. A mask of 1s in the same shape as scores,
        # except upper part above diagonal is zeros.
        mask = torch.tril(torch.ones(scores.shape))

        # Apply mask. Replace all values in scores in same positions as 0
        # in mask with -inf, so that softmax will ignore these values.
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to each row (index -1)
        weights = torch.softmax(scores, dim=-1)

        # Finally apply V
        return weights @ V

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.heads = [AttentionHead(embed_dim, self.head_dim) for _ in range(num_heads)]
        self.Wo = torch.nn.Parameter(torch.randn(embed_dim, embed_dim) / embed_dim ** 0.5)

    def parameters(self):
        params = [self.Wo]

        for h in self.heads:
            params += h.parameters()

        return params

    def __call__(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return out @ self.Wo

class FeedForward:
    def __init__(self, embed_dim):
        hidden = 4 * embed_dim
        self.W1 = torch.nn.Parameter(torch.randn(embed_dim, hidden) / embed_dim ** 0.5)
        self.W2 = torch.nn.Parameter(torch.randn(hidden, embed_dim) / embed_dim ** 0.5)

    def parameters(self):
        return [self.W1, self.W2]

    def __call__(self, x):
        return torch.relu(x @ self.W1) @ self.W2
    
class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(embed_dim))
        self.beta = torch.nn.Parameter(torch.zeros(embed_dim))

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class TransformerBlock:
    def __init__(self, embed_dim, num_heads):
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim)

    def __call__(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def parameters(self):
        return (
            self.ln1.parameters() +
            self.ln2.parameters() +
            self.mha.parameters() +
            self.ff.parameters()
        )

class TransformerModel:
    def __init__(self, vocab_size, context_len, embed_dim, num_heads, num_layers):
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEmbedding(context_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        self.Wout = torch.nn.Parameter(torch.randn(embed_dim, vocab_size) / embed_dim ** 0.5)

    def parameters(self):
        params = []
        params += self.token_embed.parameters()
        params += self.pos_embed.parameters()
        params += [self.Wout]

        for block in self.blocks:
            params += block.parameters()

        return params

    def save(self, path="weights.pt"):
        tensors = [p.detach().clone() for p in self.parameters()]
        torch.save(tensors, path)

    def load(self, path="weights.pt"):
        tensors = torch.load(path)
        params = self.parameters()

        if len(tensors) != len(params):
            raise ValueError("Saved weights do not match model parameters")

        for p, t in zip(params, tensors):
            p.data.copy_(t)

    def __call__(self, x):
        tok = self.token_embed(x)
        pos = self.pos_embed(torch.arange(x.shape[1]))
        h = tok + pos

        for block in self.blocks:
            h = block(h)

        logits = h @ self.Wout
        return logits