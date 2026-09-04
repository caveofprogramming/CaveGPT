
import torch

class Vocab:
    def __init__(self, sp):
        self.sp = sp
        self.vocab_size = sp.get_piece_size()

    def encode(self, text):
        ids = self.sp.encode(text, out_type=int)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        return self.sp.decode(ids.tolist())

class TokenDataset:
    def __init__(self, text, vocab: Vocab):
        self.vocab = vocab
        self.ids = vocab.encode(text)

class BatchLoader:
    def __init__(self, dataset: TokenDataset):
        self.ids = dataset.ids

    def get_batch(self, context_length, batch_size):
        ids = self.ids
        max_start = len(ids) - context_length - 1

        ix = torch.randint(0, max_start, (batch_size,))

        x = torch.stack([ids[i : i + context_length] for i in ix])
        y = torch.stack([ids[i + 1 : i + context_length + 1] for i in ix])

        return x, y

    
class TokenEmbedding:
    def __init__(self, vocab_size, embed_dim):
        weight = torch.randn(vocab_size, embed_dim)
        self.weight = torch.nn.Parameter(weight / embed_dim ** 0.5)

    def __call__(self, x):
        return self.weight[x]

    def parameters(self):
        return [self.weight]

class PositionalEmbedding:
    def __init__(self, context_length, embed_dim):
        weight = torch.randn(context_length, embed_dim)
        self.weight = torch.nn.Parameter(weight / embed_dim ** 0.5)

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
        Q = x @ self.Wq # query
        K = x @ self.Wk # key
        V = x @ self.Wv # value

        scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)
        mask = torch.tril(torch.ones(scores.shape))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        return weights @ V

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.heads = [AttentionHead(embed_dim, self.head_dim) for _ in range(num_heads)]
        weights = torch.randn(embed_dim, embed_dim)
        self.Wo = torch.nn.Parameter(weights / embed_dim ** 0.5)

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
        w1 = torch.randn(embed_dim, hidden) / embed_dim ** 0.5
        self.W1 = torch.nn.Parameter(w1)
        w2 = torch.randn(hidden, embed_dim) / embed_dim ** 0.5
        self.W2 = torch.nn.Parameter(w2)

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
        weightsOut = torch.randn(embed_dim, vocab_size) 
        self.Wout = torch.nn.Parameter(weightsOut / embed_dim ** 0.5)

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




    