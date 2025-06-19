from dataclasses import dataclass
import torch
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map_with_path, tree_map
from functools import partial

@dataclass
class GPTConfig:
    block_size: int = 1024 # maximum sequence length
    vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers (blocks)
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class CausalSelfAttention(nn.Module):
    mask_bias = nn.MultiHeadAttention.create_additive_causal_mask(GPTConfig.block_size)

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # out projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, value for all heads in a batch and move head forward
        # nh is number of heads, hs is head size, C is number of channels = nh * hs
        # eg. GPT-2 (124 M): n_head = 12, hs = 64, so nh * hs = C = 768 channels
        qkv = self.c_attn(x)
        q, k, v = qkv.split(qkv.shape[2] // self.n_embd, axis=2) # was self.embd
        q = q.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)
        k = k.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)
        v = v.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)

        # attention (materializes the large (T, T) matrix for all queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)

        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 * (q.shape[-1])**-0.5, mask=CausalSelfAttention.mask_bias)

        y = y.transpose((0, 2, 1, 3)).reshape((B, T, C)) # reassemble all head outputs
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approx='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    @staticmethod
    def _init_weights_fn(path, weights):
        if 'bias' in path:
            return nn.init.constant(0.0)(weights)
        elif any(sub in path for sub in ['wte', 'wpe', 'c_attn', 'c_fc', 'lm_head']):
            return nn.init.normal(mean=0.0, std=0.02)(weights)
        elif 'c_proj' in path:
            return nn.init.normal(mean=0.0, std=0.02 * (2 * GPTConfig.n_layer)**-0.5)(weights)
        elif "ln" in path:
            return nn.init.constant(1.0)(weights)

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = {
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': [Block(config) for _ in range(config.n_layer)],
            'ln_f': nn.LayerNorm(config.n_embd)
        }

        # init params
        self.update(tree_map_with_path(GPT._init_weights_fn, self.parameters()))

    def __call__(self, idx):
        # idx is of shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size {self.config.block_size}"
        # forward the token and position embeddings
        pos = mx.arange(0, T, dtype=mx.int64) # shape (T)
        pos_emb = self.transformer['wpe'](pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer['wte'](idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer['h']:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer['ln_f'](x)
        logits = self.transformer['wte'].as_linear(x) # (B, T, vocab_size), weight sharing scheme for self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768), # 124 M parameters
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350 M parameters
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774 M parameters
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558 M parameters
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = dict(tree_flatten(model.parameters()))
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer

        # init a huggingface transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in name and shape
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # the openai checkpoints use a Conv1D module
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                sd[k] = mx.array(sd_hf[k].t().numpy())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                sd[k] = mx.array(sd_hf[k].numpy())
        return model

    def configure_optimizers(self, weight_decay, learning_rate):
        # start with all of the candidate parameters
        param_dict = dict(tree_flatten(model.parameters()))
        # create optimizer groups. Any parameters that are 2D will be weight decayed, otherwise not.
        # ie. all weight tensors in matmuls and embeddings decay, all biases and layernorms don't.

        decay_params = [p for k, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for k, p in param_dict.items() if p.ndim < 2]
        num_decay_params = sum(p.size for p in decay_params)
        num_nodecay_params = sum(p.size for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer_w_weight_decay = optim.AdamW(learning_rate=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
        optimizer_wo_weight_decay = optim.AdamW(learning_rate=learning_rate, betas=(0.9, 0.95), eps=1e-8)

        return [optimizer_wo_weight_decay, optimizer_w_weight_decay]

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = mx.array(tokens, dtype=mx.int64)
        print(f"loaded {len(self.tokens)} tokens")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].reshape((B, T)) # inputs
        y = buf[1:].reshape((B, T)) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# ----------------
import time

#mx.set_default_device(mx.cpu)

mx.random.seed(1337)
total_batch_size = 524288 # 2**19, ~0.5 M, in number of tokens
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B, T)

#torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
mx.eval(model.parameters())

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + mx.cos(mx.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize
optimizers = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)
grad_accum_steps = mx.array(grad_accum_steps)

def loss_fn(model, x, y):
    logits = model(x)
    loss = nn.losses.cross_entropy(logits.reshape((-1, logits.shape[-1])), y.reshape((-1)), reduction='mean')
    loss /= grad_accum_steps
    return loss

@mx.compile
def split_grads(grads):
    grads = tree_flatten(grads)
    weight_grads = [(k, p) for k, p in grads if p.ndim == 2]
    bias_grads = [(k, p) for k, p in grads if p.ndim == 1]
    weight_grads = tree_unflatten(weight_grads)
    bias_grads = tree_unflatten(bias_grads)
    return weight_grads, bias_grads

@partial(mx.compile, inputs=[model.state], outputs=[model.state])
def comp_loss_and_grads(x, y):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, x, y)
    return loss, grads

state = [model.state] + [optimizer.state for optimizer in optimizers]

@partial(mx.compile, inputs=state, outputs=state)
def update(grads, lr):
    grads, norm = optim.clip_grad_norm(grads, 1.0)
    for optimizer in optimizers:
        optimizer.learning_rate = lr
    weight_grads, bias_grads = split_grads(grads)
    optimizers[0].update(model, bias_grads)
    optimizers[1].update(model, weight_grads)
    return norm

for step in range(max_steps):
    t0 = time.time()
    for micro_step in range(grad_accum_steps.item()):
        x, y = train_loader.next_batch()
        #with torch.autocast(device_type=device, dtype=torch.bfloat16):
        if micro_step == 0:
            loss, grads = comp_loss_and_grads(x, y)
        else:
            d_loss, d_grads = comp_loss_and_grads(x, y)
            loss += d_loss
            grads = tree_map(lambda grads, d_grads: grads + d_grads, grads, d_grads)
        mx.eval(loss, grads)
        while True:
                pass
        print(micro_step)

    print(grads)
    import sys; sys.exit(0)

    lr = mx.array(get_lr(step))
    norm = update(grads, lr)
    mx.eval(state)
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step} | loss: {loss.item():.6f} | lr {lr.item():.4e} | norm: {norm.item():.4f} | dt: {dt * 1000:.2f} ms | tok/s: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! Right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
