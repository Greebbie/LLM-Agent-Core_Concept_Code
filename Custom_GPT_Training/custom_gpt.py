"""
Custom GPT Model - A Small, CPU-Friendly GPT Implementation

This module provides a complete GPT implementation from scratch,
designed to be small enough to train on consumer hardware (CPU or low-end GPU).

Architecture:
- Vocabulary size: 5000 (small, efficient tokenizer)
- Model dimension: 384
- Attention heads: 6
- Layers: 6
- FFN dimension: 1536
- Max sequence length: 256
- Total parameters: ~12.6M

Usage:
    from custom_gpt import CustomGPT, SimpleTokenizer, GPTConfig

    config = GPTConfig()
    model = CustomGPT(config)
    tokenizer = SimpleTokenizer(vocab_size=5000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, asdict
import pickle


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GPTConfig:
    """Configuration for Custom GPT Model."""
    vocab_size: int = 5000
    max_seq_len: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Training settings
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'GPTConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls(**json.load(f))

    @property
    def num_params(self) -> int:
        """Estimate number of parameters."""
        # Embedding: vocab_size * d_model
        embed_params = self.vocab_size * self.d_model
        # Position embedding: max_seq_len * d_model
        pos_params = self.max_seq_len * self.d_model
        # Each layer: attention (4 * d_model^2) + ffn (2 * d_model * d_ff) + norms
        layer_params = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff + 4 * self.d_model
        total_layer_params = self.n_layers * layer_params
        # Output projection: d_model * vocab_size (shared with embedding)
        return embed_params + pos_params + total_layer_params


# =============================================================================
# Simple Tokenizer (Character/Word Level)
# =============================================================================

class SimpleTokenizer:
    """
    Simple tokenizer that supports both character-level and word-level tokenization.

    Features:
    - Builds vocabulary from text
    - Supports save/load
    - Handles unknown tokens
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        mode: str = "word",  # "char" or "word"
        min_freq: int = 2
    ):
        self.vocab_size = vocab_size
        self.mode = mode
        self.min_freq = min_freq

        # Special tokens
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }

        self.token_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        if self.mode == "char":
            return list(text)
        else:
            # Simple word tokenization
            text = text.lower()
            # Split on whitespace and punctuation
            tokens = []
            current = ""
            for char in text:
                if char.isalnum():
                    current += char
                else:
                    if current:
                        tokens.append(current)
                        current = ""
                    if char.strip():
                        tokens.append(char)
            if current:
                tokens.append(current)
            return tokens

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts."""
        # Count token frequencies
        freq = {}
        for text in texts:
            for token in self._tokenize(text):
                freq[token] = freq.get(token, 0) + 1

        # Filter by minimum frequency
        freq = {k: v for k, v in freq.items() if v >= self.min_freq}

        # Sort by frequency and take top vocab_size - special_tokens
        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
        max_tokens = self.vocab_size - len(self.special_tokens)

        # Build mappings
        for token, _ in sorted_tokens[:max_tokens]:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        print(f"Vocabulary built: {len(self.token_to_id)} tokens")

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False
    ) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            elif padding and len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, "<UNK>")
            if skip_special and token in self.special_tokens:
                continue
            tokens.append(token)

        if self.mode == "char":
            return "".join(tokens)
        else:
            return " ".join(tokens)

    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "vocab_size": self.vocab_size,
            "mode": self.mode,
            "min_freq": self.min_freq,
            "token_to_id": self.token_to_id,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            mode=data["mode"],
            min_freq=data["min_freq"]
        )
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {v: k for k, v in data["token_to_id"].items()}
        return tokenizer

    def __len__(self):
        return len(self.token_to_id)


# =============================================================================
# Model Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin
        self._precompute_cos_sin()

    def _precompute_cos_sin(self):
        t = torch.arange(self.max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K."""
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional RoPE."""

    def __init__(self, config: GPTConfig, use_rope: bool = True):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rope:
            cos, sin = self.rope(x, seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # Attention mask (for padding)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-Forward Network with SwiGLU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (x * W1) * SiLU(x * W3)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =============================================================================
# Main Model
# =============================================================================

class CustomGPT(nn.Module):
    """
    Custom GPT Model - Small, CPU-friendly implementation.

    Features:
    - RoPE (Rotary Position Embedding)
    - RMSNorm (Root Mean Square Normalization)
    - SwiGLU activation
    - Pre-norm architecture
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.d_model, config.layer_norm_eps)

        # Output projection (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 = attend, 0 = mask)
            labels: [batch_size, seq_len] for computing loss

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape

        # Convert attention mask to additive mask for attention
        if attention_mask is not None:
            # [batch, seq] -> [batch, 1, 1, seq]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final norm and projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: Token ID to stop generation

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Truncate if too long
            if input_ids.shape[1] >= self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self(input_ids)
            logits = outputs["logits"][:, -1, :]  # [batch, vocab]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)

            # Apply top-p (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample or greedy
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return input_ids

    def save_pretrained(self, path: str):
        """Save model and config."""
        os.makedirs(path, exist_ok=True)

        # Save config
        self.config.save(os.path.join(path, "config.json"))

        # Save model weights
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

        print(f"Model saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> 'CustomGPT':
        """Load model from saved files."""
        # Load config
        config = GPTConfig.load(os.path.join(path, "config.json"))

        # Create model
        model = cls(config)

        # Load weights
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))

        print(f"Model loaded from {path}")
        return model

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> str:
    """Return formatted parameter count."""
    n_params = sum(p.numel() for p in model.parameters())
    if n_params >= 1e9:
        return f"{n_params / 1e9:.2f}B"
    elif n_params >= 1e6:
        return f"{n_params / 1e6:.2f}M"
    elif n_params >= 1e3:
        return f"{n_params / 1e3:.2f}K"
    return str(n_params)


def get_model_summary(model: CustomGPT) -> str:
    """Get a summary of the model architecture."""
    config = model.config
    summary = f"""
Custom GPT Model Summary
========================
Vocabulary Size: {config.vocab_size:,}
Max Sequence Length: {config.max_seq_len}
Model Dimension: {config.d_model}
Attention Heads: {config.n_heads}
Layers: {config.n_layers}
FFN Dimension: {config.d_ff}
Dropout: {config.dropout}

Total Parameters: {count_parameters(model)}
Trainable Parameters: {model.num_parameters(trainable_only=True):,}
"""
    return summary


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Create model
    config = GPTConfig()
    model = CustomGPT(config)

    print(get_model_summary(model))

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids)
    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")

    # Test with labels (training mode)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs = model(input_ids, labels=labels)
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\nGeneration:")
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
