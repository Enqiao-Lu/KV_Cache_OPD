# Towards Infinite Context Windows: Neural KV Cache Compaction

Source: https://www.baseten.co/research/towards-infinite-context-windows-neural-kv-cache-compaction/

## TL;DR

Baseten's STILL proposal replaces slow per-context KV-cache optimization with a reusable neural compactor. A frozen LLM first produces the full KV cache for a context. A small perceiver-style module then compresses that cache into a much shorter cache in one forward pass. The compact cache is meant to preserve downstream answer behavior while making memory use and prefill latency scale with the compact length instead of the original context length.

## Motivation

The blog frames KV cache compaction as an intermediate memory layer between:

- the full KV cache, which is lossless but expensive
- model weights or summaries, which are much smaller but much more lossy

The central claim is that a compact but high-fidelity working memory is necessary if LLM systems are going to operate over long sessions and eventually support continual learning.

## STILL vs Earlier Methods

The two comparison points in the blog are:

- Attention Matching: strong quality, but it still does per-context optimization or fitting work at compaction time
- Cartridges: directly optimizes a compact cache for each corpus through gradient descent, which is high quality but expensive per corpus

STILL changes the cost structure. The compactor is trained once, then reused on new contexts with a single forward pass.

## Architecture

The compactor operates independently at each transformer layer:

1. Take the full key/value cache for one layer.
2. Apply inverse RoPE to the cached keys.
3. Concatenate unrotated keys and values.
4. Use a fixed set of learned latent queries to cross-attend into that sequence.
5. Let the latent set self-attend.
6. Project the latent states into compact keys, compact values, and attention biases.
7. Re-apply RoPE to the compact keys at evenly spaced latent positions.

The article emphasizes three details that made training work:

- RoPE-aware unrotate/compress/re-rotate
- removing the final normalization from the perceiver
- identity-style initialization so each latent begins by copying nearby input positions

## Training

The LLM stays frozen. Training uses KL distillation:

1. Prefill the full context through the frozen LLM.
2. Obtain teacher logits for answer tokens using the full cache.
3. Compress the full cache with STILL.
4. Obtain student logits using the compact cache.
5. Minimize KL divergence on the answer tokens.

The blog uses extractive multiple-choice questions generated from long documents, and it stresses that answer text should be on-policy for the same model being compacted.

## Reported Results

The reported headline result is about 8x compression of an 8k-token context with roughly 85% MCQ accuracy retention. The broader message is:

- more latents improve quality smoothly
- the method generalizes across domains
- iterative re-compaction is the path toward bounded-memory handling of arbitrarily long documents

## Local Relevance

For this repo, the most important operational distinction is:

- `cartridges` produces a compact cache by optimizing per corpus
- `STILL` produces a reusable checkpoint plus a fast per-corpus build step

That is the implementation contract used locally.

## Downloaded Figures

- `paper/images/figure1-inference-pipeline.png`
- `paper/images/figure2-rope.png`
- `paper/images/figure3-init.png`
- `paper/images/figure4-line-graphs-all.png`
- `paper/images/figure5-line-graphs-2.png`
- `paper/images/figure6-bar-graphs.png`
- `paper/images/figure7-line-graphs.png`
