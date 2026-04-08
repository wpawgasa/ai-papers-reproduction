"""
Evaluation utilities (Section 6.1).

Inference settings:
- Beam search with beam_size=4
- Length penalty alpha=0.6 (Wu et al., 2016)
- Max output length = input_length + 50
- Checkpoint averaging: last 5 (base) or 20 (big) checkpoints
"""

from typing import List

import torch
import torch.nn.functional as F


def beam_search(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    bos_token: int,
    eos_token: int,
    beam_size: int = 4,
    max_len: int = 200,
    length_penalty_alpha: float = 0.6,
) -> torch.Tensor:
    """
    Beam search decoding (Section 6.1).

    "We used beam search with a beam size of 4 and length penalty
    alpha=0.6. We set the maximum output length during inference to
    input length + 50, but terminate early when possible."
    """
    device = src.device
    encoder_output = model.encode(src, src_mask)

    beams = torch.full((beam_size, 1), bos_token, dtype=torch.long, device=device)
    beam_scores = torch.zeros(beam_size, device=device)
    finished = []

    for step in range(max_len):
        tgt_mask = model.make_causal_mask(beams.size(1)).to(device)

        expanded_enc = encoder_output.expand(beam_size, -1, -1)
        expanded_src_mask = src_mask.expand(beam_size, -1, -1, -1)

        decoder_output = model.decode(
            beams, expanded_enc, expanded_src_mask, tgt_mask
        )
        logits = model.output_projection(decoder_output[:, -1, :])
        log_probs = F.log_softmax(logits, dim=-1)

        vocab_size = log_probs.size(-1)
        next_scores = beam_scores.unsqueeze(1) + log_probs
        next_scores = next_scores.view(-1)

        topk_scores, topk_indices = next_scores.topk(beam_size, dim=0)
        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size

        beams = torch.cat([beams[beam_indices], token_indices.unsqueeze(1)], dim=1)
        beam_scores = topk_scores

        for i in range(beam_size):
            if token_indices[i] == eos_token:
                length = beams[i].size(0)
                lp = ((5 + length) / 6) ** length_penalty_alpha
                finished.append((beam_scores[i] / lp, beams[i]))

        if len(finished) >= beam_size:
            break

    if finished:
        finished.sort(key=lambda x: x[0], reverse=True)
        return finished[0][1]
    return beams[0]


def greedy_decode(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    bos_token: int,
    eos_token: int,
    max_len: int = 200,
) -> torch.Tensor:
    """Simple greedy decoding for quick evaluation."""
    device = src.device
    encoder_output = model.encode(src, src_mask)

    tgt = torch.full((1, 1), bos_token, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_mask = model.make_causal_mask(tgt.size(1)).to(device)
        decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = model.output_projection(decoder_output[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == eos_token:
            break

    return tgt.squeeze(0)


def average_checkpoints(checkpoint_paths: List[str]) -> dict:
    """
    Checkpoint averaging (Section 6.1).

    "For the base models, we used a single model obtained by averaging
    the last 5 checkpoints, which were written at 10-minute intervals.
    For the big models, we averaged the last 20 checkpoints."
    """
    avg_state = None

    for path in checkpoint_paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    n = len(checkpoint_paths)
    for k in avg_state:
        avg_state[k] /= n

    return avg_state
