from __future__ import annotations

from typing import Any

import torch


def _append_token(batch: dict[str, Any], token: torch.Tensor) -> dict[str, Any]:
    updated = dict(batch)
    updated["input_ids"] = torch.cat([batch["input_ids"], token[:, None]], dim=1)
    updated["attention_mask"] = torch.cat(
        [batch["attention_mask"], torch.ones_like(token[:, None])], dim=1
    )
    updated["loss_mask"] = torch.cat(
        [batch["loss_mask"], torch.zeros_like(token[:, None], dtype=torch.bool)], dim=1
    )
    return updated


def _backbone_next(model, batch: dict[str, Any]) -> torch.Tensor:
    hidden, _ = model._backbone_hidden(batch)
    return model.thinker.lm_head(hidden[:, -1]).argmax(dim=-1)


def _draft_next(model, batch: dict[str, Any], depth: int, base_position: int) -> torch.Tensor:
    hidden, position_ids = model._backbone_hidden(batch)
    token_embedding = model.thinker.get_input_embeddings()
    for branch_index in range(1, depth + 1):
        hidden = model.branches[branch_index - 1](
            previous_hidden=hidden[:, :-1],
            shifted_embedding=token_embedding(batch["input_ids"][:, branch_index:]),
            attention_mask=batch["attention_mask"],
            position_ids=position_ids,
            text_model=model.thinker.model,
            position_offset=(
                branch_index if model.branch_position_mode == "shifted" else 0
            ),
        )
    normalized = model.thinker.model.norm(hidden[:, base_position])
    return model.thinker.lm_head(normalized).argmax(dim=-1)


@torch.no_grad()
def speculative_greedy_reference(
    model,
    prefix_batch: dict[str, Any],
    eos_token_id: int | None,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Slow batch-size-one reference loop for validating offline acceptance metrics."""
    if prefix_batch["input_ids"].shape[0] != 1:
        raise ValueError("reference verifier requires batch size 1")
    context = prefix_batch
    accepted_lengths = []
    generated = []
    while len(generated) < max_new_tokens:
        base = _backbone_next(model, context)
        context = _append_token(context, base)
        generated.append(int(base[0]))
        accepted = 1
        if eos_token_id is not None and generated[-1] == eos_token_id:
            accepted_lengths.append(accepted)
            break

        base_position = context["input_ids"].shape[1] - 2
        draft_context = context
        drafts = []
        for depth in range(1, model.depth + 1):
            proposal = _draft_next(model, draft_context, depth, base_position)
            drafts.append(proposal)
            draft_context = _append_token(draft_context, proposal)

        verify_context = context
        verifier_tokens = []
        for proposal in drafts:
            verifier_tokens.append(_backbone_next(model, verify_context))
            verify_context = _append_token(verify_context, proposal)

        rejected = False
        for proposal, verifier in zip(drafts, verifier_tokens):
            if bool(proposal.eq(verifier).all()):
                context = _append_token(context, proposal)
                generated.append(int(proposal[0]))
                accepted += 1
                if eos_token_id is not None and generated[-1] == eos_token_id:
                    break
                if len(generated) >= max_new_tokens:
                    break
            else:
                context = _append_token(context, verifier)
                generated.append(int(verifier[0]))
                rejected = True
                break
        accepted_lengths.append(accepted)
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break
        if not rejected and len(generated) >= max_new_tokens:
            break
    return {
        "generated_token_ids": generated[:max_new_tokens],
        "accepted_lengths": accepted_lengths,
        "average_accepted_length": (
            sum(accepted_lengths) / len(accepted_lengths) if accepted_lengths else 0.0
        ),
    }
