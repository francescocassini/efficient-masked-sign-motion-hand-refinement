"""P3 SOKE-NAR experiment: align masked training with iterative inference.

The working SOKE-NAR model remains untouched. This training variant adds
MoMask-style BERT 80/10/10 corruption and codebook-restricted label-smoothed
cross entropy. Inference inherits the validated P1 K-step implementation.
"""
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from .mgpt_mbart_nar_p1_step_sweep import Mbart_Based_MLM_NAR_P1_StepSweep


class Mbart_Based_MLM_NAR_P3_TrainAligned(Mbart_Based_MLM_NAR_P1_StepSweep):
    """SOKE-NAR trained on masks, wrong tokens, and unchanged selected tokens."""

    def __init__(
        self,
        *args,
        nar_mask_token_prob: float = 0.8,
        nar_random_token_prob: float = 0.1,
        nar_label_smoothing: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nar_mask_token_prob = float(nar_mask_token_prob)
        self.nar_random_token_prob = float(nar_random_token_prob)
        self.nar_label_smoothing = float(nar_label_smoothing)
        if self.nar_mask_token_prob < 0 or self.nar_random_token_prob < 0:
            raise ValueError("P3 corruption probabilities must be non-negative")
        if self.nar_mask_token_prob + self.nar_random_token_prob > 1:
            raise ValueError("P3 mask + random probabilities cannot exceed one")
        if not 0 <= self.nar_label_smoothing < 1:
            raise ValueError("P3 label smoothing must be in [0, 1)")
        keep_prob = 1 - self.nar_mask_token_prob - self.nar_random_token_prob
        print(
            "[SOKE-NAR-P3] train-aligned corruption: "
            f"mask={self.nar_mask_token_prob:.3f}, "
            f"random={self.nar_random_token_prob:.3f}, keep={keep_prob:.3f}, "
            f"restricted_label_smoothing={self.nar_label_smoothing:.3f}"
        )

    def _bert_corrupt(
        self, ids: Tensor, selected: Tensor, candidates: Tensor
    ) -> Tensor:
        corrupted = ids.clone()
        draws = torch.rand(ids.shape, device=ids.device)
        use_mask = selected & (draws < self.nar_mask_token_prob)
        use_random = selected & (
            draws >= self.nar_mask_token_prob
        ) & (
            draws < self.nar_mask_token_prob + self.nar_random_token_prob
        )
        random_codes = torch.randint(
            candidates.numel(), ids.shape, device=ids.device
        )
        random_ids = candidates.to(ids.device)[random_codes]
        corrupted[use_mask] = self.mask_emb_id
        corrupted[use_random] = random_ids[use_random]
        return corrupted

    def _restricted_masked_ce(
        self,
        logits: Tensor,
        target: Tensor,
        selected: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        candidates = candidates.to(logits.device)
        restricted = logits.index_select(-1, candidates)
        target_codes = (target.unsqueeze(-1) == candidates).to(torch.long).argmax(-1)
        labels = torch.where(selected, target_codes, torch.full_like(target_codes, -100))
        return F.cross_entropy(
            restricted.reshape(-1, restricted.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.nar_label_smoothing,
        )

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
        src: List[str],
        name: List[str],
    ):
        device = motion_tokens.device
        if self.length_only_training:
            return super().forward_encdec(texts, motion_tokens, lengths, tasks, src, name)

        motion_strings = self.motion_token_to_string(
            motion_tokens[..., 0], lengths, pattern="motion"
        )
        hand_strings = self.motion_token_to_string(
            motion_tokens[..., 1], lengths, pattern="hand"
        )
        rhand_strings = self.motion_token_to_string(
            motion_tokens[..., 2], lengths, pattern="rhand"
        )

        inputs, outputs = self.template_fulfill(
            tasks, lengths, motion_strings, texts, pattern="motion"
        )
        _, outputs_hand = self.template_fulfill(
            tasks, lengths, hand_strings, texts, pattern="hand"
        )
        _, outputs_rhand = self.template_fulfill(
            tasks, lengths, rhand_strings, texts, pattern="rhand"
        )

        source_ids, source_mask = self._encode_source(inputs, src, name, device)
        body, decoder_valid = self._encode_target(outputs, src, "body", device)
        lhand, _ = self._encode_target(outputs_hand, src, "lhand", device)
        rhand, _ = self._encode_target(outputs_rhand, src, "rhand", device)
        decoder_valid = decoder_valid.bool()

        valid_motion = self._membership(body, self._body_code_emb_ids)
        selected = self._sample_motion_mask(valid_motion)
        decoder_body = self._bert_corrupt(body, selected, self._body_code_emb_ids)
        decoder_lhand = self._bert_corrupt(lhand, selected, self._lhand_code_emb_ids)
        decoder_rhand = self._bert_corrupt(rhand, selected, self._rhand_code_emb_ids)

        logits_body, logits_lhand, logits_rhand = self._nar_logits(
            source_ids,
            source_mask,
            decoder_body,
            decoder_lhand,
            decoder_rhand,
            decoder_valid,
        )
        return {
            "loss": self._restricted_masked_ce(
                logits_body, body, selected, self._body_code_emb_ids
            ),
            "loss_hand": self._restricted_masked_ce(
                logits_lhand, lhand, selected, self._lhand_code_emb_ids
            ),
            "loss_rhand": self._restricted_masked_ce(
                logits_rhand, rhand, selected, self._rhand_code_emb_ids
            ),
        }
