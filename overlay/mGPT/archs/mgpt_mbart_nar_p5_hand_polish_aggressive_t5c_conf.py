"""T5-C cache variant of P5 aggressive that also returns confidence traces.

This file intentionally leaves the working P5 implementation untouched.  It
copies the P5 aggressive generation logic and only extends the returned
dictionary with final body/LH/RH confidence tensors for residual predictors.
"""

import math
from typing import List, Optional

import torch

from .mgpt_mbart_nar_p5_hand_polish_aggressive import (
    Mbart_Based_MLM_NAR_P5_HandPolishAggressive,
)


class Mbart_Based_MLM_NAR_P5_HandPolishAggressive_T5CConf(
    Mbart_Based_MLM_NAR_P5_HandPolishAggressive
):
    """P5 aggressive with extra confidence outputs for T5-C caches."""

    @torch.no_grad()
    def generate_direct(
        self,
        texts: List[str],
        max_length: int = 256,
        num_beams: int = 1,
        do_sample: bool = True,
        bad_words_ids: List[int] = None,
        src: List[str] = None,
        name: List[str] = None,
        lengths: Optional[List[int]] = None,
    ):
        del max_length, num_beams, do_sample, bad_words_ids
        device = self.language_model.main_lm.device
        if lengths is None and not self.length_predictor:
            raise ValueError("P5 requires lengths or an enabled length predictor")

        source_ids, source_mask = self._encode_source(texts, src, name, device)
        length_logits = None
        if self.length_predictor:
            if self.length_predictor_kind == "linear_text_units":
                token_lengths = self._linear_text_token_lengths(texts, src)
            else:
                token_lengths, length_logits = self._predict_token_lengths(
                    source_ids, source_mask, src
                )
        else:
            token_lengths = torch.tensor(
                [max(1, int(round(float(length) / self.down_t))) for length in lengths],
                dtype=torch.long,
                device=device,
            )
            token_lengths = (
                token_lengths.float() * self.oracle_length_scale
            ).round().long().clamp_min(1)

        total = int(token_lengths.max()) + 2
        positions = torch.arange(total, device=device).unsqueeze(0)
        motion_valid = positions < token_lengths.unsqueeze(1)
        decoder_valid = positions < (token_lengths + 2).unsqueeze(1)

        body = torch.full(
            (len(texts), total), self.pad_emb_id, dtype=torch.long, device=device
        )
        lhand = body.clone()
        rhand = body.clone()
        body[motion_valid] = self.mask_emb_id
        lhand[motion_valid] = self.mask_emb_id
        rhand[motion_valid] = self.mask_emb_id

        batch = torch.arange(len(texts), device=device)
        body[batch, token_lengths] = self.eos_emb_id
        lhand[batch, token_lengths] = self.eos_emb_id
        rhand[batch, token_lengths] = self.eos_emb_id
        body[batch, token_lengths + 1] = self._lang_embedding_ids(src, "body", device)
        lhand[batch, token_lengths + 1] = self._lang_embedding_ids(src, "lhand", device)
        rhand[batch, token_lengths + 1] = self._lang_embedding_ids(src, "rhand", device)

        remask = motion_valid.clone()
        body_codes = torch.zeros_like(body)
        lhand_codes = torch.zeros_like(body)
        rhand_codes = torch.zeros_like(body)

        effective_source_mask = None
        encoder_hidden_states = self.language_model.main_lm.get_encoder()(
            input_ids=source_ids,
            attention_mask=effective_source_mask,
        )[0]

        conf_b = conf_l = conf_r = None
        p3_remask_counts = []
        for step in range(self.nar_steps):
            logits_body, logits_lhand, logits_rhand = self._nar_logits_from_cached_encoder(
                encoder_hidden_states,
                effective_source_mask,
                body,
                lhand,
                rhand,
                decoder_valid,
            )
            pred_b, code_b, conf_b = self._predict_codes(
                logits_body, self._body_code_emb_ids.to(device)
            )
            pred_l, code_l, conf_l = self._predict_codes(
                logits_lhand, self._lhand_code_emb_ids.to(device)
            )
            pred_r, code_r, conf_r = self._predict_codes(
                logits_rhand, self._rhand_code_emb_ids.to(device)
            )
            body = torch.where(remask, pred_b, body)
            lhand = torch.where(remask, pred_l, lhand)
            rhand = torch.where(remask, pred_r, rhand)
            body_codes = torch.where(remask, code_b, body_codes)
            lhand_codes = torch.where(remask, code_l, lhand_codes)
            rhand_codes = torch.where(remask, code_r, rhand_codes)

            confidence = (conf_b + conf_l + conf_r) / 3.0
            confidence = confidence.masked_fill(~motion_valid, float("inf"))
            ratio = math.cos((step + 1) / self.nar_steps * math.pi / 2)
            next_remask = torch.zeros_like(remask)
            for index, length in enumerate(token_lengths.tolist()):
                count = int(math.floor(length * ratio))
                if count > 0:
                    low = confidence[index, :length].topk(count, largest=False).indices
                    next_remask[index, low] = True
            p3_remask_counts.append(next_remask[:, :total].sum(dim=1).detach())
            body[next_remask] = self.mask_emb_id
            lhand[next_remask] = self.mask_emb_id
            rhand[next_remask] = self.mask_emb_id
            remask = next_remask

        polish_l_counts = []
        polish_r_counts = []
        for ratio in self.hand_polish_ratios:
            remask_l = self._lowest_confidence_mask(
                conf_l, motion_valid, token_lengths, ratio
            )
            remask_r = self._lowest_confidence_mask(
                conf_r, motion_valid, token_lengths, ratio
            )
            polish_l_counts.append(remask_l[:, :total].sum(dim=1).detach())
            polish_r_counts.append(remask_r[:, :total].sum(dim=1).detach())
            lhand[remask_l] = self.mask_emb_id
            rhand[remask_r] = self.mask_emb_id

            _, logits_lhand, logits_rhand = self._nar_logits_from_cached_encoder(
                encoder_hidden_states,
                effective_source_mask,
                body,
                lhand,
                rhand,
                decoder_valid,
            )
            pred_l, code_l, conf_l = self._predict_greedy_codes(
                logits_lhand, self._lhand_code_emb_ids.to(device)
            )
            pred_r, code_r, conf_r = self._predict_greedy_codes(
                logits_rhand, self._rhand_code_emb_ids.to(device)
            )
            lhand = torch.where(remask_l, pred_l, lhand)
            rhand = torch.where(remask_r, pred_r, rhand)
            lhand_codes = torch.where(remask_l, code_l, lhand_codes)
            rhand_codes = torch.where(remask_r, code_r, rhand_codes)

        return {
            "outputs_tokens": [
                body_codes[index, :length].detach()
                for index, length in enumerate(token_lengths)
            ],
            "cleaned_text": [""] * len(texts),
            "outputs_tokens_hand": [
                lhand_codes[index, :length].detach()
                for index, length in enumerate(token_lengths)
            ],
            "cleaned_text_hand": [""] * len(texts),
            "outputs_tokens_rhand": [
                rhand_codes[index, :length].detach()
                for index, length in enumerate(token_lengths)
            ],
            "cleaned_text_rhand": [""] * len(texts),
            "predicted_token_lengths": token_lengths.detach(),
            "length_logits": None if length_logits is None else length_logits.detach(),
            "confidence_body": [
                conf_b[index, :length].detach()
                for index, length in enumerate(token_lengths)
            ],
            "confidence_lhand": [
                conf_l[index, :length].detach()
                for index, length in enumerate(token_lengths)
            ],
            "confidence_rhand": [
                conf_r[index, :length].detach()
                for index, length in enumerate(token_lengths)
            ],
            "p3_remask_counts": torch.stack(p3_remask_counts, dim=1).detach()
            if p3_remask_counts
            else None,
            "polish_lhand_counts": torch.stack(polish_l_counts, dim=1).detach()
            if polish_l_counts
            else None,
            "polish_rhand_counts": torch.stack(polish_r_counts, dim=1).detach()
            if polish_r_counts
            else None,
        }
