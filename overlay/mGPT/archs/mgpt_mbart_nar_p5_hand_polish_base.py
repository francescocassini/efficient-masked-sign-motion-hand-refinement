"""P5 SOKE-NAR experiment: conservative hand-only refinement after P3.

P3 generation is reproduced for the configured base steps. The body is then
frozen, while low-confidence left/right hand positions receive a small number
of additional refinement passes. This isolates hand-quality changes from body
quality changes.
"""
import math
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from .mgpt_mbart_nar_p3_train_aligned import Mbart_Based_MLM_NAR_P3_TrainAligned


class Mbart_Based_MLM_NAR_P5_HandPolishBase(Mbart_Based_MLM_NAR_P3_TrainAligned):
    """Run P3, freeze its body output, then polish uncertain hand positions."""

    def __init__(
        self,
        *args,
        hand_polish_ratios: Sequence[float],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hand_polish_ratios = tuple(float(ratio) for ratio in hand_polish_ratios)
        if not self.hand_polish_ratios:
            raise ValueError("P5 requires at least one hand-polish ratio")
        if any(ratio <= 0 or ratio >= 1 for ratio in self.hand_polish_ratios):
            raise ValueError("P5 hand-polish ratios must be in the open interval (0, 1)")
        print(
            "[SOKE-NAR-P5] P3 body-preserving hand polish: "
            f"base_steps={self.nar_steps}, ratios={self.hand_polish_ratios}"
        )

    @staticmethod
    def _lowest_confidence_mask(
        confidence: Tensor,
        motion_valid: Tensor,
        token_lengths: Tensor,
        ratio: float,
    ) -> Tensor:
        remask = torch.zeros_like(motion_valid)
        confidence = confidence.masked_fill(~motion_valid, float("inf"))
        for index, length in enumerate(token_lengths.tolist()):
            count = min(length, max(1, int(math.floor(length * ratio))))
            low = confidence[index, :length].topk(count, largest=False).indices
            remask[index, low] = True
        return remask

    @staticmethod
    def _predict_greedy_codes(logits: Tensor, candidates: Tensor):
        """Use deterministic MAP updates during polishing.

        P3 remains stochastic. Keeping the extra hand-only passes greedy avoids
        consuming RNG state and prevents P5 from changing later batches' P3
        body samples.
        """
        restricted = logits.index_select(-1, candidates)
        probabilities = torch.softmax(restricted, dim=-1)
        confidence, codes = probabilities.max(dim=-1)
        return candidates[codes], codes, confidence

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

        # Reproduce P3 generation exactly before any hand-only operation.
        conf_l = conf_r = None
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
            body[next_remask] = self.mask_emb_id
            lhand[next_remask] = self.mask_emb_id
            rhand[next_remask] = self.mask_emb_id
            remask = next_remask

        # Body and body_codes are never changed after this point.
        for ratio in self.hand_polish_ratios:
            remask_l = self._lowest_confidence_mask(
                conf_l, motion_valid, token_lengths, ratio
            )
            remask_r = self._lowest_confidence_mask(
                conf_r, motion_valid, token_lengths, ratio
            )
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
        }
