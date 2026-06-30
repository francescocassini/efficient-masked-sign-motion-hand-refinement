"""P0 SOKE-NAR experiment: cache the invariant text encoder output at inference.

This module intentionally leaves ``mgpt_mbart_nar.py`` untouched. It inherits
the trained model and changes only ``generate_direct`` so that the mBART text
encoder runs once per batch instead of once per NAR refinement step.
"""
import math
from typing import List, Optional

import torch
from torch import Tensor

from .mgpt_mbart_nar import Mbart_Based_MLM_NAR


class Mbart_Based_MLM_NAR_P0_EncoderCache(Mbart_Based_MLM_NAR):
    """Inference-equivalent SOKE-NAR with one cached source encoding per batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(
            "[SOKE-NAR-P0] encoder-cache experiment: source encoder runs once "
            "per generate_direct call; trained weights and decoding stay unchanged"
        )

    def _nar_logits_from_cached_encoder(
        self,
        encoder_hidden_states: Tensor,
        effective_source_mask: Optional[Tensor],
        decoder_body: Tensor,
        decoder_lhand: Tensor,
        decoder_rhand: Tensor,
        decoder_valid: Tensor,
    ):
        decoder_mask = self._bidirectional_mask(
            decoder_valid, self.language_model.main_lm.dtype
        )
        return self.language_model.inference_decoder(
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=effective_source_mask,
            decoder_input_ids=decoder_body,
            decoder_input_ids_hand=decoder_lhand,
            decoder_input_ids_rhand=decoder_rhand,
            decoder_attention_mask=decoder_mask,
            use_cache=False,
        )

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
            raise ValueError("SOKE-NAR requires oracle frame lengths for the first controlled test")

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

        max_tokens = int(token_lengths.max())
        total = max_tokens + 2
        positions = torch.arange(total, device=device).unsqueeze(0)
        motion_valid = positions < token_lengths.unsqueeze(1)
        decoder_valid = positions < (token_lengths + 2).unsqueeze(1)

        body = torch.full((len(texts), total), self.pad_emb_id, dtype=torch.long, device=device)
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

        # Preserve the current inference behavior exactly: generate_direct does
        # not pass source_mask to the encoder or decoder. Only its repeated
        # evaluation is removed.
        effective_source_mask = None
        encoder_hidden_states = self.language_model.main_lm.get_encoder()(
            input_ids=source_ids,
            attention_mask=effective_source_mask,
        )[0]

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
            for i, length in enumerate(token_lengths.tolist()):
                count = int(math.floor(length * ratio))
                if count > 0:
                    low = confidence[i, :length].topk(count, largest=False).indices
                    next_remask[i, low] = True
            body[next_remask] = self.mask_emb_id
            lhand[next_remask] = self.mask_emb_id
            rhand[next_remask] = self.mask_emb_id
            remask = next_remask

        return {
            "outputs_tokens": [body_codes[i, :length].detach() for i, length in enumerate(token_lengths)],
            "cleaned_text": [""] * len(texts),
            "outputs_tokens_hand": [lhand_codes[i, :length].detach() for i, length in enumerate(token_lengths)],
            "cleaned_text_hand": [""] * len(texts),
            "outputs_tokens_rhand": [rhand_codes[i, :length].detach() for i, length in enumerate(token_lengths)],
            "cleaned_text_rhand": [""] * len(texts),
            "predicted_token_lengths": token_lengths.detach(),
            "length_logits": None if length_logits is None else length_logits.detach(),
        }
