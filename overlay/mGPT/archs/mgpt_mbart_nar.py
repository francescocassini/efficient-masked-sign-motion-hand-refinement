"""SOKE-NAR: controlled non-autoregressive ablation of the original SOKE AMG.

Everything upstream of the decoder is inherited from SOKE unchanged:
pretrained mBART, full fine-tuning, language routing, word2code/name2kws
retrieval, motion vocabulary, and the synchronized three-part representation.

The only architectural treatment is generation:
  - training predicts randomly masked motion positions bidirectionally;
  - inference starts from an all-mask grid of oracle length and performs
    confidence remasking.

The original ``mgpt_mbart.py`` and ``lm_multihead.py`` remain untouched.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mgpt_mbart import Mbart_Based_MLM, correct_lang_token


class Mbart_Based_MLM_NAR(Mbart_Based_MLM):
    def __init__(
        self,
        *args,
        nar_steps: int = 12,
        nar_min_mask_ratio: float = 0.05,
        nar_on_policy_prob: float = 0.0,
        nar_sample_temperature: float = 1.0,
        nar_sample_top_k: int = 0,
        length_predictor: bool = False,
        length_predictor_kind: str = "learned",
        length_only_training: bool = False,
        length_min_tokens: int = 10,
        length_max_tokens: int = 100,
        length_hidden_dim: int = 256,
        length_dataset_dim: int = 16,
        length_explicit_features: bool = False,
        freeze_backbone_for_length: bool = False,
        oracle_length_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.model_type != "mbart_multi":
            raise ValueError("SOKE-NAR first controlled test supports model_type=mbart_multi only")

        self.nar_steps = nar_steps
        self.nar_min_mask_ratio = nar_min_mask_ratio
        # Reserved for a second controlled experiment. It is intentionally off
        # in the first AR-vs-NAR comparison.
        self.nar_on_policy_prob = nar_on_policy_prob
        self.nar_sample_temperature = nar_sample_temperature
        self.nar_sample_top_k = nar_sample_top_k
        self.length_predictor = length_predictor
        self.length_predictor_kind = length_predictor_kind
        self.length_only_training = length_only_training
        self.length_min_tokens = length_min_tokens
        self.length_max_tokens = length_max_tokens
        self.length_num_classes = length_max_tokens - length_min_tokens + 1
        self.length_explicit_features = length_explicit_features
        self.freeze_backbone_for_length = freeze_backbone_for_length
        self.oracle_length_scale = oracle_length_scale

        self.mask_emb_id = self.tok_id_to_emb_id[
            self.tokenizer.convert_tokens_to_ids("<mask>")
        ]
        self.pad_emb_id = self.tok_id_to_emb_id[self.tokenizer.pad_token_id]
        self.eos_emb_id = self.eos_idx

        self.register_buffer(
            "_body_code_emb_ids",
            self._code_embedding_ids("motion", self.m_codebook_size),
            persistent=False,
        )
        self.register_buffer(
            "_lhand_code_emb_ids",
            self._code_embedding_ids("hand", self.hand_codebook_size),
            persistent=False,
        )
        self.register_buffer(
            "_rhand_code_emb_ids",
            self._code_embedding_ids("rhand", self.rhand_codebook_size),
            persistent=False,
        )
        if self.length_predictor and self.length_predictor_kind == "learned":
            d_model = self.language_model.main_lm.config.d_model
            length_input_dim = d_model
            if self.length_explicit_features:
                self.length_dataset_embedding = nn.Embedding(3, length_dataset_dim)
                length_input_dim += length_dataset_dim + 1
            self.length_head = nn.Sequential(
                nn.LayerNorm(length_input_dim),
                nn.Linear(length_input_dim, length_hidden_dim),
                nn.GELU(),
                nn.Linear(length_hidden_dim, self.length_num_classes),
            )
            if freeze_backbone_for_length:
                for parameter in self.parameters():
                    parameter.requires_grad = False
                for parameter in self.length_head.parameters():
                    parameter.requires_grad = True
                if self.length_explicit_features:
                    for parameter in self.length_dataset_embedding.parameters():
                        parameter.requires_grad = True
        print(
            "[SOKE-NAR] controlled ablation: original SOKE source/retrieval/mBART; "
            f"bidirectional masked decoder, oracle length, K={self.nar_steps}, "
            f"on_policy_prob={self.nar_on_policy_prob}, "
            f"sampling_temperature={self.nar_sample_temperature}, "
            f"sampling_top_k={self.nar_sample_top_k}, "
            f"length_predictor={self.length_predictor}, "
            f"length_predictor_kind={self.length_predictor_kind}, "
            f"length_only_training={self.length_only_training}, "
            f"length_explicit_features={self.length_explicit_features}, "
            f"oracle_length_scale={self.oracle_length_scale}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone_for_length:
            self.language_model.eval()
            self.length_head.train(mode)
        return self

    def _code_embedding_ids(self, pattern: str, size: int) -> Tensor:
        ids = []
        for i in range(size):
            tok_id = self.tokenizer.convert_tokens_to_ids(f"<{pattern}_id_{i}>")
            ids.append(self.tok_id_to_emb_id[tok_id])
        return torch.tensor(ids, dtype=torch.long)

    def _encode_source(self, inputs, src, name, device):
        # This is deliberately the same retrieval/source path as SOKE.
        if self.num_kws_per_sen > 0:
            kw_strings = self.get_kw_strings(name, src)
            inputs = [text + kws for text, kws in zip(inputs, kw_strings)]

        source = self.tokenizer(
            inputs,
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_length=True,
        )
        source_ids = source.input_ids.to(device)
        source_mask = source.attention_mask.to(device)
        token_len = source.length.to(device)
        correct_lang_token(
            self.tokenizer,
            source_ids,
            token_len,
            src,
            part=None,
            target=False,
            model_type=self.model_type,
        )
        self.map_ids(source_ids, direction="token_to_emb")
        return source_ids, source_mask

    def _length_logits(
        self, source_ids: Tensor, source_mask: Tensor, src: List[str]
    ) -> Tensor:
        hidden = self.language_model.main_lm.get_encoder()(
            input_ids=source_ids,
            attention_mask=source_mask,
        )[0]
        weights = source_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if self.length_explicit_features:
            dataset_ids = torch.tensor(
                [{"how2sign": 0, "csl": 1, "phoenix": 2}[source] for source in src],
                dtype=torch.long,
                device=source_ids.device,
            )
            dataset_features = self.length_dataset_embedding(dataset_ids)
            source_lengths = (
                source_mask.sum(dim=1, keepdim=True).to(hidden.dtype)
                / float(self.max_length)
            )
            pooled = torch.cat([pooled, dataset_features, source_lengths], dim=-1)
        return self.length_head(pooled)

    def _predict_token_lengths(
        self, source_ids: Tensor, source_mask: Tensor, src: List[str]
    ) -> tuple[Tensor, Tensor]:
        logits = self._length_logits(source_ids, source_mask, src)
        token_lengths = logits.argmax(dim=-1) + self.length_min_tokens
        return token_lengths, logits

    def _linear_text_token_lengths(self, texts: List[str], src: List[str]) -> Tensor:
        lengths = []
        coefficients = {
            "csl": (6.415326260174925, 1.4498280609957415),
            "phoenix": (2.565627084830793, 1.7574866182940447),
        }
        for text, source in zip(texts, src):
            intercept, slope = coefficients[source]
            units = (
                sum(not char.isspace() for char in text)
                if source == "csl"
                else len(text.split())
            )
            lengths.append(round(intercept + slope * units))
        return torch.tensor(
            lengths,
            dtype=torch.long,
            device=self.language_model.main_lm.device,
        ).clamp(self.length_min_tokens, self.length_max_tokens)

    @torch.no_grad()
    def predict_token_lengths(
        self, texts: List[str], src: List[str], name: List[str]
    ) -> tuple[Tensor, Tensor]:
        if self.length_predictor_kind == "linear_text_units":
            return self._linear_text_token_lengths(texts, src), None
        device = self.language_model.main_lm.device
        source_ids, source_mask = self._encode_source(texts, src, name, device)
        return self._predict_token_lengths(source_ids, source_mask, src)

    def _encode_target(self, outputs, src, part, device):
        target = self.tokenizer(
            outputs,
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_length=True,
        )
        ids = target.input_ids.to(device)
        attention_mask = target.attention_mask.to(device)
        token_len = target.length.to(device)
        correct_lang_token(
            self.tokenizer,
            ids,
            token_len,
            src,
            part=part,
            target=True,
            model_type=self.model_type,
        )
        self.map_ids(ids, direction="token_to_emb")
        return ids, attention_mask

    @staticmethod
    def _membership(ids: Tensor, candidates: Tensor) -> Tensor:
        return (ids.unsqueeze(-1) == candidates.to(ids.device)).any(dim=-1)

    def _sample_motion_mask(self, valid_motion: Tensor) -> Tensor:
        batch, _ = valid_motion.shape
        # MoMask/MaskGIT cosine schedule, sampled independently per example.
        u = torch.rand(batch, device=valid_motion.device)
        ratios = torch.cos(u * math.pi / 2).clamp_min(self.nar_min_mask_ratio)
        selected = torch.zeros_like(valid_motion)
        for i in range(batch):
            positions = valid_motion[i].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            count = max(1, int(math.ceil(positions.numel() * float(ratios[i]))))
            chosen = positions[torch.randperm(positions.numel(), device=positions.device)[:count]]
            selected[i, chosen] = True
        return selected

    @staticmethod
    def _bidirectional_mask(valid: Tensor, dtype: torch.dtype) -> Tensor:
        # transformers 4.41 accepts a custom 4D decoder mask. Ones mean that a
        # query/key pair is visible; this bypasses mBART's default causal mask.
        allowed = valid[:, None, :, None] & valid[:, None, None, :]
        return allowed.to(dtype=dtype)

    def _nar_logits(
        self,
        source_ids,
        source_mask,
        decoder_body,
        decoder_lhand,
        decoder_rhand,
        decoder_valid,
        use_source_attention_mask=True,
    ):
        encoder = self.language_model.main_lm.get_encoder()
        # Match SOKE exactly: training supplies the source mask, while the
        # original generate_direct path does not pass it to LMMultiHead.generate.
        effective_source_mask = source_mask if use_source_attention_mask else None
        encoder_outputs = encoder(input_ids=source_ids, attention_mask=effective_source_mask)
        hidden = encoder_outputs[0]
        decoder_mask = self._bidirectional_mask(
            decoder_valid, self.language_model.main_lm.dtype
        )
        return self.language_model.inference_decoder(
            encoder_hidden_states=hidden,
            attention_mask=effective_source_mask,
            decoder_input_ids=decoder_body,
            decoder_input_ids_hand=decoder_lhand,
            decoder_input_ids_rhand=decoder_rhand,
            decoder_attention_mask=decoder_mask,
            use_cache=False,
        )

    @staticmethod
    def _masked_ce(logits, target, selected):
        labels = torch.where(selected, target, torch.full_like(target, -100))
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
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
            source_ids, source_mask = self._encode_source(texts, src, name, device)
            logits = self._length_logits(source_ids, source_mask, src)
            targets = torch.tensor(lengths, device=device, dtype=torch.long)
            targets = targets.clamp(self.length_min_tokens, self.length_max_tokens)
            targets = targets - self.length_min_tokens
            loss = F.cross_entropy(logits, targets)
            # Keep MotionGPT's existing three-stream loss interface unchanged.
            zero = loss * 0.0
            return {"loss": loss, "loss_hand": zero, "loss_rhand": zero}

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

        decoder_body = torch.where(selected, self.mask_emb_id, body)
        decoder_lhand = torch.where(selected, self.mask_emb_id, lhand)
        decoder_rhand = torch.where(selected, self.mask_emb_id, rhand)

        logits_body, logits_lhand, logits_rhand = self._nar_logits(
            source_ids,
            source_mask,
            decoder_body,
            decoder_lhand,
            decoder_rhand,
            decoder_valid,
        )
        return {
            "loss": self._masked_ce(logits_body, body, selected),
            "loss_hand": self._masked_ce(logits_lhand, lhand, selected),
            "loss_rhand": self._masked_ce(logits_rhand, rhand, selected),
        }

    def _lang_embedding_ids(self, src, part, device):
        from .mgpt_mbart import make_decoder_input_ids

        ids = make_decoder_input_ids(
            self.tokenizer, device, src, part=part, model_type=self.model_type
        )
        self.map_ids(ids, direction="token_to_emb")
        return ids.squeeze(-1)

    def _predict_codes(self, logits, candidates):
        restricted = logits.index_select(-1, candidates)
        temperature = max(float(self.nar_sample_temperature), 1e-6)
        top_k = min(max(int(self.nar_sample_top_k), 0), restricted.shape[-1])
        if temperature == 1.0 and top_k == 0:
            probs = F.softmax(restricted, dim=-1)
            confidence, codes = probs.max(dim=-1)
        else:
            restricted = restricted / temperature
            if top_k > 0:
                top_values, top_indices = restricted.topk(top_k, dim=-1)
                filtered = torch.full_like(restricted, float("-inf"))
                filtered.scatter_(-1, top_indices, top_values)
                restricted = filtered
            probs = F.softmax(restricted, dim=-1)
            batch, length, vocab = probs.shape
            codes = torch.multinomial(
                probs.reshape(-1, vocab), num_samples=1
            ).reshape(batch, length)
            confidence = probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        emb_ids = candidates[codes]
        return emb_ids, codes, confidence

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
        total = max_tokens + 2  # motion grid + EOS + target-language token
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

        for step in range(self.nar_steps):
            logits_body, logits_lhand, logits_rhand = self._nar_logits(
                source_ids,
                source_mask,
                body,
                lhand,
                rhand,
                decoder_valid,
                use_source_attention_mask=False,
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

            # Score the current triplet jointly, then allow previously committed
            # positions to be revised at the next iteration.
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

    def generate_conditional(
        self,
        texts: Optional[List[str]] = None,
        motion_tokens: Optional[Tensor] = None,
        hand_tokens: Optional[Tensor] = None,
        lengths: Optional[List[int]] = None,
        task: str = "t2m",
        with_len: bool = False,
        stage: str = "train",
        tasks: dict = None,
        src: List[str] = None,
        name: List[str] = None,
    ):
        del motion_tokens, hand_tokens, with_len, stage
        if task not in ["t2m", "m2m", "pred", "inbetween"]:
            raise NotImplementedError("SOKE-NAR first test is text-to-motion only")
        if texts is None or (lengths is None and not self.length_predictor):
            raise ValueError(
                "SOKE-NAR text-to-motion generation requires texts and either "
                "oracle lengths or an enabled length predictor"
            )
        if tasks is None:
            tasks = [{"input": ["<Caption_Placeholder>"], "output": [""]}] * len(texts)
        empty_motion = [""] * len(texts)
        # Keep SOKE's test prompt unchanged; lengths are used only to size the NAR grid.
        inputs, _ = self.template_fulfill(
            tasks, [0] * len(texts), empty_motion, texts, "test"
        )
        return self.generate_direct(inputs, src=src, name=name, lengths=lengths)
