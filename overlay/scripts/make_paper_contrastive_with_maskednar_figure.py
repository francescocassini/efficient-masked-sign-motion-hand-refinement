#!/usr/bin/env python3
"""Regenerate the qualitative figure with a matched Masked-NAR column.

This script reruns the exact P5/HandPolish generator for one sample and captures
the token state immediately before hand polishing. That gives a true matched
Masked-NAR panel: same sample, same stochastic run, same body tokens as the
HandPolish cache used by PoseSelect.
"""

from __future__ import annotations

import base64
import html
import importlib.machinery
import math
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOKE_ROOT = Path(os.environ.get("SOKENAR_ROOT", ROOT))
DATA_ROOT = Path(os.environ.get("SOKE_DATA_ROOT", SOKE_ROOT / "data"))
sys.path.insert(0, str(SOKE_ROOT))
SMPLX_PYTHON_PATH = os.environ.get("SMPLX_PYTHON_PATH")
if SMPLX_PYTHON_PATH:
    sys.path.insert(0, SMPLX_PYTHON_PATH)
os.chdir(SOKE_ROOT)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
spacy_stub = sys.modules.setdefault("spacy", types.ModuleType("spacy"))
spacy_stub.__spec__ = importlib.machinery.ModuleSpec("spacy", loader=None)

from omegaconf import OmegaConf  # noqa: E402
import pytorch_lightning as pl  # noqa: E402

from mGPT.config import get_module_config  # noqa: E402
from mGPT.data.build_data import build_data  # noqa: E402
from mGPT.models.build_model import build_model  # noqa: E402
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae  # noqa: E402
from mGPT.archs.mgpt_mbart_nar_p5_hand_polish_aggressive_t5c_conf import (  # noqa: E402
    Mbart_Based_MLM_NAR_P5_HandPolishAggressive_T5CConf,
)

import scripts.make_paper_contrastive_qualitative_figure as basefig  # noqa: E402


SAMPLE = os.environ.get("PAPER_SAMPLE", "S000576_P0008_T00")
DATASET = os.environ.get("PAPER_DATASET", "csl").lower()
FRAME_IDX = int(os.environ.get("PAPER_FRAME", "0"))
OUT_PREFIX = os.environ.get(
    "PAPER_OUT_PREFIX",
    f"paper_qualitative_contrastive_with_maskednar_{DATASET}_{SAMPLE}_f{FRAME_IDX}",
)
CONFIG = Path(
    os.environ.get(
        "PAPER_CONFIG",
        SOKE_ROOT / "configs/paper/table4_poseselect_postcache.yaml",
    )
)
CKPT = (
    Path(os.environ["PAPER_MASKED_NAR_CKPT"])
    if "PAPER_MASKED_NAR_CKPT" in os.environ
    else SOKE_ROOT / "artifacts/checkpoints/masked_nar_e19.ckpt"
)
POSESELECT_NPY = Path(
    os.environ.get(
        "PAPER_POSESELECT_NPY",
        SOKE_ROOT / "artifacts/poseselect/predictions" / DATASET / f"{SAMPLE}.npy",
    )
)
HANDPOLISH_PKL = (
    Path(os.environ["PAPER_HANDPOLISH_PKL"])
    if "PAPER_HANDPOLISH_PKL" in os.environ
    else SOKE_ROOT / "artifacts/handpolish_cache/test_rank_0_rep0" / f"{SAMPLE}.pkl"
)
SOKE_COLLAPSE_DIR = {
    "csl": "COLLAPSE_FULL_SOKE_E69_CSL_SAVE",
    "phoenix": "COLLAPSE_FULL_SOKE_E69_PHX_SAVE",
    "phx": "COLLAPSE_FULL_SOKE_E69_PHX_SAVE",
}[DATASET]
SOKE_PKL = Path(
    os.environ.get(
        "PAPER_SOKE_PKL",
        SOKE_ROOT / "artifacts/soke_ar" / SOKE_COLLAPSE_DIR / "test_rank_0" / f"{SAMPLE}.pkl",
    )
)

OUT_DIR = Path(os.environ.get("PAPER_OUT_DIR", SOKE_ROOT / "assets/figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / f"{OUT_PREFIX}.png"
OUT_PNG_2X = OUT_DIR / f"{OUT_PREFIX}_2x.png"
OUT_SVG = OUT_DIR / f"{OUT_PREFIX}.svg"
OUT_PRE_PKL = OUT_DIR / f"{OUT_PREFIX}_prepolish.pkl"


def load_cfg():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(SOKE_ROOT / "configs/assets.yaml")
    cfg_base = OmegaConf.load(Path(cfg_assets.CONFIG_FOLDER) / "default.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(CONFIG))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    data_root = DATA_ROOT
    cfg.USE_GPUS = "0"
    cfg.DEVICE = [0] if torch.cuda.is_available() else 1
    cfg.ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.NUM_NODES = 1
    cfg.SEED_VALUE = 1234
    cfg.TRAIN.NUM_WORKERS = 0
    cfg.EVAL.NUM_WORKERS = 0
    cfg.TEST.NUM_WORKERS = 0
    cfg.TEST.BATCH_SIZE = 4
    cfg.TEST.REPLICATION_TIMES = 1
    cfg.TEST.SAVE_PREDICTIONS = False
    cfg.TEST.SKIP_METRICS = True
    cfg.TEST.CHECKPOINTS = str(CKPT)
    cfg.DATASET.H2S.ROOT = str(data_root / "How2Sign")
    cfg.DATASET.H2S.CSL_ROOT = str(data_root / "CSL-Daily")
    cfg.DATASET.H2S.PHOENIX_ROOT = str(data_root / "Phoenix_2014T")
    cfg.DATASET.H2S.MEAN_PATH = str(data_root / "CSL-Daily/mean.pt")
    cfg.DATASET.H2S.STD_PATH = str(data_root / "CSL-Daily/std.pt")
    cfg.TRAIN.PRETRAINED_VAE = str(SOKE_ROOT / "deps/tokenizer_ckpt/tokenizer.ckpt")
    cfg.FOLDER = str(SOKE_ROOT / "results")
    cfg.FOLDER_EXP = str(SOKE_ROOT / "results/mgpt_t5c_extended_cache/PAPER_PREPOLISH_CAPTURE")
    return cfg


def patch_generator():
    original = Mbart_Based_MLM_NAR_P5_HandPolishAggressive_T5CConf.generate_direct

    @torch.no_grad()
    def generate_direct_capture(self, texts, max_length=256, num_beams=1, do_sample=True, bad_words_ids=None, src=None, name=None, lengths=None):
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
                token_lengths, length_logits = self._predict_token_lengths(source_ids, source_mask, src)
        else:
            token_lengths = torch.tensor(
                [max(1, int(round(float(length) / self.down_t))) for length in lengths],
                dtype=torch.long,
                device=device,
            )
            token_lengths = token_lengths.float().mul(self.oracle_length_scale).round().long().clamp_min(1)

        total = int(token_lengths.max()) + 2
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

        encoder_hidden_states = self.language_model.main_lm.get_encoder()(
            input_ids=source_ids,
            attention_mask=None,
        )[0]

        conf_b = conf_l = conf_r = None
        p3_remask_counts = []
        for step in range(self.nar_steps):
            logits_body, logits_lhand, logits_rhand = self._nar_logits_from_cached_encoder(
                encoder_hidden_states, None, body, lhand, rhand, decoder_valid
            )
            pred_b, code_b, conf_b = self._predict_codes(logits_body, self._body_code_emb_ids.to(device))
            pred_l, code_l, conf_l = self._predict_codes(logits_lhand, self._lhand_code_emb_ids.to(device))
            pred_r, code_r, conf_r = self._predict_codes(logits_rhand, self._rhand_code_emb_ids.to(device))
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

        pre_body_codes = body_codes.clone()
        pre_lhand_codes = lhand_codes.clone()
        pre_rhand_codes = rhand_codes.clone()
        pre_conf_b = None if conf_b is None else conf_b.clone()
        pre_conf_l = None if conf_l is None else conf_l.clone()
        pre_conf_r = None if conf_r is None else conf_r.clone()

        polish_l_counts = []
        polish_r_counts = []
        for ratio in self.hand_polish_ratios:
            remask_l = self._lowest_confidence_mask(conf_l, motion_valid, token_lengths, ratio)
            remask_r = self._lowest_confidence_mask(conf_r, motion_valid, token_lengths, ratio)
            polish_l_counts.append(remask_l[:, :total].sum(dim=1).detach())
            polish_r_counts.append(remask_r[:, :total].sum(dim=1).detach())
            lhand[remask_l] = self.mask_emb_id
            rhand[remask_r] = self.mask_emb_id

            _, logits_lhand, logits_rhand = self._nar_logits_from_cached_encoder(
                encoder_hidden_states, None, body, lhand, rhand, decoder_valid
            )
            pred_l, code_l, conf_l = self._predict_greedy_codes(logits_lhand, self._lhand_code_emb_ids.to(device))
            pred_r, code_r, conf_r = self._predict_greedy_codes(logits_rhand, self._rhand_code_emb_ids.to(device))
            lhand = torch.where(remask_l, pred_l, lhand)
            rhand = torch.where(remask_r, pred_r, rhand)
            lhand_codes = torch.where(remask_l, code_l, lhand_codes)
            rhand_codes = torch.where(remask_r, code_r, rhand_codes)

        return {
            "outputs_tokens": [body_codes[index, :length].detach() for index, length in enumerate(token_lengths)],
            "cleaned_text": [""] * len(texts),
            "outputs_tokens_hand": [lhand_codes[index, :length].detach() for index, length in enumerate(token_lengths)],
            "cleaned_text_hand": [""] * len(texts),
            "outputs_tokens_rhand": [rhand_codes[index, :length].detach() for index, length in enumerate(token_lengths)],
            "cleaned_text_rhand": [""] * len(texts),
            "outputs_tokens_pre_polish": [pre_body_codes[index, :length].detach() for index, length in enumerate(token_lengths)],
            "outputs_tokens_hand_pre_polish": [pre_lhand_codes[index, :length].detach() for index, length in enumerate(token_lengths)],
            "outputs_tokens_rhand_pre_polish": [pre_rhand_codes[index, :length].detach() for index, length in enumerate(token_lengths)],
            "predicted_token_lengths": token_lengths.detach(),
            "length_logits": None if length_logits is None else length_logits.detach(),
            "confidence_body": [conf_b[index, :length].detach() for index, length in enumerate(token_lengths)],
            "confidence_lhand": [conf_l[index, :length].detach() for index, length in enumerate(token_lengths)],
            "confidence_rhand": [conf_r[index, :length].detach() for index, length in enumerate(token_lengths)],
            "confidence_body_pre_polish": [pre_conf_b[index, :length].detach() for index, length in enumerate(token_lengths)],
            "confidence_lhand_pre_polish": [pre_conf_l[index, :length].detach() for index, length in enumerate(token_lengths)],
            "confidence_rhand_pre_polish": [pre_conf_r[index, :length].detach() for index, length in enumerate(token_lengths)],
            "p3_remask_counts": torch.stack(p3_remask_counts, dim=1).detach() if p3_remask_counts else None,
            "polish_lhand_counts": torch.stack(polish_l_counts, dim=1).detach() if polish_l_counts else None,
            "polish_rhand_counts": torch.stack(polish_r_counts, dim=1).detach() if polish_r_counts else None,
        }

    Mbart_Based_MLM_NAR_P5_HandPolishAggressive_T5CConf.generate_direct = generate_direct_capture
    return original


def tokens_to_feats(model, body_tokens, lhand_tokens, rhand_tokens):
    def pad_frames(value: torch.Tensor, frames: int) -> torch.Tensor:
        if value.shape[0] >= frames:
            return value[:frames]
        tail = value[-1:].repeat(frames - value.shape[0], 1)
        return torch.cat([value, tail], dim=0)

    device = model.device
    length = len(body_tokens)
    max_len = length * 4
    with torch.no_grad():
        body = torch.as_tensor(body_tokens, dtype=torch.long, device=device).clamp(0, model.vae.code_num - 1)
        lh = torch.as_tensor(lhand_tokens, dtype=torch.long, device=device).clamp(0, model.hand_vae.code_num - 1)
        rh = torch.as_tensor(rhand_tokens, dtype=torch.long, device=device).clamp(0, model.rhand_vae.code_num - 1)
        body_motion = model.vae.decode(body).squeeze(0)
        lh_motion = model.hand_vae.decode(lh).squeeze(0)
        rh_motion = model.rhand_vae.decode(rh).squeeze(0)
        frames = max(body_motion.shape[0], lh_motion.shape[0], rh_motion.shape[0], max_len)
        out = torch.zeros((frames, 133), device=device, dtype=body_motion.dtype)
        body_motion = pad_frames(body_motion, frames)
        lh_motion = pad_frames(lh_motion, frames)
        rh_motion = pad_frames(rh_motion, frames)
        out[:, :30] = body_motion[:, :30]
        out[:, -13:] = body_motion[:, 30:43]
        out[:, 30:75] = lh_motion
        out[:, 75:120] = rh_motion
        return model.datamodule.renorm4t2m(out.unsqueeze(0)).squeeze(0).detach().cpu().numpy()


def capture_prepolish():
    patch_generator()
    cfg = load_cfg()
    pl.seed_everything(cfg.SEED_VALUE, workers=True)
    datamodule = build_data(cfg)
    _ = datamodule.test_dataset
    model = build_model(cfg, datamodule)
    load_pretrained_vae(cfg, model, logger=types.SimpleNamespace(info=print, warning=print))
    load_pretrained(cfg, model, logger=types.SimpleNamespace(info=print, warning=print), phase="test")
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    target_batch = None
    target_rs_set = None
    target_index = None
    for batch in datamodule.test_dataloader():
        batch = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            rs_set = model.val_t2m_forward(batch)
        names = [str(name).split("/")[-1] for name in batch["name"]]
        if SAMPLE in names:
            target_batch = batch
            target_rs_set = rs_set
            target_index = names.index(SAMPLE)
            break
    if target_batch is None or target_rs_set is None or target_index is None:
        raise RuntimeError(f"{SAMPLE} not found while iterating test dataloader")

    batch = target_batch
    rs_set = target_rs_set
    index = target_index
    pre_feats = tokens_to_feats(
        model,
        rs_set["tokens_body"][index].detach().cpu().numpy(),
        rs_set["tokens_lhand_pre_polish"][index].detach().cpu().numpy(),
        rs_set["tokens_rhand_pre_polish"][index].detach().cpu().numpy(),
    )
    hp_feats = rs_set["m_rst"][index, : rs_set["lengths_rst"][index]].detach().cpu().numpy()
    payload = {
        "feats_rst": pre_feats,
        "feats_ref": rs_set["m_ref"][index, : rs_set["length"][index]].detach().cpu().numpy(),
        "text": batch["text"][index],
        "tokens_body": rs_set["tokens_body"][index].detach().cpu().numpy(),
        "tokens_lhand": rs_set["tokens_lhand_pre_polish"][index].detach().cpu().numpy(),
        "tokens_rhand": rs_set["tokens_rhand_pre_polish"][index].detach().cpu().numpy(),
        "handpolish_feats_rst": hp_feats,
        "final_tokens_lhand": rs_set["tokens_lhand"][index].detach().cpu().numpy(),
        "final_tokens_rhand": rs_set["tokens_rhand"][index].detach().cpu().numpy(),
        "confidence_body": rs_set["confidence_body"][index].detach().cpu().numpy(),
        "confidence_lhand": rs_set["confidence_lhand"][index].detach().cpu().numpy(),
        "confidence_rhand": rs_set["confidence_rhand"][index].detach().cpu().numpy(),
    }
    with OUT_PRE_PKL.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def add_poseselect_prediction(payload):
    from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
    from scripts.p6_hand_topk_candidate_oracle import load_regressor
    from scripts.p6_token_edit_oracle_ceiling import load_vq_pack
    from scripts.p6k_end_to_end_runner import load_selector, predict_p6k

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack_args = types.SimpleNamespace(
        config=Path("configs/soke.yaml"),
        default_config=Path("configs/default.yaml"),
        vae_ckpt=Path("deps/tokenizer_ckpt/tokenizer.ckpt"),
        mean_path=DATA_ROOT / "CSL-Daily/mean.pt",
        std_path=DATA_ROOT / "CSL-Daily/std.pt",
    )
    hp_payload = {
        "feats_rst": np.asarray(payload["handpolish_feats_rst"], dtype=np.float32),
        "feats_ref": np.asarray(payload["feats_ref"], dtype=np.float32),
        "text": payload["text"],
        "tokens_body": np.asarray(payload["tokens_body"], dtype=np.int64),
        "tokens_lhand": np.asarray(payload["final_tokens_lhand"], dtype=np.int64),
        "tokens_rhand": np.asarray(payload["final_tokens_rhand"], dtype=np.int64),
        "confidence_body": np.asarray(payload["confidence_body"], dtype=np.float32),
        "confidence_lhand": np.asarray(payload["confidence_lhand"], dtype=np.float32),
        "confidence_rhand": np.asarray(payload["confidence_rhand"], dtype=np.float32),
    }
    pack = load_vq_pack(pack_args, device)
    editor, _editor_args = load_p6_editor(
        Path("results/p6_hand_token_editor/full_both_e20_h256_l4/best_p6_hand_token_editor.pt"),
        device,
    )
    regressor, _regressor_args = load_regressor(
        Path("results/p6_hand_gain_regressor/full_both_e10_h192_l3/best_p6_hand_gain_regressor.pt"),
        device,
    )
    selector, _selector_args = load_selector(
        Path("results/p6_pose_aware_candidate_selector/full_e12_top5_b020_h256/best_p6_pose_aware_candidate_selector.pt"),
        device,
    )
    _p6d_pred, p6k_pred, stats = predict_p6k(
        editor,
        regressor,
        selector,
        hp_payload,
        pack,
        topk=5,
        budget=0.20,
        device=device,
    )
    payload["poseselect_feats_rst"] = p6k_pred.astype(np.float32)
    payload["poseselect_stats"] = stats
    with OUT_PRE_PKL.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def draw_with_maskednar(pre_payload, scale=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = basefig.load_mean_std(DATA_ROOT / "CSL-Daily/mean.pt", DATA_ROOT / "CSL-Daily/std.pt", device)
    soke = basefig.load_pickle(SOKE_PKL)
    handpolish = np.asarray(pre_payload["handpolish_feats_rst"], dtype=np.float32)
    poseselect = np.asarray(pre_payload["poseselect_feats_rst"], dtype=np.float32)

    final_delta = float(np.abs(handpolish[:, :30] - poseselect[:, :30]).max())
    pre_delta = float(np.abs(np.asarray(pre_payload["feats_rst"])[:, :30] - handpolish[:, :30]).max())
    if final_delta > 1e-7:
        raise RuntimeError(f"PoseSelect body mismatch: {final_delta}")
    if pre_delta > 1e-5:
        raise RuntimeError(f"Pre-polish Masked-NAR body mismatch: {pre_delta}")

    methods = [
        ("GT", "reference motion", "#1f7a3f"),
        ("SOKE-AR", "autoregressive baseline", "#3459d1"),
        ("Masked-NAR", "matched pre-polish state", "#6b55d8"),
        ("+ HandPolish", "same body; refined hands", "#d97a00"),
        ("+ PoseSelect", "same body; selected hands", "#b22cc8"),
    ]
    feats = {
        "GT": np.asarray(pre_payload["feats_ref"], dtype=np.float32),
        "SOKE-AR": np.asarray(soke["feats_rst"], dtype=np.float32),
        "Masked-NAR": np.asarray(pre_payload["feats_rst"], dtype=np.float32),
        "+ HandPolish": handpolish,
        "+ PoseSelect": poseselect,
    }
    joints = {name: basefig.feats_to_joints65(value, mean, std, device) for name, value in feats.items()}

    width, height = 1500 * scale, 690 * scale
    margin = 26 * scale
    label_w = 170 * scale
    header_h = 72 * scale
    col_w = 250 * scale
    col_gap = 7 * scale
    row_gap = 8 * scale
    body_h = 215 * scale
    hand_h = 150 * scale
    x0 = margin + label_w
    y0 = margin + header_h

    image = basefig.Image.new("RGB", (width, height), (255, 255, 255))
    draw = basefig.ImageDraw.Draw(image)
    for col_idx, (name, desc, color_hex) in enumerate(methods):
        x = x0 + col_idx * (col_w + col_gap)
        draw.rectangle((x, margin, x + 7 * scale, margin + header_h - 10 * scale), fill=basefig.hex_to_rgb(color_hex))
        title_size = 20 if name == "Masked-NAR" else 22
        basefig.text(draw, (x + 18 * scale, margin + 9 * scale), name, title_size * scale, color_hex, bold=True)
        basefig.text(draw, (x + 18 * scale, margin + 40 * scale), desc, 12 * scale, "#555555")

    full_bounds = basefig.bounds_for(joints, list(range(65)))
    left_bounds = basefig.hand_bounds_for(joints, wrist_idx=12, hand_start=25)
    right_bounds = basefig.hand_bounds_for(joints, wrist_idx=13, hand_start=45)
    current_y = y0
    for row_index, (row_title, row_desc) in enumerate(basefig.ROWS):
        row_h = body_h if row_index == 0 else hand_h
        basefig.text(draw, (margin + 4 * scale, current_y + 21 * scale), row_title, 17 * scale, "#333333", bold=True)
        basefig.text(draw, (margin + 4 * scale, current_y + 51 * scale), row_desc, 14 * scale, "#777777")
        for col_idx, (method, _desc, color_hex) in enumerate(methods):
            x = x0 + col_idx * (col_w + col_gap)
            box = (x, current_y, x + col_w, current_y + row_h)
            draw.rectangle(box, fill=(248, 248, 248))
            frame = joints[method][min(FRAME_IDX, len(joints[method]) - 1)]
            color = basefig.hex_to_rgb(color_hex)
            if row_index == 0:
                basefig.draw_body_panel(draw, frame, full_bounds, box, color, scale)
            elif row_index == 1:
                basefig.draw_hand_panel(draw, frame, left_bounds, box, color, scale, side="left")
            else:
                basefig.draw_hand_panel(draw, frame, right_bounds, box, color, scale, side="right")
            basefig.draw_panel_border(draw, box)
        current_y += row_h + row_gap
    return image


def write_svg_wrapper(png_path: Path, svg_path: Path) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="690" viewBox="0 0 1500 690">\n'
        f'  <image href="data:image/png;base64,{html.escape(encoded)}" x="0" y="0" width="1500" height="690"/>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg, encoding="utf-8")


def main():
    for path in (CONFIG, CKPT, POSESELECT_NPY, HANDPOLISH_PKL, SOKE_PKL):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    payload = capture_prepolish()
    payload = add_poseselect_prediction(payload)
    draw_with_maskednar(payload, scale=1).save(OUT_PNG)
    draw_with_maskednar(payload, scale=2).save(OUT_PNG_2X)
    write_svg_wrapper(OUT_PNG, OUT_SVG)
    print(f"wrote {OUT_PRE_PKL}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PNG_2X}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
