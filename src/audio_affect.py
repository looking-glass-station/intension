"""
Shared speech-affect helpers: one SER implementation, one clip writer.

Used by `prosody.py` (scores every diarized segment) and `invective.py` (reads
prosody's per-segment aggression score, writes review clips). Both load their wav
once and slice the array - nothing here re-opens the file per segment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import librosa
import torch


def extract_clip(audio: np.ndarray, sr: int, start: float, end: float, out_path: Path) -> bool:
    """Write audio[start:end] (seconds) to out_path. audio is an already-loaded array."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    a = max(0, int(start * sr))
    b = min(len(audio), int(end * sr))
    if b <= a:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio[a:b], sr)
    return True


# --------------------------------------------------------------------------- #
# audeering/wav2vec2 dimensional SER (arousal / dominance / valence)
# --------------------------------------------------------------------------- #
class _RegressionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.final_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


# transformers >= 4.31 renamed wav2vec2's weight-normed pos-conv params; the
# audeering checkpoint predates that, so `from_pretrained` silently leaves the
# positional conv randomly initialised. Remap the two keys on load.
_POS_CONV_REMAP = {
    "wav2vec2.encoder.pos_conv_embed.conv.weight_g":
        "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "wav2vec2.encoder.pos_conv_embed.conv.weight_v":
        "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


def build_ser(model_id: str):
    """
    (processor, model, device) for audeering's dimensional model, or None if it
    can't be built. That model has a regression head, so it needs a small custom
    class rather than a stock pipeline.
    """
    try:
        from huggingface_hub import hf_hub_download
        from transformers import (Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2PreTrainedModel,
                                  Wav2Vec2Processor)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = _RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.classifier(hidden)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = EmotionModel(Wav2Vec2Config.from_pretrained(model_id))
        sd = torch.load(hf_hub_download(model_id, "pytorch_model.bin"),
                        map_location="cpu", weights_only=True)
        sd = {_POS_CONV_REMAP.get(k, k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if any("pos_conv" in k for k in missing):
            return None  # remap didn't take - don't ship a half-random model
        return processor, model.to(device).eval(), device
    except Exception:
        return None


@torch.no_grad()
def ser_adv(clip: np.ndarray, sr: int, ser) -> Dict[str, float]:
    """arousal / dominance / valence in [0, 1] for one already-sliced audio clip."""
    processor, model, device = ser
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    if sr != 16000:
        clip = librosa.resample(np.ascontiguousarray(clip), orig_sr=sr, target_sr=16000)
    inputs = processor(clip, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    out = model(inputs)[0].cpu().numpy()  # [arousal, dominance, valence]
    return {"arousal": float(out[0]), "dominance": float(out[1]), "valence": float(out[2])}


def aggression_from_adv(adv: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """aggressive ~ high arousal + high dominance + low valence -> scalar in [0, 1]."""
    w = weights or {"arousal_weight": 1.0, "dominance_weight": 0.8, "valence_weight": 0.8}
    a, d, v = adv["arousal"], adv["dominance"], 1.0 - adv["valence"]
    wa, wd, wv = w["arousal_weight"], w["dominance_weight"], w["valence_weight"]
    return round(float((wa * a + wd * d + wv * v) / (wa + wd + wv)), 3)
