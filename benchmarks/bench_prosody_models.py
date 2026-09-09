"""
Score the hand-labelled prosody clips against a panel of SER models.

Reads benchmarks/prosody_eval/clips_review.csv (needs `is_aggressive_YN` filled
Y/N by listening) and the clips in prosody_eval/clips/, runs each model, and
reports how well its "hostility" score separates the Y clips from the N clips
(mean Y vs mean N, ROC-AUC, best single-threshold accuracy).

    uv run python benchmarks/bench_prosody_models.py

emotion2vec needs funasr (heavy) - run it separately and merge:
    uv run --with funasr python benchmarks/bench_prosody_models.py --emotion2vec-only

Models: tier1 (from the CSV, baseline), audeering dim, 3loi MSP-Podcast
categorical (anger+contempt+disgust) and dimensional, superb (from the CSV).
"""
from __future__ import annotations

import argparse
import csv
import functools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
EVAL = ROOT / "benchmarks" / "prosody_eval"
CLIPS = EVAL / "clips"
print = functools.partial(print, flush=True)  # noqa: A001


def auc(pos: list[float], neg: list[float]) -> float:
    """P(random Y score > random N score); 0.5 = no separation."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    vals = sorted(set(pos + neg))
    best_acc, best_t = 0.0, 0.0
    n = len(pos) + len(neg)
    for t in vals:
        acc = (sum(p >= t for p in pos) + sum(x < t for x in neg)) / n
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def load_labels() -> list[dict]:
    rows = []
    with (EVAL / "clips_review.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            yn = (r.get("is_aggressive_YN") or "").strip().upper()
            if yn in ("Y", "N") and (CLIPS / r["clip"]).exists():
                r["_label"] = yn == "Y"
                rows.append(r)
    return rows


def emotion2vec_only(rows):
    from funasr import AutoModel
    import soundfile as sf
    import librosa
    m = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", disable_update=True)
    out = {}
    for r in rows:
        wav, sr = sf.read(str(CLIPS / r["clip"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        res = m.generate(wav, granularity="utterance", extract_embedding=False)[0]
        d = {lbl.split("/")[-1].strip().lower(): s for lbl, s in zip(res["labels"], res["scores"])}
        out[r["clip"]] = round(d.get("angry", 0.0) + d.get("disgusted", 0.0), 4)
    (EVAL / "_emotion2vec.csv").write_text(
        "clip,emotion2vec_hostile\n" + "".join(f"{k},{v}\n" for k, v in out.items()), encoding="utf-8")
    print(f"emotion2vec scored {len(out)} clips -> {EVAL / '_emotion2vec.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emotion2vec-only", action="store_true")
    args = ap.parse_args()

    rows = load_labels()
    if not rows:
        raise SystemExit("no Y/N-labelled clips in clips_review.csv")
    n_y = sum(r["_label"] for r in rows)
    print(f"{len(rows)} labelled clips: {n_y} Y, {len(rows) - n_y} N\n")

    if args.emotion2vec_only:
        emotion2vec_only(rows)
        return

    sys.path.insert(0, str(SRC))
    import numpy as np
    import soundfile as sf
    import librosa
    import audio_affect

    def wav_of(clip):
        w, sr = sf.read(str(CLIPS / clip), dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        return w, sr

    scorers: dict[str, callable] = {}

    # baseline: tier1 from the CSV
    scorers["tier1 (DSP, speaker-rel)"] = lambda r: float(r["tier1"])
    scorers["superb_ang (CSV)"] = lambda r: float(r["superb_ang"]) if r.get("superb_ang") not in (None, "", "err") else float("nan")

    # audeering dimensional -> aggression scalar
    aud = audio_affect.build_ser("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    if aud:
        def _aud(r, _c={}):
            if r["clip"] not in _c:
                w, sr = wav_of(r["clip"])
                _c[r["clip"]] = audio_affect.aggression_from_adv(audio_affect.ser_adv(w, sr, aud))
            return _c[r["clip"]]
        scorers["audeering (dim->aggr)"] = _aud

    import torch

    def _fix_pos_conv(sd: dict) -> dict:
        # transformers >= 4.31 renamed weight-normed pos-conv params; these
        # checkpoints predate it, so remap weight_g/_v -> parametrizations.
        out = {}
        for k, v in sd.items():
            k = k.replace(".pos_conv_embed.conv.weight_g", ".pos_conv_embed.conv.parametrizations.weight.original0")
            k = k.replace(".pos_conv_embed.conv.weight_v", ".pos_conv_embed.conv.parametrizations.weight.original1")
            out[k] = v
        return out

    # 3loi MSP-Podcast models: custom SERModel via trust_remote_code (its __init__
    # builds a fresh wavlm-large, then from_pretrained overwrites - and reinits the
    # pos-conv). Load, then remap+reload the checkpoint to fix that layer.
    def load_3loi(name):
        try:
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForAudioClassification
            m = AutoModelForAudioClassification.from_pretrained(name, trust_remote_code=True)
            sd = load_file(hf_hub_download(name, "model.safetensors"))
            missing, _ = m.load_state_dict(_fix_pos_conv(sd), strict=False)
            if any("pos_conv" in k for k in missing):
                print(f"  ({name}: pos_conv remap failed)")
                return None
            return m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        except Exception as e:  # noqa: BLE001
            print(f"  ({name} unavailable: {type(e).__name__}: {str(e)[:140]})")
            return None
    cat = load_3loi("3loi/SER-Odyssey-Baseline-WavLM-Categorical")
    if cat is not None:
        id2label = {int(k): v for k, v in cat.config.id2label.items()}
        hostile_ids = [i for i, v in id2label.items() if v.lower() in ("angry", "contempt", "disgust")]
        mean, std = cat.config.mean, cat.config.std

        @torch.no_grad()
        def _cat(r, _c={}):
            if r["clip"] not in _c:
                w, sr = wav_of(r["clip"])
                if sr != 16000:
                    w = librosa.resample(w, orig_sr=sr, target_sr=16000)
                nw = (w - mean) / (std + 1e-6)
                wavs = torch.tensor(nw).unsqueeze(0).float().to(cat.device)
                mask = torch.ones(1, len(nw), device=cat.device)
                logits = cat(wavs, mask)[0]  # SERModel returns a bare (batch, C) tensor
                p = torch.softmax(logits, -1).cpu().numpy()
                _c[r["clip"]] = float(sum(p[i] for i in hostile_ids))
            return _c[r["clip"]]
        scorers["3loi-cat (ang+cont+disg)"] = _cat

    mattr = load_3loi("3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes")
    if mattr is not None:
        mean2, std2 = mattr.config.mean, mattr.config.std

        @torch.no_grad()
        def _mattr(r, _c={}):
            if r["clip"] not in _c:
                w, sr = wav_of(r["clip"])
                if sr != 16000:
                    w = librosa.resample(w, orig_sr=sr, target_sr=16000)
                nw = (w - mean2) / (std2 + 1e-6)
                wavs = torch.tensor(nw).unsqueeze(0).float().to(mattr.device)
                mask = torch.ones(1, len(nw), device=mattr.device)
                out = mattr(wavs, mask)[0].cpu().numpy()  # arousal, dominance, valence
                _c[r["clip"]] = audio_affect.aggression_from_adv(
                    {"arousal": float(out[0]), "dominance": float(out[1]), "valence": float(out[2])})
            return _c[r["clip"]]
        scorers["3loi-multiattr (dim)"] = _mattr

    # merge a prior emotion2vec run if present
    e2v_file = EVAL / "_emotion2vec.csv"
    if e2v_file.exists():
        e2v = {r["clip"]: float(r["emotion2vec_hostile"])
               for r in csv.DictReader(e2v_file.open(encoding="utf-8"))}
        scorers["emotion2vec (ang+disg)"] = lambda r: e2v.get(r["clip"], float("nan"))

    lines = ["# Prosody SER model comparison (against hand labels)\n",
             f"{len(rows)} clips, {n_y} aggressive / {len(rows) - n_y} not.\n",
             "| model | mean Y | mean N | AUC | best-threshold acc |",
             "|---|--:|--:|--:|--:|"]
    results = {}
    for name, fn in scorers.items():
        pos = [fn(r) for r in rows if r["_label"]]
        neg = [fn(r) for r in rows if not r["_label"]]
        pos = [x for x in pos if not math.isnan(x)]
        neg = [x for x in neg if not math.isnan(x)]
        if not pos or not neg:
            lines.append(f"| {name} | - | - | - | - |")
            continue
        a = auc(pos, neg)
        t, acc = best_threshold(pos, neg)
        results[name] = a
        lines.append(f"| {name} | {sum(pos)/len(pos):.3f} | {sum(neg)/len(neg):.3f} | "
                     f"**{a:.3f}** | {acc:.2f} @ t={t:.2f} |")

    # per-clip scores for inspection
    with (EVAL / "model_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["clip", "label"] + list(scorers)
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {"clip": r["clip"], "label": "Y" if r["_label"] else "N"}
            for name, fn in scorers.items():
                try:
                    row[name] = round(fn(r), 4)
                except Exception:  # noqa: BLE001
                    row[name] = ""
            w.writerow(row)

    (EVAL / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nper-clip: {EVAL / 'model_scores.csv'}")


if __name__ == "__main__":
    main()
