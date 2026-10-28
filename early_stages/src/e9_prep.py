"""E9 speaker features."""

import argparse, io, json, re, wave, zipfile
from pathlib import Path
import numpy as np, torch


def main(a):
    from transformers import AutoFeatureExtractor, WavLMForXVector

    fx = AutoFeatureExtractor.from_pretrained(
        "microsoft/wavlm-base-plus-sv"
    )  # speaker-verification x-vectors
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(a.device).eval()
    feats, actor, emotion = [], [], []
    with zipfile.ZipFile(a.zip) as zf:
        for n in sorted(x for x in zf.namelist() if x.endswith(".wav")):
            m = re.search(r"03-01-(\d\d)-\d\d-\d\d-\d\d-(\d\d)\.wav$", n)
            if not m:
                continue
            with wave.open(io.BytesIO(zf.read(n))) as w:
                sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype={2: np.int16, 4: np.int32}[sw])
            x = raw.reshape(-1, ch).mean(1) / float(2 ** (8 * sw - 1))
            x = (
                np.convolve(x, np.ones(3) / 3, mode="same")[::3].astype("float32")
                if sr == 48000
                else x.astype("float32")
            )  # -> 16 kHz
            inp = fx(x, sampling_rate=16000, return_tensors="pt").to(a.device)
            with torch.no_grad():
                f = model(**inp).embeddings[0]
            feats.append(torch.nn.functional.normalize(f, dim=0).cpu().numpy())
            actor.append(int(m.group(2)))
            emotion.append(int(m.group(1)) - 1)
    F = np.stack(feats).astype(np.float32)
    actor = np.array(actor)
    emotion = np.array(emotion)
    F = F - F.mean(0)
    F /= np.linalg.norm(F, axis=1, keepdims=True)  # centred cosine space
    cent = np.stack([F[actor == k].mean(0) for k in range(1, 25)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    loo = []  # leave-one-clip-out nearest centroid
    for i in range(len(F)):
        c = cent.copy()
        k = actor[i] - 1
        nk = (actor == actor[i]).sum()
        c[k] = (cent[k] * 0 + F[actor == actor[i]].sum(0) - F[i]) / (nk - 1)
        c[k] /= np.linalg.norm(c[k])
        loo.append(int((F[i] @ c.T).argmax()) + 1 == actor[i])
    R = {"n": int(len(F)), "dim": int(F.shape[1]), "speaker_nearest_centroid_loo_acc": float(np.mean(loo))}
    np.savez(Path(a.out), emb=F, actor=actor, emotion=emotion)
    Path(a.out).with_suffix(".json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
