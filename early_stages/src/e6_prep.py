"""E6 perception: RAVDESS speech."""

import argparse, io, json, re, zipfile, urllib.request
from pathlib import Path
import numpy as np, torch
import env6
from e7_prep import probe

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = out / "ravdess_speech.zip"
    if not z.exists():
        urllib.request.urlretrieve(URL, z)
    import wave
    from transformers import ClapModel, ClapFeatureExtractor

    dev = a.device
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    fx = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
    meta, embs, batch = [], [], []
    names = [n for n in zipfile.ZipFile(z).namelist() if n.endswith(".wav")]
    with zipfile.ZipFile(z) as zf:
        for n in sorted(names):
            m = re.search(r"03-01-(\d\d)-(\d\d)-\d\d-\d\d-(\d\d)\.wav$", n)
            if not m:
                continue
            with wave.open(io.BytesIO(zf.read(n))) as w:  # RAVDESS: 16-bit PCM, 48 kHz
                sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype={2: np.int16, 4: np.int32}[sw])
            wav = (raw.reshape(-1, ch).mean(1) / float(2 ** (8 * sw - 1))).astype("float32")
            if sr != 48000:  # linear resampling fallback (not expected for RAVDESS)
                t = np.arange(0, len(wav) / sr, 1 / 48000)
                wav = np.interp(t, np.arange(len(wav)) / sr, wav).astype("float32")
            meta.append(
                (
                    env6.EMOTIONS[int(m.group(1)) - 1],
                    "normal" if m.group(2) == "01" else "strong",
                    int(m.group(3)),
                )
            )
            batch.append(wav)
            if len(batch) == 32:
                with torch.no_grad():
                    e = model.get_audio_features(
                        **{
                            k: v.to(dev)
                            for k, v in fx(batch, sampling_rate=48000, return_tensors="pt").items()
                        }
                    )
                embs.append(torch.nn.functional.normalize(e, dim=-1).float().cpu())
                batch = []
        if batch:
            with torch.no_grad():
                e = model.get_audio_features(
                    **{k: v.to(dev) for k, v in fx(batch, sampling_rate=48000, return_tensors="pt").items()}
                )
            embs.append(torch.nn.functional.normalize(e, dim=-1).float().cpu())
    E = torch.cat(embs).numpy()
    emo = np.array([env6.EMOTIONS.index(x[0]) for x in meta])
    inten = np.array([int(x[1] == "strong") for x in meta])
    actor = np.array([x[2] for x in meta])
    split = np.array(["test" if a_ in env6.TEST_ACTORS else "train" for a_ in actor])
    tr, te = split == "train", split == "test"
    R = {"n": int(len(E)), "train": int(tr.sum()), "test": int(te.sum())}
    probs = {}
    for name, y, k in (("emotion", emo, 8), ("intensity", inten, 2)):
        oof, pte = probe(E[tr], y[tr], E[te], k)
        P = np.zeros((len(y), k), np.float32)
        P[tr] = oof
        P[te] = pte
        probs[name] = P
        R[f"{name}_probe_acc_train_oof"] = float((oof.argmax(1) == y[tr]).mean())
        R[f"{name}_probe_acc_test_actors"] = float((pte.argmax(1) == y[te]).mean())
    d_true = np.array(
        [env6.distress(env6.EMOTIONS[e], "strong" if i else "normal") for e, i in zip(emo, inten)]
    )
    d_hat = np.array(
        [
            env6.distress(env6.EMOTIONS[int(pe.argmax())], "strong" if pi.argmax() else "normal")
            for pe, pi in zip(probs["emotion"], probs["intensity"])
        ]
    )
    band = lambda d: np.digitize(d, [0.4, 0.6])
    R["distress_band_acc_test_actors"] = float((band(d_true[te]) == band(d_hat[te])).mean())
    np.savez(
        out / "voice.npz",
        emb=E.astype(np.float32),
        emotion=emo,
        intensity=inten,
        actor=actor,
        split=split,
        p_emotion=probs["emotion"],
        p_intensity=probs["intensity"],
    )
    (out / "PERCEPTION.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
