"""Stage F, check F2 (plan F–G sec.
usage: eval_grounding.py DET.pt ATTRS.pt PARSER_ADAPTER BASE_MODEL OUT.json [N_PER_PLACE] [SEED0]"""

import json, math, sys, time, zlib
import numpy as np


def look_views(cx, cy, z, n_yaw=8, tilts=(-30.0, -55.0)):
    v = [(cx, cy, z, math.degrees(2 * math.pi * k / n_yaw), t) for k in range(n_yaw) for t in tilts]
    return v + [(cx, cy, z, 0.0, -89.0)]


def close_views(h, hx, hy):
    """3 views from 20 m altitude, 25 m away from a candidate, looking at it."""
    out = []
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.4
        px, py = hx + 25 * math.cos(a), hy + 25 * math.sin(a)
        yaw = math.degrees(math.atan2(hy - py, hx - px))
        tilt = -math.degrees(math.atan2(20 - h, 25))
        out.append((px, py, 20.0, yaw, tilt))
    return out


class Renderer:
    def __init__(self, W, objs, files, fovy=70.0, size=(1280, 960)):
        import mujoco, city3d as C

        W.objects = objs
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(
            C.to_mjcf(W, W._static + "".join(o.xml for o in objs), W._assets), files
        )
        self.d = mujoco.MjData(self.m)
        self.m.vis.global_.fovy = fovy
        self.fovy = fovy
        self.W, self.H = size
        self.R = mujoco.Renderer(self.m, size[1], size[0], max_geom=30000)
        mujoco.mj_forward(self.m, self.d)
        self.m.vis.quality.offsamples = 0
        self.Rs = mujoco.Renderer(self.m, size[1], size[0], max_geom=30000)
        self.m.vis.quality.offsamples = 4  # labels: no multisampling
        self.Rs.enable_segmentation_rendering()
        names = {o.oid: k + 1 for k, o in enumerate(objs)}
        self.lab = np.zeros(self.m.ngeom + 1, np.int32)
        for g in range(self.m.ngeom):
            n = self.m.geom(g).name
            if n and n.split("_")[0] in names:
                self.lab[g] = names[n.split("_")[0]]

    def view(self, px, py, pz, yaw, tilt):
        mj = self.mj
        m, d = self.m, self.d
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        fwd = np.array(
            [
                math.cos(math.radians(tilt)) * math.cos(math.radians(yaw)),
                math.cos(math.radians(tilt)) * math.sin(math.radians(yaw)),
                math.sin(math.radians(tilt)),
            ]
        )
        cam.distance = 10.0
        cam.lookat[:] = np.array([px, py, pz]) + fwd * 10.0
        cam.azimuth = yaw
        cam.elevation = tilt
        d.qpos[:3] = [px, py, pz + 4.0]
        mj.mj_forward(m, d)
        self.R.update_scene(d, camera=cam)
        rgb = self.R.render().copy()
        gc = self.R.scene.camera[0]
        cpos = np.array(gc.pos, float)
        cf = np.array(gc.forward, float)
        cu = np.array(gc.up, float)
        self.R.enable_depth_rendering()
        self.R.update_scene(d, camera=cam)
        depth = self.R.render().copy()
        self.R.disable_depth_rendering()
        self.Rs.update_scene(d, camera=cam)
        seg = self.Rs.render().copy()
        gid = seg[..., 0]
        lab = np.where(
            (seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM)) & (gid >= 0), self.lab[np.clip(gid, 0, m.ngeom)], 0
        )
        return rgb, depth, (cpos, cf, cu), lab


def episode(W, mis, files, perc, parsed, true_parse):
    import grounding as G, missions as M, mission_parser as MP

    MPW = MP.well_formed
    objs = mis["objects"]
    R = Renderer(W, objs, files)
    tgt = next(o for o in objs if o.oid == mis["steps"][0]["target"])
    ti = objs.index(tgt) + 1
    rough = mis["rough"][0]
    dets = []
    seen_px = 0
    MAP = W.map
    kinds = G_KINDS

    def take(views):
        nonlocal seen_px
        for v in views:
            rgb, depth, cam, lab = R.view(*v)
            seen_px = max(seen_px, int((lab == ti).sum()))
            dets.extend(perc(rgb, depth, cam, R.fovy))

    take(look_views(rough[0], rough[1], 35.0))
    res = {}
    for name, parse in (("parsed", parsed), ("true_parse", true_parse)):
        if not MPW(parse) or not parse.get("steps"):
            res[name] = {"pick_ok": False, "decision": "no_parse", "rounds": 0}
            continue
        step = parse["steps"][0]
        local = list(dets)
        rounds = 0
        while True:
            hyps = G.fuse([d for d in local if d["cls"] != "bird"])
            ranked, p_none = G.rank(hyps, step, rough, MAP, kinds)
            dec, best = G.decide(ranked, p_none)
            if dec != "look" or rounds >= 2:
                break
            rounds += 1
            extra = []
            for p, h, _ in ranked[:2]:
                for v in close_views(h["z"], h["x"], h["y"]):
                    rgb, depth, cam, lab = R.view(*v)
                    seen_px = max(seen_px, int((lab == ti).sum()))
                    extra.extend(perc(rgb, depth, cam, R.fovy))
            local += extra
            if name == "parsed":
                dets.extend(extra)
        top = ranked[0][1] if ranked else None
        hit = lambda h: h is not None and math.hypot(h["x"] - tgt.x, h["y"] - tgt.y) < 3.0 + 0.5 * tgt.radius
        res[name] = {
            "pick_ok": hit(top),
            "decision": dec,
            "go_ok": hit(best) if dec == "go" else None,
            "rounds": rounds,
            "post": round(ranked[0][0], 3) if ranked else None,
            "n_hyp": len(hyps),
        }
    hyps = G.fuse([d for d in dets if d["cls"] != "bird"])
    near = lambda H: min(H, key=lambda h: math.hypot(h["x"] - rough[0], h["y"] - rough[1])) if H else None
    hit = lambda h: h is not None and math.hypot(h["x"] - tgt.x, h["y"] - tgt.y) < 3.0 + 0.5 * tgt.radius
    res["coord_only"] = {"pick_ok": hit(near(hyps))}
    res["kind_only"] = {"pick_ok": hit(near([h for h in hyps if h["cls"] == tgt.kind]))}
    res["visible_px"] = seen_px
    R.R.close()
    R.Rs.close()
    return res


G_KINDS = [
    "pad",
    "dropzone",
    "pickup",
    "hangar",
    "water_tower",
    "mast",
    "tent",
    "container",
    "solar",
    "windsock",
]


def main():
    det, attrs, adapter, base, out = sys.argv[1:6]
    n = int(sys.argv[6]) if len(sys.argv) > 6 else 60
    s0 = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    import city3d as C, missions as M, perceive as PC, mission_parser as MP, train_parser as TP, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    files = C.texture_files()
    perc = PC.Perceiver(det, attrs)
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token
    lm = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda"), adapter
    ).eval()
    rows = []
    t0 = time.time()
    for place in ("tokyo_res", "kazan_novosav"):
        for k in range(n):
            seed = s0 + k
            W, static, assets = C.build(place, 500 + seed)
            W._static, W._assets = static, assets
            rng = np.random.default_rng(zlib.crc32(f"F2/{place}/{seed}".encode()))
            bank = "test" if k % 3 != 0 else "train"
            mis = M.make(W, rng, bank=bank, detail=str(rng.choice(["medium", "full"])), two_step=False)
            if mis is None:
                continue
            out_txt = TP.generate(lm, tok, [{"text": mis["text"]}], bs=1)[0]
            parsed = MP.parse_output(out_txt)
            r = episode(W, mis, files, perc, parsed, mis["truth"])
            tgt = next(o for o in mis["objects"] if o.oid == mis["steps"][0]["target"])
            twins = sum(
                1
                for o in mis["objects"]
                if o.kind == tgt.kind and o is not tgt and o.facts.get("colour") == tgt.facts.get("colour")
            )
            r.update(
                place=place,
                seed=seed,
                bank=bank,
                lang=mis["lang"],
                detail=mis["detail"],
                kind=tgt.kind,
                twins=twins,
                parse_exact=MP.score(parsed, mis["truth"])["exact"],
                text=mis["text"],
            )
            rows.append(r)
            print(
                len(rows),
                place,
                k,
                {kk: r[kk] for kk in ("visible_px", "twins", "parse_exact")},
                r["parsed"],
                r["coord_only"],
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
            with open(out + ".jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summarize(rows, out)


def summarize(rows, out):
    def boot(x, y=None, n=4000):
        x = np.asarray(x, float)
        dd = x if y is None else x - np.asarray(y, float)
        if not len(dd):
            return None
        bs = dd[np.random.default_rng(0).integers(len(dd), size=(n, len(dd)))].mean(1)
        return [
            round(float(dd.mean()), 3),
            round(float(np.percentile(bs, 2.5)), 3),
            round(float(np.percentile(bs, 97.5)), 3),
            len(dd),
        ]

    vis = [r for r in rows if r["visible_px"] >= 30]
    groups = {
        "visible": vis,
        "visible_hard": [r for r in vis if r["twins"] > 0],
        "visible_hard_novel": [r for r in vis if r["twins"] > 0 and r["bank"] == "test"],
        "visible_novel": [r for r in vis if r["bank"] == "test"],
        "not_visible": [r for r in rows if r["visible_px"] < 30],
    }
    res = {"n": len(rows)}
    for g, R in groups.items():
        res[g] = {
            "pipeline": boot([r["parsed"]["pick_ok"] for r in R]),
            "true_parse": boot([r["true_parse"]["pick_ok"] for r in R]),
            "coord_only": boot([r["coord_only"]["pick_ok"] for r in R]),
            "kind_only": boot([r["kind_only"]["pick_ok"] for r in R]),
            "pipeline - coord_only": boot(
                [r["parsed"]["pick_ok"] for r in R], [r["coord_only"]["pick_ok"] for r in R]
            ),
            "go_share": round(float(np.mean([r["parsed"]["decision"] == "go" for r in R])), 3) if R else None,
            "go_correct": (
                round(float(np.mean([r["parsed"]["go_ok"] for r in R if r["parsed"]["decision"] == "go"])), 3)
                if any(r["parsed"]["decision"] == "go" for r in R)
                else None
            ),
        }
    G = res["visible_hard_novel"]
    res["F2_rule"] = {
        "pipeline>=0.85": bool(G["pipeline"] and G["pipeline"][0] >= 0.85),
        "vs_coord>=0.3_lo>0": bool(
            G["pipeline - coord_only"]
            and G["pipeline - coord_only"][0] >= 0.3
            and G["pipeline - coord_only"][1] > 0
        ),
    }
    open(out, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
