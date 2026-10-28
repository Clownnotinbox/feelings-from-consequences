"""Stage F perception."""

import math
import numpy as np

CLASSES = [
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
    "bird",
]


# footprint size = box WIDTH x depth / f, divided by the median
# (27.10: a rotated square's box is up to its diagonal wide
SIZE_CAL = {
    "water_tower": 1.235,
    "pad": 0.994,
    "pickup": 1.238,
    "dropzone": 1.327,
    "solar": 1.109,
    "hangar": 1.389,
    "mast": 2.71,
    "container": 0.947,
    "tent": 0.966,
    "windsock": 0.548,
}
MIN_DEPTH = 2.5  # m: nothing the drone looks for is this close to its camera
THIN = ("mast", "water_tower", "windsock")  # poles and tanks on legs


class Perceiver:
    def __init__(self, det_weights, attr_weights, device="cuda", imgsz=960, conf=0.25):
        from ultralytics import YOLO
        import torch, train_attrs as TA

        self.det = YOLO(det_weights)
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.attr, _ = TA.build(pretrained=False)
        self.attr.load_state_dict(torch.load(attr_weights, map_location=device))
        self.attr.to(device).eval()
        self.TA = TA
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def attrs(self, crops):
        import torch

        if not crops:
            return []
        x = torch.from_numpy(np.stack(crops)).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = (x - self.mean) / self.std
        with torch.no_grad():
            o = self.TA.run(self.attr, x)
        P = {h: torch.softmax(v, 1).cpu().numpy() for h, v in o.items()}
        return [{h: P[h][i] for h in P} for i in range(len(crops))]

    def __call__(self, rgb, depth, cam, fovy):
        """cam = (pos, forward, up) world vectors; depth = planar depth in metres. Returns detections."""
        from PIL import Image

        H, W = rgb.shape[:2]
        f = (H / 2) / math.tan(math.radians(fovy) / 2)
        pos, fwd, up = (np.asarray(v, float) for v in cam)
        right = np.cross(fwd, up)
        r = self.det.predict(
            rgb[..., ::-1].copy() if False else rgb, imgsz=self.imgsz, conf=self.conf, verbose=False
        )[0]
        dets, crops, keep = [], [], []
        for c, s, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            x0, y0, x1, y1 = b
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = x1 - x0, y1 - y0
            # position from the NEAR cluster of the box's 3D points
            # (changed 26.10 after F2
            st = max(1, int(min(w, h) // 24))
            ys, xs = np.mgrid[
                int(max(0, y0 + h * 0.1)) : int(min(H, y1 - h * 0.1)) + 1 : st,
                int(max(0, x0 + w * 0.1)) : int(min(W, x1 - w * 0.1)) + 1 : st,
            ]
            dz = depth[ys, xs]
            ok = np.isfinite(dz) & (dz > 0)
            if not ok.any():
                continue
            ys, xs, dz = ys[ok], xs[ok], dz[ok]
            d_near = float(np.percentile(dz, 10))
            gap = max(4.0, 0.15 * d_near)
            if d_near < MIN_DEPTH:
                continue  # the drone's own parts or the load on its hook: never a target
            cen = (np.abs(xs - cx) < w * 0.2) & (np.abs(ys - cy) < h * 0.2)
            d_cen = float(np.median(dz[cen])) if cen.any() else float(np.median(dz))
            if (
                CLASSES[int(c)] in THIN and d_cen > d_near + gap
            ):  # a thin structure whose box centre sees the background -> its near
                m = dz < d_near + gap
                ys, xs, dz = ys[m], xs[m], dz[m]
                rays = (
                    fwd[None]
                    + ((xs - W / 2) / f)[:, None] * right[None]
                    - ((ys - H / 2) / f)[:, None] * up[None]
                )
                Q = pos[None] + dz[:, None] * rays
                P = np.array([np.median(Q[:, 0]), np.median(Q[:, 1]), np.percentile(Q[:, 2], 5)])
                d = float(np.median(dz))
            else:  # a solid / flat object
                d = d_cen
                P = pos + d * (fwd + ((cx - W / 2) / f) * right - ((cy - H / 2) / f) * up)
            det = {
                "cls": CLASSES[int(c)],
                "conf": float(s),
                "x": float(P[0]),
                "y": float(P[1]),
                "z": float(P[2]),
                "px": float(min(w, h)),
                "size": float(w * d / f / SIZE_CAL.get(CLASSES[int(c)], 1.0)),
                "height": float(h * d / f),
                "box": [float(x0), float(y0), float(x1), float(y1)],
                "depth": d,
            }
            if CLASSES[int(c)] != "bird" and min(w, h) >= 10:
                mx, my = int(w * 0.2) + 2, int(h * 0.2) + 2
                crop = rgb[max(0, int(y0) - my) : int(y1) + my, max(0, int(x0) - mx) : int(x1) + mx]
                crops.append(np.array(Image.fromarray(crop).resize((112, 112))))
                keep.append(len(dets))
            det["attrs"] = {}
            dets.append(det)
        for i, a in zip(keep, self.attrs(crops)):
            dets[i]["attrs"] = a
        return dets
