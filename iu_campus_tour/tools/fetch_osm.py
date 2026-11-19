#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "data"
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
USER_AGENT = "IU-Campus-Virtual-Tour/1.0 (student computer graphics project)"
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


# Parameters: --url, --center, --radius, --bbox, --raw
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="link OpenStreetMap có #map=zoom/lat/lon")
    ap.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--radius", type=float, default=350.0, help="bán kính quanh tâm (m), mặc định 350")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    ap.add_argument("--raw", action="store_true", help="lưu thêm campus_osm_raw.json (dữ liệu gốc Overpass)")
    return ap.parse_args()


def center_from_url(url):
    m = re.search(r"#map=\d+/(-?\d+\.?\d*)/(-?\d+\.?\d*)", url) or \
        re.search(r"mlat=(-?\d+\.?\d*).*?mlon=(-?\d+\.?\d*)", url)
    if not m:
        sys.exit("Không đọc được tọa độ từ link. Link phải có dạng ...#map=18/10.87/106.80")
    return float(m.group(1)), float(m.group(2))


def get_bbox(args):
    if args.bbox:
        return tuple(args.bbox)
    if args.url:
        lat, lon = center_from_url(args.url)
    elif args.center:
        lat, lon = args.center
    else:
        sys.exit("Cần một trong: --url, --center hoặc --bbox (xem --help)")
    dlat = args.radius / M_PER_DEG_LAT
    dlon = args.radius / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat)))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


# Overpass
def build_query(bbox):
    b = ",".join(f"{v:.6f}" for v in bbox)
    return f"""[out:json][timeout:90];
(
  way["building"]({b});
  relation["building"]({b});
  way["highway"]({b});
  way["amenity"~"^(parking|motorcycle_parking|bicycle_parking)$"]({b});
  way["natural"="water"]({b});
  node["natural"="tree"]({b});
);
out geom;"""


def fetch(query):
    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for url in ENDPOINTS:
        for attempt in range(2):
            try:
                print(f"Đang tải từ {url} ...")
                req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                print(f"  lỗi: {e}; thử lại...")
                time.sleep(3)
    sys.exit(f"Không tải được dữ liệu Overpass ({last}). Kiểm tra mạng rồi chạy lại.")


# Parse Overpass data
def assemble_rings(segments):
    """Nối các đoạn đường (way) rời nhau thành các vòng khép kín (cho multipolygon)."""
    segs = [list(s) for s in segments if len(s) > 1]
    rings = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[::-1][1:]
                elif s[-1] == ring[0]:
                    ring = s[:-1] + ring
                elif s[0] == ring[0]:
                    ring = s[::-1][:-1] + ring
                else:
                    continue
                segs.pop(i)
                changed = True
                break
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring[:-1])
    return rings


def parse_number(text):
    m = re.match(r"\s*(-?\d+(?:[.,]\d+)?)", str(text or ""))
    return float(m.group(1).replace(",", ".")) if m else None


def convert(osm):
    """Dữ liệu Overpass -> dict (tọa độ mét)."""
    out = {"buildings": [], "roads": [], "parking": [], "water": [], "trees": []}
    latlon = lambda geom: [(g["lat"], g["lon"]) for g in geom]

    for el in osm.get("elements", []):
        tags = el.get("tags", {})
        t, geom = el["type"], el.get("geometry")
        eid = f"{t}/{el['id']}"

        if t == "node" and tags.get("natural") == "tree":
            out["trees"].append((el["lat"], el["lon"]))
        elif t == "way" and geom:
            pts = latlon(geom)
            closed = len(pts) >= 4 and pts[0] == pts[-1]
            if "building" in tags and closed:
                out["buildings"].append(_building(eid, tags, pts[:-1], []))
            elif "highway" in tags:
                out["roads"].append({"id": eid, "kind": tags["highway"], "name": tags.get("name", ""),
                                     "points": pts})
            elif tags.get("amenity") in ("parking", "motorcycle_parking", "bicycle_parking") and closed:
                out["parking"].append({"id": eid, "ring": pts[:-1]})
            elif tags.get("natural") == "water" and closed:
                out["water"].append({"id": eid, "ring": pts[:-1]})
        elif t == "relation" and "building" in tags:
            outers = assemble_rings([latlon(m["geometry"]) for m in el.get("members", [])
                                     if m.get("role") == "outer" and m.get("geometry")])
            inners = assemble_rings([latlon(m["geometry"]) for m in el.get("members", [])
                                     if m.get("role") == "inner" and m.get("geometry")])
            for k, ring in enumerate(outers):
                out["buildings"].append(_building(f"{eid}#{k}", tags, ring, inners if k == 0 else []))
    return out


def _building(eid, tags, ring, holes):
    return {"id": eid, "name": tags.get("name", ""), "name_en": tags.get("name:en", ""),
            "kind": tags["building"], "levels": parse_number(tags.get("building:levels")),
            "height": parse_number(tags.get("height")), "ring": ring, "holes": holes}


def to_local(data, lat0, lon0):
    """(lat, lon) -> (x, z) mét: x = (lon-lon0)·111320·cos(lat0) (Đông +), z = -(lat-lat0)·110574 (Bắc = -Z)."""
    kx = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    f = lambda p: [round((p[1] - lon0) * kx, 2), round(-(p[0] - lat0) * M_PER_DEG_LAT, 2)]
    for b in data["buildings"]:
        b["outer"] = [f(p) for p in b.pop("ring")]
        b["holes"] = [[f(p) for p in h] for h in b["holes"]]
    for r in data["roads"]:
        r["points"] = [f(p) for p in r["points"]]
    for key in ("parking", "water"):
        for r in data[key]:
            r["ring"] = [f(p) for p in r["ring"]]
    data["trees"] = [f(p) for p in data["trees"]]
    return data


def recenter(data):
    """Dời gốc (0,0) về giữa nhóm tòa nhà. Trả về độ dời (sx, sz) đã trừ."""
    pts = [p for b in data["buildings"] for p in b["outer"]]
    if not pts:
        return 0.0, 0.0
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    sx, sz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    sh = lambda p: [round(p[0] - sx, 2), round(p[1] - sz, 2)]
    for b in data["buildings"]:
        b["outer"] = [sh(p) for p in b["outer"]]
        b["holes"] = [[sh(p) for p in h] for h in b["holes"]]
    for r in data["roads"]:
        r["points"] = [sh(p) for p in r["points"]]
    for key in ("parking", "water"):
        for r in data[key]:
            r["ring"] = [sh(p) for p in r["ring"]]
    data["trees"] = [sh(p) for p in data["trees"]]
    return sx, sz


def area(ring):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:] + ring[:1]))) / 2


def main():
    args = parse_args()
    bbox = get_bbox(args)
    lat0, lon0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    print(f"Khung tải: S={bbox[0]:.5f} W={bbox[1]:.5f} N={bbox[2]:.5f} E={bbox[3]:.5f}")

    osm = fetch(build_query(bbox))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.raw:
        (OUT_DIR / "campus_osm_raw.json").write_text(json.dumps(osm), encoding="utf-8")

    data = to_local(convert(osm), lat0, lon0)
    sx, sz = recenter(data)
    data["source"] = "OpenStreetMap contributors (ODbL) - https://www.openstreetmap.org/copyright"
    data["origin"] = {"lat": lat0, "lon": lon0, "shift_x": sx, "shift_z": sz,
                      "note": "x=Dong, z=Nam (Bac=-Z); goc (0,0) = giua nhom toa nha"}
    path = OUT_DIR / "campus_osm.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    print(f"\nĐã lưu {path}")
    print(f"  tòa nhà: {len(data['buildings'])}, đường: {len(data['roads'])}, bãi xe: {len(data['parking'])}, "
          f"nước: {len(data['water'])}, cây: {len(data['trees'])}")
    print("\nDanh sách tòa nhà (lớn -> nhỏ). Dùng cột 'tên' để điền building_overrides.json:")
    print(f"  {'id':<22}{'tên':<34}{'loại':<12}{'tầng':>5}{'cao(m)':>8}{'m²':>8}{'đỉnh':>6}{'lỗ':>4}")
    for b in sorted(data["buildings"], key=lambda b: -area(b["outer"])):
        print(f"  {b['id']:<22}{(b['name'] or '-')[:32]:<34}{b['kind']:<12}"
              f"{b['levels'] if b['levels'] is not None else '-':>5}"
              f"{b['height'] if b['height'] is not None else '-':>8}{area(b['outer']):>8.0f}"
              f"{len(b['outer']):>6}{len(b['holes']):>4}")


if __name__ == "__main__":
    main()
