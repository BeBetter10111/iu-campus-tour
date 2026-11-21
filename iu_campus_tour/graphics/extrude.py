"""Đùn (extrude) đường viền mặt bằng 2D thành khối tòa nhà: tường + mái.

Hệ tọa độ: x = Đông, y = lên, z = Nam (Bắc là -Z, khớp camera mặc định nhìn về -Z).
Vì vậy trên mặt phẳng (x, z) "ngược chiều kim đồng hồ" theo công thức diện tích có dấu
tương ứng với pháp tuyến hướng XUỐNG của tam giác; các hàm dưới đây đã tính sẵn và
được kiểm thử bằng tests/test_osm_pipeline.py (mái hướng lên, tường hướng ra ngoài).
"""

import numpy as np

EPS = 1e-9


# ------------------------------------------------------------------ hình học 2D
def signed_area(poly):
    """Diện tích có dấu:  A = ½ Σ (x_i·z_{i+1} - x_{i+1}·z_i).  A > 0: đỉnh xếp "ngược chiều" trong (x,z)."""
    p = np.asarray(poly, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return 0.5 * float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))


def clean_ring(poly, tol=0.02):
    """Bỏ điểm trùng (khoảng cách < tol m) kể cả điểm đóng vòng lặp lại điểm đầu."""
    pts = []
    for p in np.asarray(poly, dtype=np.float64):
        if not pts or np.hypot(*(p - pts[-1])) >= tol:
            pts.append(p)
    if len(pts) > 1 and np.hypot(*(pts[0] - pts[-1])) < tol:
        pts.pop()
    return np.array(pts)


def _cross(a, b, c):
    """(b-a) × (c-a) trong mặt phẳng; > 0 nếu a→b→c rẽ trái (đỉnh lồi khi đa giác có A > 0)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_triangle(p, a, b, c, eps=1e-9):
    d1, d2, d3 = _cross(a, b, p), _cross(b, c, p), _cross(c, a, p)
    return d1 >= -eps and d2 >= -eps and d3 >= -eps


def triangulate(poly):
    """Chia đa giác đơn (có thể lõm) thành tam giác bằng thuật toán "cắt tai" (ear clipping).

    Tai = đỉnh lồi mà tam giác (đỉnh trước, đỉnh, đỉnh sau) không chứa đỉnh nào khác.
    Cắt tai rồi lặp lại cho tới khi còn 3 đỉnh. Độ phức tạp O(n³), thừa nhanh cho tòa nhà (n < 100).

    Trả về (list[(i, j, k)], số_lần_phải_cắt_ép). Số lần cắt ép > 0 nghĩa là đường viền có điểm
    suy biến (tự cắt, thẳng hàng...) -> mái có thể sai nhẹ.
    """
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    if n < 3:
        return [], 0
    idx = list(range(n))
    if signed_area(p) < 0:                  # Thuật toán dưới đây giả định A > 0
        idx.reverse()
    tris, forced = [], 0
    while len(idx) > 3:
        m = len(idx)
        ear = None
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = p[i0], p[i1], p[i2]
            if _cross(a, b, c) <= EPS:
                continue                    # Đỉnh lõm hoặc thẳng hàng: không thể là tai
            if any(_in_triangle(p[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            ear = k
            break
        if ear is None:                     # Suy biến: cắt ép đỉnh lồi nhất để không bị kẹt
            forced += 1
            ear = max(range(m), key=lambda k: _cross(p[idx[k - 1]], p[idx[k]], p[idx[(k + 1) % m]]))
        i0, i1, i2 = idx[ear - 1], idx[ear], idx[(ear + 1) % len(idx)]
        tris.append((i0, i1, i2))
        del idx[ear]
    tris.append(tuple(idx))
    return tris, forced


# ------------------------------------------------------------------- dựng mesh
def side_of_normal(nx, nz):
    """Hướng la bàn của mặt tường theo pháp tuyến ngoài (Bắc = -Z, Đông = +X)."""
    if abs(nx) >= abs(nz):
        return "E" if nx > 0 else "W"
    return "S" if nz > 0 else "N"


def build_walls(outer, holes, height, tile_w, floor_h, kind_of_side=None, default_kind="main", base=0.0):
    """Dựng tường. Mỗi cạnh của đường viền là một hình chữ nhật đứng.

        Pháp tuyến ngoài của cạnh p_i→p_j (đỉnh xếp theo chiều A < 0):  n = (-dz, 0, dx) / |e|
        UV: u = quãng_đường_dọc_tường / tile_w ;  v = độ_cao / floor_h  (mỗi tầng = 1 ô texture)

    base > 0: khối NÂNG LÊN (cầu nối, mái đua...): tường chạy từ y = base lên y = height; v bắt đầu ở
    base/floor_h nên các tầng vẫn thẳng hàng với tòa nhà bên cạnh.

    Trả về dict {loại_mặt_tiền: (vertices (N,8) float32, indices uint32)}.
    kind_of_side: dict {"N"/"E"/"S"/"W": tên_mặt_tiền} để mỗi hướng tường một kiểu khác nhau.
    """
    groups = {}
    top_v = height / floor_h
    bot_v = base / floor_h

    def add_ring(ring, outward_negative):
        ring = np.asarray(ring, dtype=np.float64)
        # Vòng ngoài: A < 0 để n = (-dz, dx) hướng ra ngoài. Vòng lỗ: ngược lại (ra khỏi khối đặc = vào lỗ).
        if (signed_area(ring) < 0) != outward_negative:
            ring = ring[::-1]
        s = 0.0
        for i in range(len(ring)):
            p0, p1 = ring[i], ring[(i + 1) % len(ring)]
            dx, dz = p1[0] - p0[0], p1[1] - p0[1]
            length = float(np.hypot(dx, dz))
            if length < 1e-3:
                continue
            nx, nz = -dz / length, dx / length
            kind = default_kind
            if kind_of_side:
                kind = kind_of_side.get(side_of_normal(nx, nz), default_kind)
            u0, u1 = s / tile_w, (s + length) / tile_w
            s += length
            v, idx = groups.setdefault(kind, ([], []))
            start = len(v)                      # chỉ số đỉnh đầu tiên của hình chữ nhật này (KHÔNG đặt tên là `base`: trùng tham số độ cao đáy)
            v += [[p0[0], base,   p0[1], nx, 0, nz, u0, bot_v],
                  [p1[0], base,   p1[1], nx, 0, nz, u1, bot_v],
                  [p1[0], height, p1[1], nx, 0, nz, u1, top_v],
                  [p0[0], height, p0[1], nx, 0, nz, u0, top_v]]
            idx += [start, start + 1, start + 2, start, start + 2, start + 3]

    add_ring(outer, True)
    for h in holes or []:
        add_ring(h, False)
    return {k: (np.array(v, np.float32), np.array(i, np.uint32)) for k, (v, i) in groups.items()}


def build_roof(outer, height, tile=6.0, both_faces=False):
    """Mái phẳng ở độ cao `height`, chia tam giác bằng ear clipping.

    both_faces=True: thêm mặt dưới (dùng cho mái hiên/nhà để xe không có tường, nhìn từ dưới lên vẫn thấy).
    Trả về (vertices, indices, số_lần_cắt_ép).
    """
    outer = np.asarray(outer, dtype=np.float64)
    tris, forced = triangulate(outer)
    verts = [[x, height, z, 0, 1, 0, x / tile, z / tile] for x, z in outer]
    idx = []
    for i0, i1, i2 in tris:
        # Tam giác (i0,i1,i2) có A > 0 trong (x,z) -> pháp tuyến ngược hướng +Y -> đảo thứ tự để mặt trên hướng lên
        idx += [i0, i2, i1]
    if both_faces:
        n = len(verts)
        verts += [[x, height, z, 0, -1, 0, x / tile, z / tile] for x, z in outer]
        for i0, i1, i2 in tris:
            idx += [n + i0, n + i1, n + i2]
    return np.array(verts, np.float32), np.array(idx, np.uint32), forced


def build_underside(outer, base, tile=6.0):
    """Mặt DƯỚI của khối nâng lên (nhìn từ dưới lên, ví dụ đi bộ dưới cầu nối): pháp tuyến hướng xuống.
    Tam giác từ ear clipping có A > 0 -> pháp tuyến đã hướng xuống, giữ nguyên thứ tự đỉnh."""
    outer = np.asarray(outer, dtype=np.float64)
    tris, _ = triangulate(outer)
    verts = [[x, base, z, 0, -1, 0, x / tile, z / tile] for x, z in outer]
    idx = [k for tri in tris for k in tri]
    return np.array(verts, np.float32), np.array(idx, np.uint32)


DOOR_SNAP_MAX = 8.0     # m: cửa chỉ định cách tường xa hơn mức này thì bỏ qua (và báo cáo)
DOOR_HEIGHT = 2.7       # m
DOOR_MIN_PIECE = 0.15   # m: chừa mảng tường tối thiểu ở hai đầu cạnh


def _quad(v, idx, p0, p1, y0, y1, n, u0, u1, v0, v1, interior=False):
    """Một hình chữ nhật đứng từ p0 tới p1 (điểm 2D (x,z)) cao y0..y1.
    interior=True: mặt NGƯỢC lại (nhìn từ trong phòng): đảo thứ tự đỉnh và pháp tuyến -> không z-fighting với mặt ngoài
    đặt trùng chỗ, vì mỗi mặt chỉ được vẽ khi nhìn từ phía của nó (face culling)."""
    start = len(v)
    if not interior:
        v += [[p0[0], y0, p0[1], n[0], 0, n[1], u0, v0], [p1[0], y0, p1[1], n[0], 0, n[1], u1, v0],
              [p1[0], y1, p1[1], n[0], 0, n[1], u1, v1], [p0[0], y1, p0[1], n[0], 0, n[1], u0, v1]]
    else:
        m = (-n[0], -n[1])
        v += [[p1[0], y0, p1[1], m[0], 0, m[1], u1, v0], [p0[0], y0, p0[1], m[0], 0, m[1], u0, v0],
              [p0[0], y1, p0[1], m[0], 0, m[1], u0, v1], [p1[0], y1, p1[1], m[0], 0, m[1], u1, v1]]
    idx += [start, start + 1, start + 2, start, start + 2, start + 3]


def _merge_gaps(gaps):
    out = []
    for a, b in sorted(gaps):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def build_walls_ground(outer, height, tile_w, floor_h, default_kind, ground_kind, kind_of_side=None,
                       doors=None, door_h=DOOR_HEIGHT, interior=False):
    """Tường có TẦNG TRỆT RIÊNG và CỬA RA VÀO.

    - Tầng trệt (y 0..floor_h): mặt tiền `ground_kind` (ví dụ tường trắng không cửa sổ). Các tầng trên dùng kiểu của từng hướng.
    - doors: [{"at": [x, z], "width": m}]. Mỗi cửa được "dính" vào cạnh tường gần nhất (tối đa DOOR_SNAP_MAX m) và đặt giữa
      điểm chiếu; cạnh bị cắt thành các mảng tường + một ô trống cao door_h, phía trên ô trống là mảng "lanh tô".
    - interior=True: thêm mặt trong của tường tầng trệt (màu sáng), sàn và trần tầng trệt -> đi vào bên trong được.
      Tường độ dày 0: mặt trong và mặt ngoài trùng vị trí nhưng quay lưng nhau.

    Trả về dict: walls {kind: (v, i)}, interior {"walls","floor","ceiling"} | None, wall_edges (M,4) | None, doors [báo cáo].
    """
    ring = np.asarray(outer, dtype=np.float64)
    if signed_area(ring) > 0:
        ring = ring[::-1]                                   # A < 0: n = (-dz, dx) hướng ra ngoài
    n_e = len(ring)
    edges = []
    for i in range(n_e):
        p0, p1 = ring[i], ring[(i + 1) % n_e]
        L = float(np.hypot(*(p1 - p0)))
        if L < 1e-3:
            edges.append(None)
            continue
        d = (p1 - p0) / L
        edges.append((p0, p1, L, d, (-d[1], d[0])))

    gaps = {i: [] for i in range(n_e)}
    report = []
    for door in doors or []:
        at = np.asarray(door["at"], dtype=np.float64)
        best = None
        for i, e in enumerate(edges):
            if e is None:
                continue
            p0, p1, L, d, nrm = e
            t = float(np.clip((at - p0) @ d, 0.0, L))
            dist = float(np.hypot(*(at - (p0 + d * t))))
            if best is None or dist < best[0]:
                best = (dist, i, t)
        if best is None or best[0] > DOOR_SNAP_MAX:
            report.append({"at": tuple(at), "ok": False, "why": f"cách tường {best[0]:.1f} m" if best else "không có tường"})
            continue
        dist, i, t = best
        p0, p1, L, d, nrm = edges[i]
        w = min(float(door.get("width", 2.6)), L - 2 * DOOR_MIN_PIECE)
        if w < 0.8:
            report.append({"at": tuple(at), "ok": False, "why": "cạnh tường quá ngắn"})
            continue
        s0 = min(max(DOOR_MIN_PIECE, t - w / 2), L - DOOR_MIN_PIECE - w)
        gaps[i].append((s0, s0 + w))
        c = p0 + d * (s0 + w / 2)
        report.append({"at": (float(c[0]), float(c[1])), "normal": (float(nrm[0]), float(nrm[1])),
                       "width": w, "snap_dist": dist, "ok": True})

    band_top = min(floor_h, height)
    has_upper = height > band_top + 0.05
    walls, iv, ii, solid = {}, [], [], []
    s_cum = 0.0
    for i, e in enumerate(edges):
        if e is None:
            continue
        p0, p1, L, d, nrm = e
        kind_up = (kind_of_side or {}).get(side_of_normal(*nrm), default_kind)
        gs = _merge_gaps(gaps[i])
        pieces, cur = [], 0.0
        for a, b in gs:
            if a > cur + 1e-6:
                pieces.append((cur, a))
            cur = b
        if cur < L - 1e-6:
            pieces.append((cur, L))
        gv, gi = walls.setdefault(ground_kind, ([], []))
        for a, b in pieces:                                  # mảng tường đặc ở tầng trệt
            pa, pb = p0 + d * a, p0 + d * b
            _quad(gv, gi, pa, pb, 0.0, band_top, nrm, (s_cum + a) / tile_w, (s_cum + b) / tile_w, 0.0, band_top / floor_h)
            if interior:
                _quad(iv, ii, pa, pb, 0.0, band_top, nrm, (s_cum + a) / tile_w, (s_cum + b) / tile_w, 0.0, band_top / floor_h, True)
            solid.append([pa[0], pa[1], pb[0], pb[1]])
        if door_h < band_top - 0.05:                         # lanh tô phía trên ô cửa
            for a, b in gs:
                pa, pb = p0 + d * a, p0 + d * b
                _quad(gv, gi, pa, pb, door_h, band_top, nrm, (s_cum + a) / tile_w, (s_cum + b) / tile_w,
                      door_h / floor_h, band_top / floor_h)
                if interior:
                    _quad(iv, ii, pa, pb, door_h, band_top, nrm, (s_cum + a) / tile_w, (s_cum + b) / tile_w,
                          door_h / floor_h, band_top / floor_h, True)
        if has_upper:                                        # các tầng trên: nguyên cạnh
            uv_, ui_ = walls.setdefault(kind_up, ([], []))
            _quad(uv_, ui_, p0, p1, band_top, height, nrm, s_cum / tile_w, (s_cum + L) / tile_w,
                  band_top / floor_h, height / floor_h)
        s_cum += L

    out = {"walls": {k: (np.array(v, np.float32), np.array(i_, np.uint32)) for k, (v, i_) in walls.items()},
           "interior": None, "wall_edges": None, "doors": report}
    if interior:
        tris, _ = triangulate(ring)
        fl_v = np.array([[x, 0.02, z, 0, 1, 0, x / 2.0, z / 2.0] for x, z in ring], np.float32)
        fl_i = np.array([k for tri in tris for k in (tri[0], tri[2], tri[1])], np.uint32)         # hướng lên
        ce_v = np.array([[x, band_top, z, 0, -1, 0, x / 6.0, z / 6.0] for x, z in ring], np.float32)
        ce_i = np.array([k for tri in tris for k in tri], np.uint32)                              # hướng xuống
        out["interior"] = {"walls": (np.array(iv, np.float32), np.array(ii, np.uint32)),
                           "floor": (fl_v, fl_i), "ceiling": (ce_v, ce_i)}
        out["wall_edges"] = np.array(solid, dtype=np.float64) if solid else np.zeros((0, 4))
    return out


def extrude_arrays(outer, holes, height, tile_w=10.8, floor_h=3.6, roof_tile=6.0,
                   kind_of_side=None, default_kind="main", roof_only=False, base=0.0,
                   ground_kind=None, doors=None, enterable=False):
    """Toàn bộ hình học một tòa nhà dưới dạng mảng NumPy (không cần GPU -> dễ kiểm thử).

    base > 0: khối nâng lên từ y = base (có thêm mặt dưới "bottom").
    Trả về dict:  {"walls": {kind: (v, i)}, "roof": (v, i), "bottom": (v, i) | None, "forced": int, "holes_ignored": bool}
    roof_only=True: chỉ có tấm mái (2 mặt) ở độ cao height, không tường (mái che, nhà để xe).
    """
    outer = clean_ring(outer)
    holes = [clean_ring(h) for h in (holes or []) if len(h) >= 3]
    if len(outer) < 3:
        raise ValueError("Đường viền có ít hơn 3 đỉnh")
    extra = {"interior": None, "wall_edges": None, "doors": []}
    if roof_only:
        walls = {}
    elif (ground_kind or doors or enterable) and base == 0:
        g = build_walls_ground(outer, height, tile_w, floor_h, default_kind, ground_kind or default_kind,
                               kind_of_side, doors, interior=enterable)
        walls, extra = g["walls"], {k: g[k] for k in ("interior", "wall_edges", "doors")}
    else:
        walls = build_walls(outer, holes, height, tile_w, floor_h, kind_of_side, default_kind, base=base)
    rv, ri, forced = build_roof(outer, height, roof_tile, both_faces=roof_only)
    bottom = build_underside(outer, base, roof_tile) if (base > 0 and not roof_only) else None
    return {"walls": walls, "roof": (rv, ri), "bottom": bottom, "forced": forced,
            "holes_ignored": bool(holes), **extra}
