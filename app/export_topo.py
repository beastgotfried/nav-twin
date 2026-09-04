"""Generate the dashboard's contour field as a seamless SVG.

Writes dashboard/public/topo.svg.

WHY THIS IS COMPUTED RATHER THAN AN ASSET OR A FILTER
-----------------------------------------------------
Three versions of this background were tried and the first two both failed for
reasons worth writing down.

1. An SVG `feTurbulence` filter with a discrete transfer function. Cheap, no
   asset, and it does produce closed bands. But quantised noise is not
   contours: the bands have no consistent line weight, they terminate in
   mid-air, and at any real size it reads as a filter effect rather than as a
   map.

2. A bitmap tile exported from the design file. Genuinely drawn, so the line
   quality was right, but it is a raster: it softens when scaled, it cannot be
   restyled, and 116 kB of PNG has to ship for something that is ten thousand
   line segments.

3. This. A periodic scalar field, contoured by marching squares, emitted as
   SVG paths.

The field is a sum of sinusoids whose wave vectors are INTEGER multiples of
the tile frequency:

    h(x, y) = sum_k a_k * sin(2*pi*(u_k*x + v_k*y) + p_k),   u_k, v_k integers

Every term has period exactly 1 in both axes, so the sum does too, so the tile
is seamless BY CONSTRUCTION rather than by blending or mirroring its edges.
That matters because the field drifts continuously behind the dashboard: a
tile that only nearly matches shows a seam once per cycle, and a seam sliding
across the screen is exactly the kind of thing that reads as a rendering
fault.

Contours are then extracted at evenly spaced levels. Every fifth level is an
index contour and is drawn heavier, which is the convention on a real
topographic sheet and is what makes the result legible as elevation rather
than as abstract swirls.

The output is unlabelled and carries no scale bar, no coordinates and no place
name, deliberately. It is atmosphere, not cartography: the moment it asserts
terrain it is making a claim this project does not model, on a system whose
whole argument is that everything on screen is computed. See CLAUDE.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "dashboard" / "public" / "topo.svg"

# Tile resolution in SVG user units. The paths are vector, so this only sets
# the coordinate space and the sampling grid, not the output quality.
SIZE = 1000
# Samples per axis for the marching-squares grid. Higher is smoother contours
# and a bigger file; 360 keeps the curves clean at full-screen size and the
# output near 100 kB.
N = 360

# Contour levels across the field's range, and how often an index contour
# falls. Both chosen by looking at the render: fewer levels read as blobs,
# many more turn the field into hatching.
LEVELS = 28
INDEX_EVERY = 5

SEED = 7


def field(n: int, seed: int) -> np.ndarray:
    """A periodic height field on the unit square, sampled n x n.

    Integer wave vectors are what make it periodic. Amplitudes fall off with
    spatial frequency (roughly 1/|k|), which is what gives a natural landscape
    its mix of broad forms and small detail instead of uniform ripple.
    """
    rng = np.random.default_rng(seed)
    # Sample on [0, 1) with the endpoint EXCLUDED. Including it would repeat
    # the first column and row and put a visible double line at the seam.
    xs = np.arange(n) / n
    x, y = np.meshgrid(xs, xs, indexing="xy")

    h = np.zeros((n, n), dtype=float)
    waves: list[tuple[int, int]] = []
    # Low frequencies carry the landform, higher ones the detail.
    for u in range(-4, 5):
        for v in range(-4, 5):
            if u == 0 and v == 0:
                continue
            if u * u + v * v > 16:
                continue
            # Only half the plane: k and -k are the same wave.
            if (u, v) < (0, 0):
                continue
            waves.append((u, v))

    for u, v in waves:
        k = math.hypot(u, v)
        amp = 1.0 / (k ** 1.85)
        phase = rng.uniform(0, 2 * math.pi)
        h += amp * np.sin(2 * math.pi * (u * x + v * y) + phase)

    # Normalise to 0..1 so the level spacing is meaningful.
    h -= h.min()
    h /= h.max()
    return h


def _interp(a: float, b: float, level: float) -> float:
    """Where between two corner samples the level crosses. Guarded against
    the degenerate equal-corner case, which would divide by zero."""
    if abs(b - a) < 1e-12:
        return 0.5
    return (level - a) / (b - a)


def contour_segments(h: np.ndarray, level: float) -> list[tuple[float, float, float, float]]:
    """Marching squares.

    Returns unordered line segments in grid coordinates, which stitch() then
    chains into polylines.

    The grid is treated as WRAPPING, so a contour running off the right edge
    continues at the left. Without that the tile has contours that stop dead at
    its border and the seam becomes visible the moment it drifts.
    """
    n = h.shape[0]
    segs: list[tuple[float, float, float, float]] = []

    # Corner samples of every cell, with wraparound on the far edges.
    tl = h
    tr = np.roll(h, -1, axis=1)
    bl = np.roll(h, -1, axis=0)
    br = np.roll(np.roll(h, -1, axis=0), -1, axis=1)

    case = (
        (tl > level).astype(np.uint8)
        | ((tr > level).astype(np.uint8) << 1)
        | ((br > level).astype(np.uint8) << 2)
        | ((bl > level).astype(np.uint8) << 3)
    )

    ys, xs = np.nonzero((case != 0) & (case != 15))
    for gy, gx in zip(ys.tolist(), xs.tolist()):
        c = int(case[gy, gx])
        a, b, d, e = (
            float(tl[gy, gx]),
            float(tr[gy, gx]),
            float(br[gy, gx]),
            float(bl[gy, gx]),
        )
        # Crossing points on each edge, in cell-local coordinates.
        top = (gx + _interp(a, b, level), float(gy))
        right = (float(gx + 1), gy + _interp(b, d, level))
        bottom = (gx + _interp(e, d, level), float(gy + 1))
        left = (float(gx), gy + _interp(a, e, level))

        # The saddle cases (5 and 10) are resolved consistently rather than by
        # the cell average. At this line weight the difference is invisible and
        # the consistent choice avoids isolated stray segments.
        table = {
            1: [(left, top)],
            2: [(top, right)],
            3: [(left, right)],
            4: [(right, bottom)],
            5: [(left, top), (right, bottom)],
            6: [(top, bottom)],
            7: [(left, bottom)],
            8: [(left, bottom)],
            9: [(top, bottom)],
            10: [(left, bottom), (top, right)],
            11: [(right, bottom)],
            12: [(left, right)],
            13: [(top, right)],
            14: [(left, top)],
        }
        for p, q in table[c]:
            segs.append((p[0], p[1], q[0], q[1]))
    return segs


def stitch(segs: list[tuple[float, float, float, float]], scale: float) -> list[str]:
    """Chain segments that share an endpoint into polylines.

    Marching squares emits each cell's crossing independently, so a single
    contour ring arrives as a few hundred disconnected two-point segments.
    Written out that way the file was 920 kB of "M...L..." for a background.
    Chaining them into runs emits one M and many L, which is the same picture
    an order of magnitude smaller, and it also lets stroke-linejoin do its job
    at the corners.

    Endpoints are matched on a rounded key. The crossings are computed from the
    same corner samples on both sides of a shared edge, so they agree to
    floating-point noise and quantising to 1e-4 of a cell is safely below any
    real gap.
    """
    def key(x: float, y: float) -> tuple[int, int]:
        return (round(x * 1e4), round(y * 1e4))

    from collections import defaultdict

    ends: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x0, y0, x1, y1) in enumerate(segs):
        ends[key(x0, y0)].append(i)
        ends[key(x1, y1)].append(i)

    used = [False] * len(segs)
    out: list[str] = []

    for start in range(len(segs)):
        if used[start]:
            continue
        used[start] = True
        x0, y0, x1, y1 = segs[start]
        chain = [(x0, y0), (x1, y1)]

        # Extend from the tail, then from the head, until nothing connects.
        for _ in range(2):
            while True:
                cx, cy = chain[-1]
                nxt = None
                for j in ends[key(cx, cy)]:
                    if used[j]:
                        continue
                    a, b, c, d = segs[j]
                    if key(a, b) == key(cx, cy):
                        nxt, pt = j, (c, d)
                    elif key(c, d) == key(cx, cy):
                        nxt, pt = j, (a, b)
                    else:
                        continue
                    break
                if nxt is None:
                    break
                used[nxt] = True
                chain.append(pt)
            chain.reverse()

        # Relative line commands with integer deltas. Absolute coordinates at
        # one decimal place were most of the file: a delta is one or two
        # characters where an absolute is five or six, and at this line weight
        # rounding to a whole user unit (1/1000 of the tile) is invisible.
        px = round(chain[0][0] * scale)
        py = round(chain[0][1] * scale)
        parts = [f"M{px} {py}"]
        deltas = []
        for x, y in chain[1:]:
            qx, qy = round(x * scale), round(y * scale)
            if (qx, qy) != (px, py):
                deltas.append(f"{qx - px} {qy - py}")
                px, py = qx, qy
        if not deltas:
            continue
        parts.append("l" + ",".join(deltas))
        out.append("".join(parts))
    return out


def build() -> str:
    h = field(N, SEED)
    scale = SIZE / N

    index_d: list[str] = []
    minor_d: list[str] = []

    for i in range(1, LEVELS):
        level = i / LEVELS
        segs = contour_segments(h, level)
        if not segs:
            continue
        parts = stitch(segs, scale)
        (index_d if i % INDEX_EVERY == 0 else minor_d).extend(parts)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
        f'viewBox="0 0 {SIZE} {SIZE}">'
        f'<g fill="none" stroke="#ffffff" stroke-linecap="round" '
        f'stroke-linejoin="round">'
        f'<path stroke-width="0.9" stroke-opacity="0.30" d="{"".join(minor_d)}"/>'
        f'<path stroke-width="1.7" stroke-opacity="0.55" d="{"".join(index_d)}"/>'
        f"</g></svg>"
    )


def main() -> None:
    svg = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"  {OUT.name}: {len(svg)} B, {LEVELS - 1} levels, {N}x{N} grid, seed {SEED}")

    # The tile must be seamless, so assert it rather than trusting the maths.
    # Opposite edges of the sampled field have to agree to within one sampling
    # step's worth of change, which is what periodicity guarantees.
    h = field(N, SEED)
    dx = float(np.abs(h[:, 0] - np.roll(h, -1, axis=1)[:, N - 1]).max())
    dy = float(np.abs(h[0, :] - np.roll(h, -1, axis=0)[N - 1, :]).max())
    assert dx < 1e-9 and dy < 1e-9, f"tile is not seamless: dx={dx}, dy={dy}"
    print(f"  seamless: max edge mismatch {max(dx, dy):.2e}")


if __name__ == "__main__":
    main()
