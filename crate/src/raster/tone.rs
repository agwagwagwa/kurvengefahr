//! **Tonal hatching** (`hatch`): engraving-style cross-hatch. Tone is quantized into `levels` bands;
//! each band lays one set of parallel hatch lines at its own angle, but only *where the image is at
//! least that dark*. Darker regions clear more thresholds, so they accrue more overlapping passes —
//! light greys get a single sparse rake, blacks get a dense cross-hatch. Pure line work, no fills.
//!
//! **Pressure hatch** (`pressurehatch`): the same even rake of parallel lines, but tone rides on pen
//! *pressure* rather than line *density* — each point carries the local darkness as its pressure, so
//! light greys draw feather-light and blacks bear down hard. The line lifts over near-white paper so
//! whites stay white. On a pressure-pen plotter this renders the full tonal range from one uniform,
//! constant-spacing hatch; on a plain pen-up/down machine (no `downLight`) it degrades to a flat
//! single hatch. The per-point pressure survives place/clip/optimize and drives a Z ramp in `emit`.
//!
//! **Aligned hatching** (`alignedhatch`): the same tone bands as `hatch`, but every band shares one
//! angle and is instead *offset* across the rake by `spacing·(k-1)/levels`. Darker regions interleave
//! more lines between the base ones, so tone reads as line density with no crossings; full black is
//! an even rake `spacing/levels` apart.
//!
//! **Radial aligned hatching** (`radialhatch`): aligned hatching around a centre given in box UV
//! (0..1 across the image; values outside put the centre off the image). `rings` lays concentric
//! circles (bands offset in radius); `rays` lays spokes through the centre (bands offset in angle,
//! `spacing` apart at the farthest corner, so they converge inward). An off-image centre only sweeps
//! the arc that actually covers the box.

use super::{stroke, Grid, Params};
use crate::geom::{Point, Stroke};

/// March step along each hatch line (mm). Fine enough to catch tone boundaries cleanly without
/// generating excess vertices.
const STEP: f32 = crate::tess::RASTER_HATCH_STEP;

pub fn hatch(grid: &Grid, p: &Params) -> Vec<Stroke> {
    let levels = p.levels.clamp(1, MAX_LEVELS) as usize;
    let spacing = p.spacing.max(0.2);
    let mut out = Vec::new();
    // Band k (1..=levels) inks where inkness ≥ k/(levels+1); each gets its own angle so successive
    // bands cross rather than retread. The angles are spread evenly across a half-turn (a hatch line
    // and its 180° flip are the same), so any band count cross-hatches cleanly — at 4 bands this is
    // the classic 0/45/90/135, but it generalizes to whatever `levels` is, no snapping to 45°.
    for k in 1..=levels {
        let cutoff = k as f32 / (levels as f32 + 1.0);
        let angle = p.angle + (k - 1) as f32 * 180.0 / levels as f32;
        let start = out.len();
        hatch_pass(grid, angle, spacing, 0.0, |ink| ink >= cutoff, &mut out);
        // Tag the band so exports can keep each tone band's rake as its own layer.
        for s in &mut out[start..] {
            s.part = k as u16;
        }
    }
    out
}

/// Upper bound on tone bands — generous, just a guard against a pathological `levels`. Each band is
/// one full hatch pass, so this also bounds the work.
const MAX_LEVELS: u32 = 16;

/// Rake parallel lines (direction `angle_deg`, `spacing` apart) across the box, emitting a stroke for
/// each maximal run where `keep(inkness)` holds. Shared by tone bands. `offset` (mm) shifts the whole
/// rake along its normal — aligned hatching uses it to interleave bands; the line grid is anchored to
/// the box (not to the inked region), so equal angles + offsets always land on the same lines.
pub fn hatch_pass(
    grid: &Grid,
    angle_deg: f32,
    spacing: f32,
    offset: f32,
    keep: impl Fn(f32) -> bool,
    out: &mut Vec<Stroke>,
) {
    let a = angle_deg.to_radians();
    let (dx, dy) = (a.cos(), a.sin()); // line direction
    let (nx, ny) = (-dy, dx); // line normal (offset axis)
    let (tw, th) = (grid.tw, grid.th);
    // Project the box corners onto the direction/normal axes to get the sweep ranges.
    let corners = [(0.0, 0.0), (tw, 0.0), (0.0, th), (tw, th)];
    let proj = |px: f32, py: f32, ax: f32, ay: f32| px * ax + py * ay;
    let (mut tmin, mut tmax, mut omin, mut omax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &(cx, cy) in &corners {
        let t = proj(cx, cy, dx, dy);
        let o = proj(cx, cy, nx, ny);
        tmin = tmin.min(t);
        tmax = tmax.max(t);
        omin = omin.min(o);
        omax = omax.max(o);
    }
    let steps = ((tmax - tmin) / STEP).floor() as usize;
    let mut o = omin + spacing * 0.5 + offset.rem_euclid(spacing);
    while o <= omax {
        let line = (0..=steps).map(|i| {
            let t = tmin + i as f32 * STEP;
            (o * nx + t * dx, o * ny + t * dy)
        });
        runs(grid, line, &keep, out);
        o += spacing;
    }
}

/// Walk a sampled curve and emit a stroke for each maximal run of in-box samples where
/// `keep(inkness)` holds. The one place hatch passes (straight, ring, ray) decide pen-down.
fn runs(
    grid: &Grid,
    samples: impl Iterator<Item = (f32, f32)>,
    keep: &impl Fn(f32) -> bool,
    out: &mut Vec<Stroke>,
) {
    let eps = 1e-3;
    let (tw, th) = (grid.tw, grid.th);
    let mut seg: Vec<Point> = Vec::new();
    for (x, y) in samples {
        let inside = x >= -eps && y >= -eps && x <= tw + eps && y <= th + eps;
        if inside && keep(grid.ink_mm(x, y)) {
            seg.push(Point {
                x,
                y,
                pressure: 1.0,
            });
        } else if seg.len() >= 2 {
            out.push(stroke(std::mem::take(&mut seg)));
        } else {
            seg.clear();
        }
    }
    if seg.len() >= 2 {
        out.push(stroke(seg));
    }
}

/// Aligned tonal hatching: `hatch`'s tone bands, all at `p.angle`, band k shifted by
/// `spacing·(k-1)/levels` across the rake (see the module header).
pub fn aligned_hatch(grid: &Grid, p: &Params) -> Vec<Stroke> {
    let levels = p.levels.clamp(1, MAX_LEVELS) as usize;
    let spacing = p.spacing.max(0.2);
    let mut out = Vec::new();
    for k in 1..=levels {
        let cutoff = k as f32 / (levels as f32 + 1.0);
        let offset = spacing * (k - 1) as f32 / levels as f32;
        let start = out.len();
        hatch_pass(
            grid,
            p.angle,
            spacing,
            offset,
            |ink| ink >= cutoff,
            &mut out,
        );
        for s in &mut out[start..] {
            s.part = k as u16;
        }
    }
    out
}

/// Radial aligned hatching: aligned bands as concentric rings or converging rays around the UV
/// centre (see the module header).
pub fn radial_hatch(grid: &Grid, p: &Params) -> Vec<Stroke> {
    use std::f32::consts::TAU;
    let levels = p.levels.clamp(1, MAX_LEVELS) as usize;
    let spacing = p.spacing.max(0.2);
    let (tw, th) = (grid.tw, grid.th);
    let (cx, cy) = (p.center_u * tw, p.center_v * th);
    let inside = (0.0..=tw).contains(&cx) && (0.0..=th).contains(&cy);
    let corners = [(0.0, 0.0), (tw, 0.0), (0.0, th), (tw, th)];

    // Radial extent of the box from the centre: nearest box point .. farthest corner.
    let rmin = {
        let dx = (0.0f32 - cx).max(cx - tw).max(0.0);
        let dy = (0.0f32 - cy).max(cy - th).max(0.0);
        dx.hypot(dy)
    };
    let rmax = corners
        .iter()
        .map(|&(x, y)| (x - cx).hypot(y - cy))
        .fold(0.0f32, f32::max);
    if rmax <= 0.0 {
        return vec![];
    }
    // Angular window covering the box. A centre inside (or on) the box needs the full turn; outside,
    // the box subtends < π, spanned by its corners — unwrap their angles around the direction to the
    // box centre so the window never straddles the ±π seam.
    let (a0, a1) = if inside {
        (0.0, TAU)
    } else {
        let mid = (th * 0.5 - cy).atan2(tw * 0.5 - cx);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &(x, y) in &corners {
            let mut d = (y - cy).atan2(x - cx) - mid;
            d = (d + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI;
            lo = lo.min(d);
            hi = hi.max(d);
        }
        (mid + lo, mid + hi)
    };
    let full = inside;
    let at = |r: f32, a: f32| (cx + r * a.cos(), cy + r * a.sin());

    let mut out = Vec::new();
    for k in 1..=levels {
        let cutoff = k as f32 / (levels as f32 + 1.0);
        let keep = |ink: f32| ink >= cutoff;
        let frac = (k - 1) as f32 / levels as f32;
        let start = out.len();
        if p.radial_mode == "rays" {
            // Rays `spacing` apart at the farthest corner; on a full turn round the count so the
            // pattern tiles seamlessly.
            let mut da = spacing / rmax;
            if full {
                da = TAU / (TAU / da).round().max(1.0);
            }
            let steps = ((rmax - rmin) / STEP).ceil().max(1.0) as usize;
            let mut a = a0 + da * (0.5 + frac);
            while a <= a1 {
                let ray =
                    (0..=steps).map(|i| at(rmin + (rmax - rmin) * i as f32 / steps as f32, a));
                runs(grid, ray, &keep, &mut out);
                a += da;
            }
        } else {
            // Rings: radii on the `spacing` grid anchored at the centre (so bands interleave), from
            // the first one reaching the box out to the farthest corner.
            let base = spacing * (0.5 + frac);
            let mut r = base + ((rmin - base) / spacing).ceil().max(0.0) * spacing;
            while r <= rmax {
                let steps = ((a1 - a0) * r / STEP).ceil().max(8.0) as usize;
                let arc = (0..=steps).map(|i| at(r, a0 + (a1 - a0) * i as f32 / steps as f32));
                let first = out.len();
                runs(grid, arc, &keep, &mut out);
                // A full ring inked across its 0/2π seam comes out as two runs meeting there — join
                // them so the pen draws it in one go.
                if full && out.len() - first >= 2 {
                    let (s0, s1) = (
                        out[first].points[0],
                        *out.last().unwrap().points.last().unwrap(),
                    );
                    if (s0.x - s1.x).hypot(s0.y - s1.y) < 1e-3 {
                        let head = out.remove(first);
                        out.last_mut()
                            .unwrap()
                            .points
                            .extend_from_slice(&head.points[1..]);
                    }
                }
                r += spacing;
            }
        }
        for s in &mut out[start..] {
            s.part = k as u16;
        }
    }
    out
}

/// Inkness floor below which the pen lifts (paper): only near-white breaks a line, so light greys
/// still draw (feather-light) rather than punching gaps.
const INK_FLOOR: f32 = 0.06;

/// Contrast curve for the pressure map, pivoted on mid grey: `c == 1` is linear (pressure = inkness),
/// `c > 1` expands the tonal range (more pressure contrast — for flat source images), `c < 1`
/// compresses it toward mid. Clamped to a usable pressure.
fn contrast(v: f32, c: f32) -> f32 {
    (0.5 + (v - 0.5) * c).clamp(0.0, 1.0)
}

/// Pressure-modulated single hatch: one even rake of parallel lines (`angle_deg`, `spacing` apart),
/// each marched point tagged with `pressure = contrast(inkness)` so darkness reads as pen pressure,
/// not line density. Lines break wherever inkness drops below [`INK_FLOOR`] (paper). Same sweep
/// geometry as [`hatch_pass`]; only the per-point tagging and the break rule differ.
pub fn pressure_hatch(grid: &Grid, p: &Params) -> Vec<Stroke> {
    let spacing = p.spacing.max(0.2);
    let c = p.pressure_contrast.max(0.0);
    let a = p.angle.to_radians();
    let (dx, dy) = (a.cos(), a.sin()); // line direction
    let (nx, ny) = (-dy, dx); // line normal (offset axis)
    let (tw, th) = (grid.tw, grid.th);
    let corners = [(0.0, 0.0), (tw, 0.0), (0.0, th), (tw, th)];
    let proj = |px: f32, py: f32, ax: f32, ay: f32| px * ax + py * ay;
    let (mut tmin, mut tmax, mut omin, mut omax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &(cx, cy) in &corners {
        let t = proj(cx, cy, dx, dy);
        let o = proj(cx, cy, nx, ny);
        tmin = tmin.min(t);
        tmax = tmax.max(t);
        omin = omin.min(o);
        omax = omax.max(o);
    }
    let eps = 1e-3;
    let mut out = Vec::new();
    let mut o = omin + spacing * 0.5;
    while o <= omax {
        let mut seg: Vec<Point> = Vec::new();
        let mut t = tmin;
        while t <= tmax {
            let x = o * nx + t * dx;
            let y = o * ny + t * dy;
            let inside = x >= -eps && y >= -eps && x <= tw + eps && y <= th + eps;
            let ink = if inside { grid.ink_mm(x, y) } else { 0.0 };
            if ink > INK_FLOOR {
                seg.push(Point {
                    x,
                    y,
                    pressure: contrast(ink, c),
                });
            } else if seg.len() >= 2 {
                out.push(stroke(std::mem::take(&mut seg)));
            } else {
                seg.clear();
            }
            t += STEP;
        }
        if seg.len() >= 2 {
            out.push(stroke(seg));
        }
        o += spacing;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hatch_tags_each_tone_band_as_its_own_part() {
        // A uniformly black image clears every threshold, so every band rakes the whole box.
        let rgba = [0u8, 0, 0, 255].repeat(4 * 4);
        let grid = Grid::build(&rgba, 4, 4, 20.0, 20.0, false);
        let p = Params {
            levels: 3,
            spacing: 2.0,
            angle: 0.0,
            ..Params::default()
        };
        let out = hatch(&grid, &p);
        let mut parts: Vec<u16> = out.iter().map(|s| s.part).collect();
        parts.dedup();
        assert_eq!(parts, vec![1, 2, 3]);
        assert!(out.iter().all(|s| s.part >= 1 && s.part <= 3));
    }

    fn black(n: usize, mm: f32) -> Grid {
        let rgba = [0u8, 0, 0, 255].repeat(n * n);
        Grid::build(&rgba, n, n, mm, mm, false)
    }

    #[test]
    fn aligned_hatch_interleaves_bands_at_one_angle() {
        // Horizontal lines over full black: every band rakes the box, band k shifted by
        // spacing·(k-1)/levels, so the union is an even rake spacing/levels apart with no repeats.
        let grid = black(4, 20.0);
        let p = Params {
            levels: 4,
            spacing: 2.0,
            angle: 0.0,
            ..Params::default()
        };
        let out = aligned_hatch(&grid, &p);
        let mut ys: Vec<f32> = out
            .iter()
            .map(|s| {
                assert!(s.points.iter().all(|q| (q.y - s.points[0].y).abs() < 1e-3));
                s.points[0].y
            })
            .collect();
        ys.sort_by(f32::total_cmp);
        assert!(ys.len() >= 38, "{} lines", ys.len());
        for w in ys.windows(2) {
            assert!((w[1] - w[0] - 0.5).abs() < 1e-3, "gap {}", w[1] - w[0]);
        }
    }

    #[test]
    fn radial_rings_are_concentric_and_rays_converge() {
        let grid = black(4, 20.0);
        let mut p = Params {
            levels: 2,
            spacing: 2.0,
            center_u: 0.5,
            center_v: 0.5,
            ..Params::default()
        };
        // Rings: every point of a stroke sits at one radius from the centre (10,10).
        let rings = radial_hatch(&grid, &p);
        assert!(!rings.is_empty());
        for s in &rings {
            let r0 = (s.points[0].x - 10.0).hypot(s.points[0].y - 10.0);
            assert!(s
                .points
                .iter()
                .all(|q| ((q.x - 10.0).hypot(q.y - 10.0) - r0).abs() < 1e-2));
        }
        // An inner ring fully inside the box is one closed stroke (seam joined).
        let inner: Vec<_> = rings
            .iter()
            .filter(|s| (s.points[0].x - 10.0).hypot(s.points[0].y - 10.0) < 5.0)
            .collect();
        assert!(!inner.is_empty());
        for s in inner {
            let (a, b) = (s.points[0], *s.points.last().unwrap());
            assert!((a.x - b.x).hypot(a.y - b.y) < 1e-2);
        }
        // Rays: every stroke is collinear with the centre.
        p.radial_mode = "rays".into();
        let rays = radial_hatch(&grid, &p);
        assert!(!rays.is_empty());
        for s in &rays {
            let (a, b) = (s.points[0], *s.points.last().unwrap());
            let cross = (a.x - 10.0) * (b.y - 10.0) - (a.y - 10.0) * (b.x - 10.0);
            assert!(cross.abs() < 1e-2, "ray misses centre");
        }
    }

    #[test]
    fn radial_centre_off_image_stays_in_box() {
        let grid = black(4, 20.0);
        for mode in ["rings", "rays"] {
            for (u, v) in [(-1.0, 0.5), (3.0, -2.0), (0.5, 1.5), (-20.0, -20.0)] {
                let p = Params {
                    spacing: 1.0,
                    center_u: u,
                    center_v: v,
                    radial_mode: mode.into(),
                    ..Params::default()
                };
                let out = radial_hatch(&grid, &p);
                assert!(!out.is_empty(), "{mode} at ({u},{v}) produced nothing");
                for s in &out {
                    for q in &s.points {
                        assert!(q.x > -0.01 && q.y > -0.01 && q.x < 20.01 && q.y < 20.01);
                    }
                }
            }
        }
    }
}
