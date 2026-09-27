package main

// Drawing: the radar scope (world = km, north up, radar at the center) and the
// per-step NLL timeline. Raylib calls between ui.begin_draw and ui.render,
// clipped to the canvas regions; coordinates are layout units.

import "core:fmt"
import "core:math"
import dt "odinml:dt"
import ui "odinui:ui"
import rl "odinui:deps/raylib"

PHOSPHOR :: rl.Color{70, 210, 130, 255}
DT_COL :: rl.Color{90, 190, 255, 255}
PF_COL :: rl.Color{255, 170, 70, 255}
TRUTH_COL :: rl.Color{245, 245, 250, 255}
ECHO_COLS := [dt.Echo]rl.Color{.True = {110, 230, 150, 255}, .Clutter = {255, 200, 80, 255}, .Failure = {255, 100, 110, 255}}

with_alpha :: proc(c: rl.Color, a: f32) -> rl.Color {
	return {c.r, c.g, c.b, u8(clamp(a, 0, 1) * f32(c.a))}
}

// ---- world ↔ screen ----------------------------------------------------------

Scope :: struct {
	cx, cy, scale: f32, // screen px per km
	r:             f32, // view radius, km
}

scope_of :: proc(box: ui.Region_Box, view_r: f32) -> Scope {
	return {box.x + box.width * 0.5, box.y + box.height * 0.5, 0.46 * min(box.width, box.height) / view_r, view_r}
}

to_screen :: proc(s: Scope, x, y: f64) -> rl.Vector2 {
	return {s.cx + f32(x) * s.scale, s.cy - f32(y) * s.scale}
}

label :: proc(text: string, p: rl.Vector2, col: rl.Color) {
	font := ui.glyph_font(ui.theme.font_small)
	if font.glyphCount == 0 do return
	size := ui.text_draw_size(ui.theme.font_small, ui.theme.size_small)
	rl.DrawTextEx(font, fmt.ctprint(text), {ui.snap_px(p.x), ui.snap_px(p.y)}, size, 0, col)
}

// k-sigma ellipse of a 2D Gaussian as a closed polyline.
ellipse :: proc(s: Scope, mu: [2]f64, cov: matrix[2, 2]f64, k: f64, thick: f32, col: rl.Color) {
	a, b, c := cov[0, 0], cov[0, 1], cov[1, 1]
	h := math.sqrt(0.25 * (a - c) * (a - c) + b * b)
	l1, l2 := max(0.5 * (a + c) + h, 0), max(0.5 * (a + c) - h, 0)
	th := 0.5 * math.atan2(2 * b, a - c)
	e1 := [2]f64{math.cos(th), math.sin(th)} * (k * math.sqrt(l1))
	e2 := [2]f64{-math.sin(th), math.cos(th)} * (k * math.sqrt(l2))
	N :: 72
	prev := to_screen(s, mu.x + e1.x, mu.y + e1.y)
	for i in 1 ..= N {
		phi := 2 * math.PI * f64(i) / N
		p := mu + math.cos(phi) * e1 + math.sin(phi) * e2
		q := to_screen(s, p.x, p.y)
		rl.DrawLineEx(prev, q, thick, col)
		prev = q
	}
}

arrow :: proc(from, to: rl.Vector2, thick: f32, col: rl.Color) {
	rl.DrawLineEx(from, to, thick, col)
	d := to - from
	n := math.sqrt(d.x * d.x + d.y * d.y)
	if n < 1 do return
	u := d / n
	w := rl.Vector2{-u.y, u.x}
	rl.DrawTriangle(to, to - 9 * u - 5 * w, to - 9 * u + 5 * w, col)
	rl.DrawTriangle(to, to - 9 * u + 5 * w, to - 9 * u - 5 * w, col)
}

// ---- density texture -----------------------------------------------------------

HEAT_N :: 220

Heat :: struct {
	tex:       rl.Texture2D,
	pixels:    []rl.Color,
	valid:     bool,
	t:         int,
	seed:      u64,
}

heat_init :: proc(h: ^Heat) {
	img := rl.GenImageColor(HEAT_N, HEAT_N, {0, 0, 0, 0})
	h.tex = rl.LoadTextureFromImage(img)
	rl.UnloadImage(img)
	rl.SetTextureFilter(h.tex, .BILINEAR)
	h.pixels = make([]rl.Color, HEAT_N * HEAT_N)
}

heat_destroy :: proc(h: ^Heat) {
	rl.UnloadTexture(h.tex)
	delete(h.pixels)
}

// Position density of the DT posterior at step t, rendered over [-r, r]².
heat_update :: proc(h: ^Heat, tr: ^Track, t: int) {
	if h.valid && h.t == t && h.seed == tr.seed do return
	h.valid, h.t, h.seed = true, t, tr.seed
	Comp2 :: struct {
		w:       f64,
		mu:      [2]f64,
		inv:     matrix[2, 2]f64,
		norm:    f64,
	}
	comps: [K]Comp2
	for c, k in tr.post[t] {
		mu, cov := position_marginal(c)
		det := cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
		inv := (1 / det) * matrix[2, 2]f64{cov[1, 1], -cov[0, 1], -cov[1, 0], cov[0, 0]}
		comps[k] = {c.w, mu, inv, c.w / (2 * math.PI * math.sqrt(max(det, 1e-300)))}
	}
	r := f64(tr.view_r)
	dens := make([]f64, HEAT_N * HEAT_N, context.temp_allocator)
	peak: f64 = 1e-300
	for j in 0 ..< HEAT_N {
		y := r - (f64(j) + 0.5) / HEAT_N * 2 * r
		for i in 0 ..< HEAT_N {
			x := -r + (f64(i) + 0.5) / HEAT_N * 2 * r
			d: f64 = 0
			for c in comps {
				dx := [2]f64{x - c.mu.x, y - c.mu.y}
				q := dx.x * (c.inv[0, 0] * dx.x + c.inv[0, 1] * dx.y) + dx.y * (c.inv[1, 0] * dx.x + c.inv[1, 1] * dx.y)
				d += c.norm * math.exp(-0.5 * q)
			}
			dens[j * HEAT_N + i] = d
			peak = max(peak, d)
		}
	}
	for d, i in dens {
		a := math.sqrt(d / peak) // sqrt: keeps the tails visible
		h.pixels[i] = {DT_COL.r, DT_COL.g, DT_COL.b, u8(200 * a)}
	}
	rl.UpdateTexture(h.tex, raw_data(h.pixels))
}

// ---- scope -----------------------------------------------------------------------

draw_scope :: proc(v: ^View, tr: ^Track) {
	box, ok := ui.region("Scope")
	if !ok do return
	s := scope_of(box, tr.view_r)
	center := rl.Vector2{s.cx, s.cy}
	R := s.r * s.scale

	// scope face: rings every km (labelled every 2), bearing lines every 30°
	rl.DrawCircleV(center, R, {6, 16, 12, 255})
	step: f32 = s.r > 8 ? 2 : 1
	for rr := step; rr <= s.r; rr += step {
		rl.DrawCircleLinesV(center, rr * s.scale, with_alpha(PHOSPHOR, 0.22))
		if int(rr) % 2 == 0 do label(fmt.tprintf("%.0f km", rr), {s.cx + 4, s.cy - rr * s.scale + 2}, with_alpha(PHOSPHOR, 0.5))
	}
	for deg := 0; deg < 360; deg += 30 {
		a := f32(deg) * math.PI / 180
		rl.DrawLineEx(center, center + R * rl.Vector2{math.sin(a), -math.cos(a)}, 1, with_alpha(PHOSPHOR, 0.12))
	}
	label("N", {s.cx - 4, s.cy - R - 18}, with_alpha(PHOSPHOR, 0.8))
	// sweep with a fading wake (cosmetic)
	for i in 0 ..< 24 {
		a0 := v.sweep - f32(i + 1) * 0.035
		rl.DrawCircleSector(center, R, (a0 - math.PI / 2) * 180 / math.PI, (a0 + 0.035 - math.PI / 2) * 180 / math.PI, 2,
			with_alpha(PHOSPHOR, 0.10 * (1 - f32(i) / 24)))
	}
	rl.DrawLineEx(center, center + R * rl.Vector2{math.cos(v.sweep - math.PI / 2), math.sin(v.sweep - math.PI / 2)}, 1.5, with_alpha(PHOSPHOR, 0.5))

	t := v.t
	post := tr.post[t][:]

	if v.density {
		heat_update(&v.heat, tr, t)
		tl := to_screen(s, -f64(s.r), f64(s.r))
		rl.DrawTexturePro(v.heat.tex, {0, 0, HEAT_N, HEAT_N}, {tl.x, tl.y, 2 * R, 2 * R}, {0, 0}, 0, rl.WHITE)
	}

	if v.particles {
		for p in tr.pf_pts[t] do rl.DrawCircleV(to_screen(s, p[0], p[2]), 1.3, with_alpha(PF_COL, 0.35))
		mu := [2]f64{tr.pf_mean[t][0], tr.pf_mean[t][2]}
		C := tr.pf_cov[t]
		ellipse(s, mu, {C[0, 0], C[0, 2], C[2, 0], C[2, 2]}, 2, 1.5, with_alpha(PF_COL, 0.9))
	}

	if v.prior {
		for c in tr.prior[t] {
			mu, cov := position_marginal(c)
			ellipse(s, mu, cov, 2, 1, with_alpha(ui.to_rl_color(ui.theme.text_dim), 0.25 + 0.6 * f32(c.w)))
		}
	}

	if v.components {
		for c, k in post {
			mu, cov := position_marginal(c)
			a := 0.2 + 0.8 * f32(c.w)
			ellipse(s, mu, cov, 1, 2, with_alpha(DT_COL, a))
			ellipse(s, mu, cov, 2, 1, with_alpha(DT_COL, 0.6 * a))
			if c.w > 0.05 {
				p := to_screen(s, mu.x, mu.y)
				rl.DrawCircleV(p, 2.5, with_alpha(DT_COL, a))
				label(fmt.tprintf("%d: %.0f%%", k, 100 * c.w), p + {6, -16}, with_alpha(DT_COL, a))
			}
		}
		// belief velocity: where the mean is heading (3 steps of the mean velocity)
		m: [2]f64
		for c in post do m += c.w * [2]f64{c.mu[0], c.mu[2]}
		vel := mean_velocity(post)
		arrow(to_screen(s, m.x, m.y), to_screen(s, m.x + 3 * vel.x, m.y + 3 * vel.y), 2, with_alpha(DT_COL, 0.9))
	}

	// readings of the last few steps: blip at (range, bearing); the newest also as a ray
	ser := tr.series
	HISTORY :: 10
	for age := HISTORY - 1; age >= 0; age -= 1 {
		i := t - age
		if i < 0 do continue
		b, r := ser.z_bear[i], min(ser.z_range[i], f64(s.r)) // failures read up to 20 km: pin to the rim
		p := to_screen(s, r * math.sin(b), r * math.cos(b))
		col := v.echo_types ? ECHO_COLS[ser.echo[i]] : PHOSPHOR
		fade := 1 - f32(age) / HISTORY
		if age == 0 {
			rl.DrawLineEx(center, center + (R + 10) * rl.Vector2{f32(math.sin(b)), -f32(math.cos(b))}, 1, with_alpha(col, 0.35))
			rl.DrawCircleLinesV(p, 9 + 5 * (1 - v.t_frac), with_alpha(col, 0.8))
		}
		rl.DrawCircleV(p, 3 + 2 * fade, with_alpha(col, 0.25 + 0.75 * fade))
	}

	if v.truth {
		for i in 1 ..= t {
			a, b := ser.x[i - 1], ser.x[i]
			rl.DrawLineEx(to_screen(s, a[0], a[2]), to_screen(s, b[0], b[2]), 1.5, with_alpha(TRUTH_COL, 0.15 + 0.5 * f32(i) / f32(t + 1)))
		}
		x := ser.x[t]
		p := to_screen(s, x[0], x[2])
		rl.DrawCircleV(p, 4, TRUTH_COL)
		rl.DrawCircleLinesV(p, 7, with_alpha(TRUTH_COL, 0.6))
	}

	// legend
	lx, ly := box.x + 14, box.y + 12
	label(fmt.tprintf("step %d / %d", t, T - 1), {lx, ly}, ui.to_rl_color(ui.theme.text))
	items := []struct {
		name: string,
		col:  rl.Color,
	}{{"truth", TRUTH_COL}, {"DT belief", DT_COL}, {"particle filter", PF_COL}, {"echo", ECHO_COLS[.True]}, {"clutter", ECHO_COLS[.Clutter]}, {"failure", ECHO_COLS[.Failure]}}
	for it, i in items {
		y := ly + 24 + f32(i) * 18
		rl.DrawCircleV({lx + 5, y + 8}, 4, it.col)
		label(it.name, {lx + 16, y}, ui.to_rl_color(ui.theme.text_dim))
	}
}

// ---- timeline: −log q(x_t) per step, DT vs particle filter ------------------------

NLL_LO :: -4.0
NLL_HI :: 6.0

Timeline :: struct {
	x0, x1, y0, y1: f32,
}

timeline_of :: proc(box: ui.Region_Box) -> Timeline {
	return {box.x + 56, box.x + box.width - 18, box.y + 28, box.y + box.height - 22}
}

tl_x :: proc(tl: Timeline, t: f32) -> f32 {return tl.x0 + t / (T - 1) * (tl.x1 - tl.x0)}
tl_y :: proc(tl: Timeline, v: f64) -> f32 {
	f := (clamp(v, NLL_LO, NLL_HI) - NLL_LO) / (NLL_HI - NLL_LO)
	return tl.y1 - f32(f) * (tl.y1 - tl.y0)
}

timeline_input :: proc(v: ^View) {
	box, ok := ui.region("Timeline")
	if !ok || !ui.mouse_down() || !ui.mouse_in_box(box) do return
	tl := timeline_of(box)
	f := (ui.input.mouse_x - tl.x0) / (tl.x1 - tl.x0)
	v.t = clamp(int(math.round(f * (T - 1))), 0, T - 1)
	v.t_frac, v.playing = 0, false
}

draw_timeline :: proc(v: ^View, tr: ^Track) {
	box, ok := ui.region("Timeline")
	if !ok do return
	tl := timeline_of(box)
	dim := ui.to_rl_color(ui.theme.text_dim)
	grid := ui.to_rl_color(ui.theme.border)
	rl.DrawRectangleRec({box.x, box.y, box.width, box.height}, ui.to_rl_color(ui.theme.panel))
	rl.DrawLineEx({box.x, box.y}, {box.x + box.width, box.y}, 1, grid)

	label("−log q(x_t) per step   (lower is better)", {box.x + 14, box.y + 6}, dim)
	for g in ([]f64{-4, -2, 0, 2, 4, 6}) {
		y := tl_y(tl, g)
		rl.DrawLineEx({tl.x0, y}, {tl.x1, y}, 1, with_alpha(grid, g == 0 ? 1 : 0.5))
		label(fmt.tprintf("%+.0f", g), {box.x + 18, y - 8}, dim)
	}
	// rangefinder junk: ticks under the axis
	if v.echo_types {
		for e, i in tr.series.echo {
			if e == .True do continue
			x := tl_x(tl, f32(i))
			rl.DrawLineEx({x, tl.y1 + 4}, {x, tl.y1 + 12}, 2, ECHO_COLS[e])
		}
	}
	series := []struct {
		vals: []f64,
		col:  rl.Color,
		name: string,
	}{{tr.nll_pf[:], PF_COL, "particle filter"}, {tr.nll_dt[:], DT_COL, "DT"}}
	for sr, k in series {
		for i in 1 ..< T {
			rl.DrawLineEx({tl_x(tl, f32(i - 1)), tl_y(tl, sr.vals[i - 1])}, {tl_x(tl, f32(i)), tl_y(tl, sr.vals[i])}, 2, sr.col)
		}
		label(fmt.tprintf("%s  mean %+.2f", sr.name, mean_nll(sr.vals)), {tl.x1 - 330 + f32(k) * 170, box.y + 6}, sr.col)
	}
	x := tl_x(tl, f32(v.t) + v.t_frac)
	rl.DrawLineEx({x, tl.y0 - 6}, {x, tl.y1 + 4}, 1.5, ui.to_rl_color(ui.theme.text))
}
