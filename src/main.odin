package main

// ============================================================================
// Radar — the M4 Distribution Transformer tracking a target it has never seen.
//
//   make run            (clones odin-ml and odin-ui-v2 into deps/ on first use)
//
// The scope shows the true track, every rangefinder/bearing reading as a blip
// (echo, clutter, or sensor failure), and the DT's belief: its 4-component
// mixture over position as a density and one ellipse per component, updated
// once per reading and pushed through the dynamics in between. A 5000-particle
// filter runs alongside for comparison; the strip below scores both filters at
// every step (−log density at the true state, lower is better).
//
// Space play/pause, ←/→ step, N new track, F fullscreen, Esc quit.
// Click or drag the timeline to scrub.
// ============================================================================

import "core:fmt"
import "core:math"
import "core:os"
import dt "odinml:dt"
import ml "odinml:ml"
import ui "odinui:ui"
import rl "odinui:deps/raylib"

FONT :: #load("../assets/Inter-Medium.ttf")

View :: struct {
	t:          int, // current step
	t_frac:     f32, // time within the step while playing
	playing:    bool,
	speed:      f32, // steps per second
	density, components, particles, truth, echo_types, prior: bool,
	sweep:      f32, // radar sweep angle (cosmetic)
	meta:       []ml.Meta,
	heat:       Heat,
}

main :: proc() {
	ml.setup_from_env()
	path := dt.FUSION_CHECKPOINT
	shot: string // --shot=file.png: play a while, save a screenshot, quit
	for a in os.args[1:] {
		if len(a) > 7 && a[:7] == "--shot=" do shot = a[7:]
		else do path = a
	}
	ml.seed(0)
	m := dt.dt_full_model(dt.FUSION_MODEL, dt.OBS_FEAT)
	params := dt.dt_full_params(m)
	meta, ok := ml.load(path, params[:])
	if !ok {
		fmt.eprintfln("no trained model at %s — run `make dt-fusion` first", path)
		os.exit(1)
	}

	tr := new(Track)
	seed: u64 = 1
	track_make(tr, &m, seed)

	v := View{speed = 12, playing = shot != "", density = true, components = true, particles = false, truth = true, echo_types = true, meta = meta}

	ui.theme.panel_w = 300
	ui.init({
		title      = "radar — Distribution Transformer filter",
		width      = 1280,
		height     = 860,
		resizable  = true,
		high_dpi   = true,
		msaa_4x    = true,
		target_fps = 60,
		min_width  = 900,
		min_height = 600,
	})
	defer ui.shutdown()
	ui.load_font_data(ui.theme.font_title, ui.theme.size_title, FONT)
	ui.load_font_data(ui.theme.font_body, ui.theme.size_body, FONT)
	ui.load_font_data(ui.theme.font_small, ui.theme.size_small, FONT)
	heat_init(&v.heat)
	defer heat_destroy(&v.heat)

	frames := 0
	for ui.frame() {
		free_all(context.temp_allocator)
		frames += 1
		if shot != "" && frames == 150 {
			ui.screenshot(fmt.ctprint(shot))
			ui.request_quit()
			break
		}
		if ui.key_pressed(.Escape) {
			ui.request_quit()
			break
		}
		if ui.key_pressed_bare(.F) do ui.toggle_fullscreen()
		ui.zoom_shortcuts()
		if ui.key_pressed_bare(.Space) do v.playing = !v.playing
		if ui.key_pressed_bare(.Right) do step_to(&v, v.t + 1)
		if ui.key_pressed_bare(.Left) do step_to(&v, v.t - 1)
		new_track := ui.key_pressed_bare(.N)

		if v.playing {
			v.t_frac += ui.frame_dt * v.speed
			for v.t_frac >= 1 {
				v.t_frac -= 1
				if v.t < T - 1 do v.t += 1
				else do v.playing = false
			}
		}
		v.sweep = math.mod(v.sweep + ui.frame_dt * 2.4, 2 * math.PI)

		ui.begin_layout()
		if ui.root_begin("Root", true, .TopToBottom) {
			if ui.top_bar_begin() {
				ui.body("Radar — Distribution Transformer filter on an unseen track", ui.theme.text_dim)
				ui.element_end()
			}
			if ui.row_begin("Body", {grow_h = true}) {
				new_track |= build_panel(&v, tr)
				if ui.column_begin("Main", {grow_h = true}) {
					ui.canvas_begin("Scope", ui.Color{})
					ui.element_end()
					if ui.row_begin("TimelineBox", {height = 170}) {
						ui.canvas_begin("Timeline", ui.Color{}) // transparent: drawn under the UI
						ui.element_end()
						ui.element_end()
					}
					ui.element_end()
				}
				ui.element_end()
			}
			ui.element_end()
		}
		cmds := ui.end_layout()

		if new_track {
			seed += 1
			track_make(tr, &m, seed)
			v.t, v.t_frac, v.playing = 0, 0, true
			v.heat.valid = false
		}
		timeline_input(&v)

		ui.begin_draw()
		if ui.begin_clip("Scope") {
			draw_scope(&v, tr)
			ui.end_clip()
		}
		if ui.begin_clip("Timeline") {
			draw_timeline(&v, tr)
			ui.end_clip()
		}
		ui.render(&cmds)
		ui.end_draw()
	}
}

step_to :: proc(v: ^View, t: int) {
	v.t = clamp(t, 0, T - 1)
	v.t_frac = 0
	v.playing = false
}

build_panel :: proc(v: ^View, tr: ^Track) -> (new_track: bool) {
	if !ui.panel_begin("Panel") do return
	ui.title(fmt.tprintf("Track #%d", tr.seed))
	ui.dim("fresh random track, never seen in training")
	ui.divider("d0")

	ui.section("PLAYBACK")
	v.playing = ui.toggle("play", v.playing ? "PLAYING" : "PAUSED", v.playing)
	if ui.row_begin("steps", {gap = 8}) {
		if ui.button("back", "← step") do step_to(v, v.t - 1)
		if ui.button("fwd", "step →") do step_to(v, v.t + 1)
		ui.element_end()
	}
	t := ui.slider("t", "Step", f32(v.t), 0, T - 1, "%.0f")
	if int(math.round(t)) != v.t do step_to(v, int(math.round(t)))
	v.speed = ui.slider("speed", "Steps / second", v.speed, 1, 20, "%.0f")
	if ui.button("new", "New track", {accent = true}) do new_track = true

	ui.section("SHOW")
	v.density = ui.toggle("density", "DT density", v.density)
	v.components = ui.toggle("components", "DT components (1 and 2 sd)", v.components)
	v.prior = ui.toggle("prior", "DT prior (after predict)", v.prior)
	v.particles = ui.toggle("particles", "Particle filter", v.particles)
	v.truth = ui.toggle("truth", "True track", v.truth)
	v.echo_types = ui.toggle("echo", "Reveal clutter / failures", v.echo_types)

	ui.section(fmt.tprintf("STEP %d", v.t))
	s := tr.series
	ui.body(fmt.tprintf("range %.2f km   bearing %.1f°", s.z_range[v.t], s.z_bear[v.t] * 180 / math.PI))
	echo := [dt.Echo]string{.True = "echo", .Clutter = "clutter", .Failure = "sensor failure"}
	if v.echo_types do ui.dim(fmt.tprintf("rangefinder saw: %s", echo[s.echo[v.t]]))
	ui.body(fmt.tprintf("−log q(x)   DT %+.2f   PF %+.2f", tr.nll_dt[v.t], tr.nll_pf[v.t]))
	ui.dim(fmt.tprintf("track mean  DT %+.2f   PF %+.2f", mean_nll(tr.nll_dt[:]), mean_nll(tr.nll_pf[:])))
	for c, k in tr.post[v.t] {
		_, cov := position_marginal(c)
		sd := math.sqrt(0.5 * (cov[0, 0] + cov[1, 1]))
		ui.dim(fmt.tprintf("component %d   w %.2f   sd %.2f km", k, c.w, sd))
	}

	ui.section("MODEL")
	for key in ([]string{"train", "train_seconds", "eval_update_kl", "eval_filter_nll"}) {
		if val, found := ml.meta_get(v.meta, key); found do ui.dim(fmt.tprintf("%s: %s", key, val))
	}
	ui.divider("d1")
	ui.dim("space play   ←/→ step   N new track   F fullscreen")
	ui.panel_end("Panel")
	return
}
