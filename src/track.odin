package main

// One unseen track, filtered two ways and kept whole so the view can scrub:
// the DT (update) + exact predict, and a bootstrap particle filter.

import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import dt "odinml:dt"
import ml "odinml:ml"

T :: 100 // steps per track, as in the evaluation
K :: dt.K_FUSION
N_PARTICLES :: 5000
N_SHOWN :: 600 // particles drawn per step

Vec :: dt.Vec
Mat :: dt.Mat

Track :: struct {
	seed:           u64,
	series:         dt.Series,
	prior, post:    [T][K]dt.Component, // the DT's belief before / after each update
	pf_mean:        [T]Vec,
	pf_cov:         [T]Mat,
	pf_pts:         [T][N_SHOWN]Vec,
	nll_dt, nll_pf: [T]f64,
	view_r:         f32, // scope radius (km) that fits the track
}

track_make :: proc(tr: ^Track, m: ^dt.Dt_Full, seed: u64) {
	delete(tr.series.x)
	delete(tr.series.z_range)
	delete(tr.series.z_bear)
	delete(tr.series.echo)
	tr.seed = seed
	rand.reset(seed)
	tr.series = dt.sample_series(T)

	// DT filter
	arena: mem.Dynamic_Arena
	ml.arena_init(&arena, 4 * mem.Megabyte)
	defer mem.dynamic_arena_destroy(&arena)
	cur: [K]dt.Component
	dt.initial_prior(cur[:])
	for t in 0 ..< T {
		tr.prior[t] = cur
		{
			context.allocator = mem.dynamic_arena_allocator(&arena)
			zr, zb := []f64{tr.series.z_range[t]}, []f64{tr.series.z_bear[t]}
			dt.dt_update(m, cur[:], zr, zb, cur[:])
		}
		mem.dynamic_arena_reset(&arena)
		tr.post[t] = cur
		tr.nll_dt[t] = -dt.mixture_log_pdf(cur[:], tr.series.x[t])
		dt.predict_mixture(cur[:])
	}

	// particle filter
	pf := dt.pf_make(N_PARTICLES, seed * 7 + 1)
	defer dt.pf_destroy(pf)
	for t in 0 ..< T {
		if t > 0 do dt.pf_predict(pf)
		tr.pf_mean[t], tr.pf_cov[t] = dt.pf_update(pf, tr.series.z_range[t], tr.series.z_bear[t])
		for i in 0 ..< N_SHOWN do tr.pf_pts[t][i] = pf.ps[i * (N_PARTICLES / N_SHOWN)]
		c, ok := dt.component_from_cov(1, tr.pf_mean[t], tr.pf_cov[t] + 1e-3 * linalg.identity(Mat))
		tr.nll_pf[t] = ok ? -dt.component_log_pdf(c, tr.series.x[t]) : math.inf_f64(1)
	}

	r: f64 = 3
	for x in tr.series.x do r = max(r, math.sqrt(x[0] * x[0] + x[2] * x[2]))
	tr.view_r = f32(math.ceil(r + 1))
}

// Position marginal (px, py) of a 4D component: mean and 2×2 covariance.
position_marginal :: proc(c: dt.Component) -> (mu: [2]f64, cov: matrix[2, 2]f64) {
	S := dt.covariance(c)
	return {c.mu[0], c.mu[2]}, {S[0, 0], S[0, 2], S[2, 0], S[2, 2]}
}

mean_velocity :: proc(mix: []dt.Component) -> (v: [2]f64) {
	for c in mix do v += c.w * [2]f64{c.mu[1], c.mu[3]}
	return
}

mean_nll :: proc(v: []f64) -> f64 {
	return math.sum(v) / f64(len(v))
}
