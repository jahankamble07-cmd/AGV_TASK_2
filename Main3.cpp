/*
 * Task 3: Noisy LiDAR-Based Dynamic Obstacle Avoidance
 * Noise Handling Method: Spatial Median Filter
 *
 * Prerequisites (edit simulation.hpp):
 *   1. In simulationinstance constructor: change genobs(10, ...) -> genobs(5, ...)
 *   2. In update(): change raycastagent(players[i]) -> raycastagent(players[i], 0.05f)
 *
 * How Median Filtering Works Here:
 *   Each ray distance is replaced by the median of itself and its two
 *   neighbours (window=3). Median is ideal for "salt-and-pepper" spikes —
 *   a single outlier ray can never survive because it is never the median
 *   of three values when its two neighbours are coherent.
 *   Window is wrap-around so no edge artefacts occur.
 */

#include "draw.hpp"
#include "geometry.hpp"
#include "simulation.hpp"

// ── Tunable parameters ───────────────────────────────────────────────────── //
namespace cfg {
    constexpr int   STATIC_COUNT   = 3;      // out of (genobs n - 1) dynamic obs

    // VFH+ scoring weights
    constexpr ftype HIST_DECAY     = 2.5f;
    constexpr ftype FORWARD_ALPHA  = 0.55f;  // slightly stronger forward bias
    constexpr ftype STEER_BETA     = 0.30f;

    // Speed / clearance thresholds
    constexpr ftype SAFE_DIST      = 0.30f;
    constexpr ftype BRAKE_DIST     = 0.12f;
    constexpr ftype ACCEL_SCALE    = 3.0f;
    constexpr ftype SPEED_SMOOTH   = 0.15f;

    // PD steering
    constexpr ftype STEER_KP       = 0.40f;
    constexpr ftype STEER_KD       = 0.10f;

    // Median filter window half-width  (total window = 2*W+1)
    constexpr int   MED_HALF_W     = 1;      // window = 3 rays
}
// ─────────────────────────────────────────────────────────────────────────── //

// ── Median of three values (no heap, no sort) ────────────────────────────── //
inline ftype median3(ftype a, ftype b, ftype c) {
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (a > b) swap(a, b);
    return b;  // middle element
}

// ── Generic median over a small window using partial insertion sort ───────── //
//    Works for MED_HALF_W up to ~4 without dynamic allocation.
inline ftype windowMedian(const array<ftype, rays>& d, int centre) {
    constexpr int W = 2 * cfg::MED_HALF_W + 1;  // window size (must be odd)
    ftype buf[W];
    for (int k = 0; k < W; ++k)
        buf[k] = d[(centre - cfg::MED_HALF_W + k + rays) % rays];
    // insertion sort (tiny W, no overhead)
    for (int i = 1; i < W; ++i) {
        ftype key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; --j; }
        buf[j + 1] = key;
    }
    return buf[W / 2];
}

int main() {

    // ── Persistent controller state ───────────────────────────────────── //
    static ftype prev_steer_err = 0.0f;
    static ftype filtered_accel = 0.0f;

    // ── Agent definition ──────────────────────────────────────────────── //
    agent myagent;

    myagent.calculate_1 =
        [](const envmap& curmap,
           const array<pair<point, point>, playercount>& playerdata,
           const array<point, rays>& raycasts,
           const agent& curplayer,
           ftype& a, ftype& steer) {

            const point& mypos = playerdata[0].first;

            // ── Step 1. Raw distances from raycasts ──────────────────── //
            array<ftype, rays> raw_dist{};
            for (int i = 0; i < rays; i++)
                raw_dist[i] = dist(mypos, raycasts[i]);

            // ── Step 2. Spatial Median Filter ────────────────────────── //
            //    Replace each distance with the median of its neighbourhood.
            //    Eliminates spike outliers while preserving real edges.
            array<ftype, rays> clean_dist{};
            for (int i = 0; i < rays; i++)
                clean_dist[i] = windowMedian(raw_dist, i);

            // ── Step 3. Build polar obstacle-cost histogram ──────────── //
            array<ftype, rays> hist{};
            for (int i = 0; i < rays; i++) {
                ftype d = max(clean_dist[i], 1e-3f);
                hist[i] = pow(1.0f / d, cfg::HIST_DECAY);
            }

            // ── Step 4. Smooth histogram (3-tap Gaussian) ────────────── //
            array<ftype, rays> smooth{};
            for (int i = 0; i < rays; i++) {
                smooth[i] = 0.25f * hist[(i - 1 + rays) % rays]
                          + 0.50f * hist[i]
                          + 0.25f * hist[(i + 1) % rays];
            }

            // ── Step 5. Score candidate directions (VFH+) ────────────── //
            int   best_ray   = 0;
            ftype best_score = -1e9f;
            for (int i = 0; i < rays; i++) {
                ftype delta = 2.0f * PI * i / (ftype)rays;
                if (delta > PI) delta -= 2.0f * PI;
                ftype score = -smooth[i]
                            + cfg::FORWARD_ALPHA * cos(delta)
                            - cfg::STEER_BETA    * abs(delta);
                if (score > best_score) { best_score = score; best_ray = i; }
            }

            // ── Step 6. PD steering ───────────────────────────────────── //
            ftype steer_err = 2.0f * PI * best_ray / (ftype)rays;
            if (steer_err > PI) steer_err -= 2.0f * PI;
            ftype d_err = steer_err - prev_steer_err;
            prev_steer_err = steer_err;
            steer = cfg::STEER_KP * steer_err + cfg::STEER_KD * d_err;

            // ── Step 7. Speed — forward ±60° cone clearance ───────────── //
            const int cone = rays / 6;
            ftype cone_dist = 0.0f;
            for (int i = -cone; i <= cone; i++)
                cone_dist += clean_dist[(i + rays) % rays];
            cone_dist /= (ftype)(2 * cone + 1);

            ftype target_a;
            if (cone_dist > cfg::SAFE_DIST) {
                target_a = acceldelta * cfg::ACCEL_SCALE;
            } else if (cone_dist < cfg::BRAKE_DIST) {
                target_a = -acceldelta * cfg::ACCEL_SCALE;
            } else {
                ftype t = (cone_dist - cfg::BRAKE_DIST)
                        / (cfg::SAFE_DIST - cfg::BRAKE_DIST);
                target_a = acceldelta * cfg::ACCEL_SCALE * (2.0f * t - 1.0f);
            }
            filtered_accel = cfg::SPEED_SMOOTH * target_a
                           + (1.0f - cfg::SPEED_SMOOTH) * filtered_accel;
            a = filtered_accel;
        };

    // ── Simulation setup ──────────────────────────────────────────────── //
    array<agent, playercount> myagents;
    for (int i = 0; i < playercount; i++) myagents[i] = myagent;

    simulationinstance s(myagents, 60.0f);

    // ── Obstacle movement specifiers ───────────────────────────────────── //
    const int num_obs = (int)s.mp.size() - 1;  // excludes boundary wall
    vector<vector<point>> initial_obs(s.mp.begin(), s.mp.begin() + num_obs);

    for (int i = 0; i < num_obs; i++) {
        if (i < cfg::STATIC_COUNT) {
            // Static obstacle — no movement
            s.movementspecifier[i] = [](vector<point>&, const ftype) {};
        } else {
            // Lissajous sinusoidal motion
            auto  init   = initial_obs[i];
            ftype phase  = 2.0f * PI * i / (ftype)(num_obs - cfg::STATIC_COUNT);
            ftype amp    = 0.04f + 0.015f * (i % 3);
            ftype period = 2.50f + 0.400f * (i % 5);

            s.movementspecifier[i] =
                [init, phase, amp, period](vector<point>& obs, const ftype t) {
                    ftype dx = amp * sin(2.0f * PI * t / period          + phase);
                    ftype dy = amp * cos(2.0f * PI * t / (period * 1.3f) + phase);
                    obs = point(dx, dy) + init;
                };
        }
    }

    s.humanmode  = false;
    s.visualmode = true;
    s.run();

    return 0;
}