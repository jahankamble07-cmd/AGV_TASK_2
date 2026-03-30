/*
 * Task 2: Dynamic Obstacle Avoidance Controller  —  VFH+ planner
 *
 * Environment
 * -----------
 *   10 obstacles (boundary wall always static):
 *     Indices [0 .. FIXED_OBS-1]   → fixed in place (parked / infrastructure)
 *     Indices [FIXED_OBS .. N-2]   → Lissajous sinusoidal motion
 *
 * Controller — Vector Field Histogram+ (VFH+)
 * -------------------------------------------
 *   1. Build a polar obstacle-cost histogram from raycasts (inverse-distance).
 *   2. Smooth histogram with a 3-sector Gaussian kernel to avoid boundary flicker.
 *   3. Score each direction:  -cost  +  alpha*cos(delta)  -  beta*|delta|
 *   4. PD steer toward best direction (derivative term damps oscillation).
 *   5. Speed: average clearance in front +-60 cone,
 *      exponential low-pass filter on acceleration to reduce jerk.
 */

#include "draw.hpp"
#include "geometry.hpp"
#include "simulation.hpp"

// ── Tunable parameters ──────────────────────────────────────────────────── //
namespace planner {
    constexpr int   FIXED_OBS     = 4;      // obstacles kept static

    constexpr ftype DIST_EXPONENT = 3.8f;   // inverse-dist cost exponent (higher → closer obs cost more)
    constexpr ftype BLUR[3]       = {0.20f, 0.60f, 0.20f};  // 3-tap Gaussian smoother

    constexpr ftype GOAL_WEIGHT   = 0.35f;  // reward for facing forward
    constexpr ftype TURN_PENALTY  = 0.12f;  // penalise large heading deviations

    constexpr ftype CLEAR_THRESH  = 0.50f;  // full-speed clearance threshold
    constexpr ftype STOP_THRESH   = 0.22f;  // hard-brake clearance threshold
    constexpr ftype DRIVE_GAIN    = 2.5f;   // acceleration output scale
    constexpr ftype VEL_ALPHA     = 0.10f;  // speed low-pass blend factor

    constexpr ftype KP            = 0.65f;  // proportional steer gain
    constexpr ftype KD            = 0.18f;  // derivative  steer gain
}
// ──────────────────────────────────────────────────────────────────────────── //

int main() {

    // ------------------------------------------------------------------ //
    //  Persistent planner state — plain statics, no heap allocation needed
    // ------------------------------------------------------------------ //
    static ftype prev_heading_err = 0.0f;
    static ftype smooth_accel     = 0.0f;

    // ------------------------------------------------------------------ //
    //  VFH+ Controller
    // ------------------------------------------------------------------ //
    agent myagent;

    myagent.calculate_1 =
        [](const envmap& curmap,
           const array<pair<point, point>, playercount>& playerdata,
           const array<point, rays>& raycasts,
           const agent& curplayer,
           ftype& a, ftype& steer) {

            const point& mypos = playerdata[0].first;

            // ── 1. Raw polar obstacle-cost histogram ─────────────────── //
            array<ftype, rays> hist{};
            for (int i = 0; i < rays; i++) {
                ftype d = dist(mypos, raycasts[i]);
                hist[i] = (d > 1e-3f) ? pow(1.0f / d, planner::DIST_EXPONENT) : 1e6f;
            }

            // ── 2. Smooth histogram with wrap-around ─────────────────── //
            array<ftype, rays> smooth{};
            for (int i = 0; i < rays; i++) {
                smooth[i] = planner::BLUR[0] * hist[(i - 1 + rays) % rays]
                          + planner::BLUR[1] * hist[i]
                          + planner::BLUR[2] * hist[(i + 1) % rays];
            }

            // ── 3. Score each candidate direction ────────────────────── //
            int   best_ray   = 0;
            ftype best_score = -1e9f;

            for (int i = 0; i < rays; i++) {
                ftype delta = 2.0f * PI * i / (ftype)rays;
                if (delta > PI) delta -= 2.0f * PI;

                ftype score = -smooth[i]
                            + planner::GOAL_WEIGHT  * cos(delta)
                            - planner::TURN_PENALTY * abs(delta);

                if (score > best_score) {
                    best_score = score;
                    best_ray   = i;
                }
            }

            // ── 4. PD steering toward best direction ─────────────────── //
            ftype heading_err = 2.0f * PI * best_ray / (ftype)rays;
            if (heading_err > PI) heading_err -= 2.0f * PI;

            ftype d_err = heading_err - prev_heading_err;
            prev_heading_err = heading_err;

            steer = planner::KP * heading_err + planner::KD * d_err;

            // ── 5. Speed: average clearance in front +-60 cone ───────── //
            const int cone = rays / 6;
            ftype cone_dist = 0.0f;
            for (int i = -cone; i <= cone; i++)
                cone_dist += dist(mypos, raycasts[((i % rays) + rays) % rays]);
            cone_dist /= (ftype)(2 * cone + 1);

            ftype target_a;
            if (cone_dist > planner::CLEAR_THRESH) {
                target_a = acceldelta * planner::DRIVE_GAIN;
            } else if (cone_dist < planner::STOP_THRESH) {
                target_a = -acceldelta * planner::DRIVE_GAIN;
            } else {
                ftype t = (cone_dist - planner::STOP_THRESH)
                        / (planner::CLEAR_THRESH - planner::STOP_THRESH);
                target_a = acceldelta * planner::DRIVE_GAIN * (2.0f * t - 1.0f);
            }

            smooth_accel = planner::VEL_ALPHA * target_a
                         + (1.0f - planner::VEL_ALPHA) * smooth_accel;
            a = smooth_accel;
        };

    // ------------------------------------------------------------------ //
    //  Simulation setup
    // ------------------------------------------------------------------ //
    array<agent, playercount> myagents;
    for (int i = 0; i < playercount; i++) myagents[i] = myagent;

    simulationinstance s(myagents, /*endtime=*/60.0f);

    // ------------------------------------------------------------------ //
    //  Obstacle movement specifiers
    //  [0 .. FIXED_OBS-1]  → static
    //  [FIXED_OBS .. N-2]  → Lissajous dynamic
    //  [N-1]               → boundary wall (always static)
    // ------------------------------------------------------------------ //
    const int num_obs = (int)s.mp.size() - 1;

    vector<vector<point>> initial_obs(s.mp.begin(), s.mp.begin() + num_obs);

    auto no_movement = [](vector<point>&, const ftype) {};

    for (int i = 0; i < num_obs; i++) {
        if (i < planner::FIXED_OBS) {
            s.movementspecifier[i] = no_movement;
        } else {
            auto  init   = initial_obs[i];
            ftype phase  = 2.0f * PI * i / (ftype)(num_obs - planner::FIXED_OBS);
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