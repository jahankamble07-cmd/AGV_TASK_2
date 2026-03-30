#include "draw.hpp"
#include "geometry.hpp"
#include "simulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr ftype GRID_STEP     = 0.20f;
constexpr ftype WP_REACH_DIST = 0.06f;
constexpr ftype ADJ_FACTOR    = 1.50f;
constexpr ftype BORDER_SAFE   = 0.10f;

constexpr ftype KP = 0.22f;
constexpr ftype KI = 0.0002f;
constexpr ftype KD = 0.06f;

constexpr ftype DIST_EMERGENCY = 0.10f;
constexpr ftype DIST_DANGER    = 0.20f;
constexpr ftype DIST_CAUTION   = 0.35f;
constexpr int   CONE_HALF      = 3;

constexpr int   WP_LIMIT       = 160;

const ftype ACCEL_FULL = acceldelta;
const ftype ACCEL_HALF = acceldelta * 0.5f;

ftype wrapAngle(ftype a) {
    while (a >  PI) a -= 2.f * PI;
    while (a < -PI) a += 2.f * PI;
    return a;
}

bool inFreeSpace(const point& p, const envmap& map) {
    if (!contains(map.back(), p)) return false;
    for (int i = 0; i < (int)map.size() - 1; ++i)
        if (contains(map[i], p)) return false;
    return true;
}

vector<point> buildWaypoints(const envmap& map, const point& agentStart) {
    const ftype lo = -1.0f + BORDER_SAFE;
    const ftype hi =  1.0f - BORDER_SAFE;

    vector<point> freeGrid;
    for (ftype x = lo; x <= hi; x += GRID_STEP)
        for (ftype y = lo; y <= hi; y += GRID_STEP)
            if (inFreeSpace({x, y}, map))
                freeGrid.push_back({x, y});

    const int N = (int)freeGrid.size();
    const ftype adjThresh = ADJ_FACTOR * GRID_STEP;
    vector<point> centroids;

    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            if (dist(freeGrid[i], freeGrid[j]) > adjThresh) continue;
            for (int k = j + 1; k < N; ++k) {
                if (dist(freeGrid[i], freeGrid[k]) > adjThresh) continue;
                if (dist(freeGrid[j], freeGrid[k]) > adjThresh) continue;
                point c = {
                    (freeGrid[i].x + freeGrid[j].x + freeGrid[k].x) / 3.f,
                    (freeGrid[i].y + freeGrid[j].y + freeGrid[k].y) / 3.f
                };
                if (inFreeSpace(c, map))
                    centroids.push_back(c);
            }
        }

    if (centroids.empty()) centroids = freeGrid;

    vector<bool>  visited(centroids.size(), false);
    vector<point> ordered;
    ordered.reserve(centroids.size());
    point cursor = agentStart;

    while ((int)ordered.size() < (int)centroids.size()) {
        ftype bestD = 1e9f;
        int   bestI = -1;
        for (int i = 0; i < (int)centroids.size(); ++i) {
            if (visited[i]) continue;
            ftype d = dist(cursor, centroids[i]);
            if (d < bestD) { bestD = d; bestI = i; }
        }
        if (bestI == -1) break;
        visited[bestI] = true;
        ordered.push_back(centroids[bestI]);
        cursor = centroids[bestI];
    }

    int limit = (int)ordered.size() < WP_LIMIT ? (int)ordered.size() : WP_LIMIT;
    ordered.resize(limit);
    return ordered;
}

int main() {
    vector<point> waypoints;
    int   wpIdx      = 0;
    bool  wpBuilt    = false;

    ftype pidIntegral = 0.f;
    ftype pidPrevErr  = 0.f;

    agent myagent;

    myagent.calculate_1 = [&](
        const envmap&                                curmap,
        const array<pair<point,point>, playercount>& playerdata,
        const array<point, rays>&                    raycasts,
        const agent&                                 curplayer,
        ftype&                                       a,
        ftype&                                       steer) -> void
    {
        const point agentPos   = playerdata[0].first;
        const ftype agentTheta = playerdata[0].second.y;
        const ftype agentSpeed = playerdata[0].second.x;

        if (!wpBuilt) {
            waypoints = buildWaypoints(curmap, agentPos);
            wpBuilt   = true;
            printf("[Task 1] Built %zu waypoints (capped at %d).\n", waypoints.size(), WP_LIMIT);
        }

        if (wpIdx >= (int)waypoints.size()) {
            a     = (agentSpeed > 1e-4f) ? -ACCEL_FULL : 0.f;
            steer = 0.f;
            return;
        }

        array<ftype, rays> distances;
        for (int i = 0; i < rays; ++i)
            distances[i] = dist(agentPos, raycasts[i]);

        ftype forwardDist = distances[0];
        int   threatRay   = 0;
        for (int i = 1; i <= CONE_HALF; ++i) {
            if (distances[i] < forwardDist)
                { forwardDist = distances[i];        threatRay = i; }
            if (distances[rays - i] < forwardDist)
                { forwardDist = distances[rays - i]; threatRay = rays - i; }
        }

        bool  threatOnLeft   = (threatRay >= 1 && threatRay <= CONE_HALF);
        ftype emergencySteer = threatOnLeft ? -(PI / 6.f) : (PI / 6.f);

        if (forwardDist < DIST_EMERGENCY) {
            a     = -ACCEL_FULL;
            steer = emergencySteer;
            ++wpIdx;
            pidIntegral = 0.f;
            printf("[Task 1] Emergency! Skipping to waypoint %d.\n", wpIdx + 1);
            return;
        }

        if (forwardDist < DIST_DANGER) {
            a = -ACCEL_FULL;
            const point& tgt   = waypoints[wpIdx];
            ftype desiredTheta = atan2(tgt.y - agentPos.y, tgt.x - agentPos.x);
            ftype pidSteer     = KP * wrapAngle(desiredTheta - agentTheta);
            steer = 0.7f * emergencySteer + 0.3f * pidSteer;
            return;
        }

        const point& target = waypoints[wpIdx];
        ftype dx = target.x - agentPos.x;
        ftype dy = target.y - agentPos.y;

        if (hypot(dx, dy) < WP_REACH_DIST) {
            printf("[Task 1] Waypoint %d / %zu reached.\n", wpIdx + 1, waypoints.size());
            ++wpIdx;
            pidIntegral = 0.f;
            return;
        }

        ftype desiredTheta = atan2(dy, dx);
        ftype headingErr   = wrapAngle(desiredTheta - agentTheta);

        constexpr ftype VCLIP = 0.01f;
        ftype dynKP = KP * (1.0f - 0.5f * agentSpeed / VCLIP);

        pidIntegral += headingErr * (ftype)dt;
        pidIntegral  = clip(pidIntegral, PI);
        ftype deriv  = (headingErr - pidPrevErr) / (ftype)dt;
        pidPrevErr   = headingErr;

        steer = dynKP * headingErr + KI * pidIntegral + KD * deriv;

        bool sharpTurn   = fabsf(headingErr) > PI / 4.f;
        bool cautionZone = (forwardDist < DIST_CAUTION);
        a = (sharpTurn || cautionZone) ? ACCEL_HALF : ACCEL_FULL;
    };

    array<agent, playercount> myagents;
    for (int i = 0; i < playercount; ++i) myagents[i] = myagent;

    simulationinstance s(myagents, 60.f);
    s.humanmode  = false;
    s.visualmode = true;
    s.run();
    return 0;
}