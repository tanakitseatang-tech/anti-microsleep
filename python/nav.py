"""
Navigation helpers for the driver page (all offline, on the board):

  * Location: GPS module on the MCU (pushed via Bridge "gps"), or a test location
    (maps/test_location.json) while no GPS fix is available.
  * Rest stops: maps/pois.json (OpenStreetMap: fuel, rest areas/services, cafes).
  * Nearest stop: straight-line pre-filter, preferring stops AHEAD of the truck,
    then real road distance from a local OSRM routing server when one is running
    (falls back to straight line otherwise).
"""

from collections import deque
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request

EARTH_R = 6371000.0
TYPES = ["fuel", "rest_area", "cafe"]
CELL = 0.1                     # grid index cell size (degrees, ~11 km)
SEARCH_KM = (5, 15, 40, 100)   # widen the search until something is found
CANDIDATES = 8                 # straight-line candidates sent to the router
AHEAD_DEG = 70                 # stop within +-70 deg of the heading = "ahead"
AHEAD_MIN_KMH = 15             # heading only trusted above this speed
BEHIND_PENALTY = 2.5           # behind-the-truck stops count as this much farther
RECOMPUTE_S = 5.0
RECOMPUTE_M = 50.0
GPS_STALE_S = 6.0
GPS_FIX_HOLD_S = 120.0         # keep last valid real GPS fix briefly indoors; fall back to test_location after 2 min
OSRM_PORT = 5000
OSRM_TIMEOUT_S = 2.0
HEADING_TOL_DEG = 45           # route must leave the truck within +-45 deg of its heading
# Thailand drives on the LEFT: a stop on the right of the heading is across the road.
# Crossing a divided highway means a U-turn, which is slow and hard for a truck,
# so an opposite-side stop counts as this much farther (on top of the road distance).
OPPOSITE_SIDE_PENALTY_M = 1000
SIDE_MAX_M = 3000              # only judge the side for stops this close (farther = road may bend)
# OSRM moves ("snaps") each point to the nearest road it knows. The board's road data has
# no small roads, so a stop on a lane can snap to a main road far away and the road
# distance then ends there, not at the stop. Snap gaps are added to the distance; a stop
# (or the truck) farther than this from any known road gets no road distance at all.
SNAP_MAX_M = 500
SNAP_SPEED_MS = 20 / 3.6       # time estimate for the off-road gap (~20 km/h on a lane)
OFF_ROAD_FACTOR = 1.4          # ranking only: typical road/straight ratio for an off-road stop


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def side_of_road(heading, bearing, dist_m):
    """'same' (left of heading = our side when driving on the left), 'opposite', or None."""
    if heading is None or dist_m > SIDE_MAX_M:
        return None
    rel = (bearing - heading) % 360
    if rel < 10 or rel > 350 or 170 < rel < 190:   # dead ahead / behind: can't tell
        return None
    return "opposite" if rel < 180 else "same"


def _docker_host_ip():
    """IP of the host as seen from inside the app container (default gateway)."""
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                p = line.split()
                if p[1] == "00000000":
                    g = int(p[2], 16)
                    return ".".join(str((g >> s) & 255) for s in (0, 8, 16, 24))
    except Exception:
        pass
    return "127.0.0.1"


class PoiIndex:
    def __init__(self, path):
        self.pois, self.grid, self.updated = [], {}, None
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            self.pois, self.updated = d.get("pois", []), d.get("updated")
        except Exception:
            return
        for i, (lat, lon, _k, _n) in enumerate(self.pois):
            self.grid.setdefault((int(lat // CELL), int(lon // CELL)), []).append(i)

    def near(self, lat, lon, radius_km, kind=None):
        r = int(math.ceil(radius_km / 11.0)) + 1
        ci, cj = int(lat // CELL), int(lon // CELL)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for i in self.grid.get((ci + di, cj + dj), ()):
                    p = self.pois[i]
                    if kind is None or p[2] == kind:
                        yield p

    def in_bbox(self, s, w, n, e, limit=400):
        out = []
        for i in range(int(s // CELL), int(n // CELL) + 1):
            for j in range(int(w // CELL), int(e // CELL) + 1):
                for k in self.grid.get((i, j), ()):
                    p = self.pois[k]
                    if s <= p[0] <= n and w <= p[1] <= e:
                        out.append(p)
                        if len(out) >= limit:
                            return out
        return out


class Navigator:
    def __init__(self, maps_dir, logger):
        self.log = logger
        self.maps_dir = maps_dir
        self.index = PoiIndex(os.path.join(maps_dir, "pois.json"))
        self.test_file = os.path.join(maps_dir, "test_location.json")
        self.last_gps_file = os.path.join(maps_dir, "last_gps.json")
        self.test_loc = self._load_test()
        self.lock = threading.Lock()
        self.calc_lock = threading.Lock()
        self.gps = {"state": -1, "t": 0.0}
        self._sats_hist = deque(maxlen=12)      # (t, sats) for stable satellite count
        self._pos_hist = deque(maxlen=8)        # (t, lat, lon, raw_speed) for speed & coord smoothing
        self._smooth_lat = None
        self._smooth_lon = None
        self._smooth_speed = 0.0
        self._last_fix = self._load_last_gps()  # persisted real GPS fix for indoor / blind-spot entry
        self._last_gps_save_t = 0.0
        self.osrm = f"http://{_docker_host_ip()}:{OSRM_PORT}"
        self.osrm_ok = None
        self.result, self._last_at, self._last_t = {}, None, 0.0
        self._sim = {
            "active": False, "pts": [], "seg": 0, "dist": 0.0,
            "speed_kmh": 80.0, "t": 0.0, "lat": None, "lon": None,
            "course": 120.0, "name": "บางนา-ตราด → ชลบุรี (จำลอง)",
        }
        logger.info(f"Rest stops loaded: {len(self.index.pois)} (OSM {self.index.updated}); router {self.osrm}")

    # ------------------------------------------------------------ location
    def _load_last_gps(self):
        try:
            with open(self.last_gps_file) as f:
                d = json.load(f)
            # keep the real time of that fix (not "now"), so after a reboot an old position is
            # known to be old and is not sent to the company as if it were live
            t = d.get("t")
            if not isinstance(t, (int, float)):
                t = os.path.getmtime(self.last_gps_file)
            return {
                "state": 1, "t": float(t), "sats": int(d.get("sats", 6)), "hdop": 1.5,
                "lat": float(d["lat"]), "lon": float(d["lon"]),
                "speed": 0.0, "course": d.get("course"), "indoor_hold": True,
            }
        except Exception:
            return None

    def _save_last_gps(self, g):
        now = time.time()
        if now - self._last_gps_save_t < 5.0:
            return
        self._last_gps_save_t = now
        try:
            with open(self.last_gps_file, "w") as f:
                json.dump({
                    "lat": g["lat"], "lon": g["lon"],
                    "sats": g.get("sats", 6), "course": g.get("course"), "t": now,
                }, f)
        except Exception:
            pass

    def _load_test(self):
        try:
            with open(self.test_file) as f:
                d = json.load(f)
            c = d.get("course")
            return float(d["lat"]), float(d["lon"]), (None if c is None else float(c) % 360)
        except Exception:
            return None

    def set_test_location(self, lat, lon, course=None):
        """Test location while there is no GPS. course (deg, 0 = north) = pretend driving direction."""
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("bad coordinates")
        course = None if course in (None, "") else float(course) % 360
        with open(self.test_file, "w") as f:
            json.dump({"lat": lat, "lon": lon, "course": course}, f)
        with self.lock:
            self.test_loc, self._last_at = (lat, lon, course), None

    def on_gps(self, state, lat_e6, lon_e6, speed_x10, course_x10, sats, hdop_x10):
        """Bridge handler: called by the MCU once per second. Stabilizes sats, coords, and walking speed."""
        now = time.time()
        raw_state = int(state)
        raw_sats = max(0, int(sats))

        with self.lock:
            # 1. Stabilize satellite count over a 10-second rolling window
            if raw_sats > 0:
                self._sats_hist.append((now, raw_sats))
            while self._sats_hist and now - self._sats_hist[0][0] > 10.0:
                self._sats_hist.popleft()
            if self._sats_hist:
                vals = sorted(v for _, v in self._sats_hist)
                stable_sats = vals[len(vals) // 2]   # median prevents 1-second spikes/drops
            else:
                stable_sats = raw_sats

            g = {"state": raw_state, "t": now, "sats": stable_sats, "hdop": hdop_x10 / 10.0}

            if raw_state == 1 and (lat_e6 != 0 or lon_e6 != 0):
                raw_lat, raw_lon = lat_e6 / 1e6, lon_e6 / 1e6
                raw_spd = max(0.0, speed_x10 / 10.0)

                # Smooth coordinates (EMA) to suppress multipath wander when walking/stationary
                if self._smooth_lat is None or haversine_m(self._smooth_lat, self._smooth_lon, raw_lat, raw_lon) > 80.0:
                    self._smooth_lat, self._smooth_lon = raw_lat, raw_lon
                else:
                    alpha = 0.35 if raw_spd < 12.0 else 0.65
                    self._smooth_lat += alpha * (raw_lat - self._smooth_lat)
                    self._smooth_lon += alpha * (raw_lon - self._smooth_lon)

                self._pos_hist.append((now, self._smooth_lat, self._smooth_lon, raw_spd))
                while len(self._pos_hist) > 1 and now - self._pos_hist[0][0] > 5.0:
                    self._pos_hist.popleft()

                # Calculate stabilized speed: check real net displacement over window to kill stationary GPS drift
                if len(self._pos_hist) >= 2:
                    t0, lat0, lon0, _ = self._pos_hist[0]
                    dt = max(1.0, now - t0)
                    net_dist_m = haversine_m(lat0, lon0, self._smooth_lat, self._smooth_lon)
                    net_kmh = (net_dist_m / dt) * 3.6
                    med_raw = sorted(p[3] for p in self._pos_hist)[len(self._pos_hist) // 2]
                    # If moved < 2.2 m over the window or both speeds < 1.3 km/h -> standing still (0 km/h)
                    if (dt >= 2.5 and net_dist_m < 2.2) or (med_raw < 1.3 and net_kmh < 1.3):
                        target_spd = 0.0
                    else:
                        target_spd = 0.6 * med_raw + 0.4 * net_kmh
                else:
                    target_spd = 0.0 if raw_spd < 1.5 else raw_spd

                if target_spd == 0.0 and self._smooth_speed < 1.8:
                    self._smooth_speed = 0.0
                else:
                    self._smooth_speed = 0.4 * target_spd + 0.6 * self._smooth_speed
                    if self._smooth_speed < 0.8:
                        self._smooth_speed = 0.0

                g.update(
                    lat=round(self._smooth_lat, 6),
                    lon=round(self._smooth_lon, 6),
                    speed=round(self._smooth_speed, 1),
                    course=course_x10 / 10.0 if course_x10 >= 0 else None,
                    indoor_hold=False,
                )
                self._last_fix = dict(g)
                self._save_last_gps(g)
            elif self._last_fix and (now - self._last_fix["t"] < GPS_FIX_HOLD_S):
                # Hold the last known real GPS / indoor Wi-Fi fix when inside a building or blind spot
                decay = 0.7 if (now - self._last_fix["t"] < 8.0) else 0.0
                self._smooth_speed = round(self._smooth_speed * decay, 1) if self._smooth_speed >= 0.8 else 0.0
                g = {
                    **self._last_fix,
                    "state": 1,
                    "t": now,
                    "sats": max(stable_sats, self._last_fix.get("sats", 5)),
                    "speed": self._smooth_speed,
                    "indoor_hold": (now - self._last_fix["t"]) >= 8.0,
                }

            if self.gps.get("state") != g["state"]:
                self.log.info(f"GPS state {self.gps.get('state')} -> {g['state']} (sats {g['sats']})")
            self.gps = g

    def set_indoor_location(self, lat, lon, speed=None, course=None):
        """Indoor Wi-Fi / A-GPS assist from the driver's browser when satellite signals are blocked inside buildings."""
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("bad coordinates")
        now = time.time()
        with self.lock:
            # Only override if hardware satellite GPS is not actively locked right this second
            if self._last_fix and not self._last_fix.get("indoor_hold") and (now - self._last_fix.get("t", 0) < 4.0):
                return
            spd = 0.0 if speed is None else max(0.0, round(float(speed) * 3.6, 1))
            if spd < 1.2:
                spd = 0.0
            g = {
                "state": 1, "t": now, "sats": max(self.gps.get("sats", 0), 5), "hdop": 1.5,
                "lat": round(lat, 6), "lon": round(lon, 6),
                "speed": spd, "course": None if course is None else round(float(course) % 360, 1),
                "indoor_hold": True,
            }
            self._smooth_lat, self._smooth_lon, self._smooth_speed = lat, lon, spd
            self._last_fix = dict(g)
            self.gps = g
            self._save_last_gps(g)

    def start_sim(self, speed_kmh=80.0):
        """Start driving along a real Thai highway (Bang Na -> Chonburi) for indoor demos."""
        speed = max(20.0, min(120.0, float(speed_kmh or 80.0)))
        start_pt, end_pt = (13.6682, 100.6340), (13.3611, 100.9847)
        pts = None
        coords = f"{start_pt[1]:.6f},{start_pt[0]:.6f};{end_pt[1]:.6f},{end_pt[0]:.6f}"
        for base in (self.osrm, "https://router.project-osrm.org"):
            try:
                u = f"{base}/route/v1/driving/{coords}?overview=full&geometries=geojson"
                with urllib.request.urlopen(u, timeout=4.0) as r:
                    d = json.load(r)
                if d.get("code") == "Ok" and d.get("routes"):
                    pts = [(c[1], c[0]) for c in d["routes"][0]["geometry"]["coordinates"]]
                    if len(pts) >= 2:
                        break
            except Exception:
                continue
        if not pts or len(pts) < 2:
            # Offline fallback waypoints along Bang Na-Trad / Highway 34 -> Chonburi
            pts = [
                (13.6682, 100.6340), (13.6572, 100.6780), (13.6408, 100.7155),
                (13.6115, 100.7580), (13.5880, 100.8120), (13.5685, 100.8750),
                (13.5320, 100.9280), (13.4850, 100.9680), (13.4220, 100.9860),
                (13.3611, 100.9847),
            ]
        course = bearing_deg(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
        with self.lock:
            self._sim = {
                "active": True, "pts": pts, "seg": 0, "dist": 0.0,
                "speed_kmh": speed, "t": time.monotonic(),
                "lat": pts[0][0], "lon": pts[0][1], "course": course,
                "name": "บางนา-ตราด → ชลบุรี (จำลอง)",
            }
            self._last_at = None
        self.log.info(f"Demo drive simulator started ({len(pts)} road points, {speed:.0f} km/h)")
        return True

    def stop_sim(self):
        with self.lock:
            self._sim["active"] = False
            self._last_at = None
        self.log.info("Demo drive simulator stopped")
        return False

    def _step_sim(self):
        s = self._sim
        pts = s["pts"]
        if not s["active"] or len(pts) < 2:
            return None
        now = time.monotonic()
        dt = max(0.0, min(5.0, now - s["t"]))
        s["t"] = now
        move_m = dt * (s["speed_kmh"] / 3.6)
        seg, dist = s["seg"], s["dist"] + move_m
        while seg < len(pts) - 1:
            a, b = pts[seg], pts[seg + 1]
            seg_len = max(0.5, haversine_m(a[0], a[1], b[0], b[1]))
            if dist <= seg_len:
                frac = dist / seg_len
                s["lat"] = a[0] + (b[0] - a[0]) * frac
                s["lon"] = a[1] + (b[1] - a[1]) * frac
                s["course"] = round(bearing_deg(a[0], a[1], b[0], b[1]), 1)
                s["seg"], s["dist"] = seg, dist
                return dict(s)
            dist -= seg_len
            seg += 1
        # Loop back to start cleanly
        s["seg"], s["dist"] = 0, 0.0
        s["lat"], s["lon"] = pts[0]
        s["course"] = round(bearing_deg(pts[0][0], pts[0][1], pts[1][0], pts[1][1]), 1)
        return dict(s)

    def location(self):
        with self.lock:
            sim = self._step_sim() if self._sim.get("active") else None
            g, last_fix, test = dict(self.gps), self._last_fix, self.test_loc
        if sim is not None:
            return {
                "gps": 1, "sats": 9, "mcu_age": 0.0, "indoor": False,
                "lat": sim["lat"], "lon": sim["lon"], "speed": sim["speed_kmh"],
                "course": sim["course"], "source": "sim", "sim": True,
                "route_name": sim["name"],
            }
        age = time.time() - g["t"] if g.get("t") else None
        fresh = age is not None and age < GPS_STALE_S
        gps_state = g.get("state", -1) if fresh else (1 if last_fix else -1)
        base = {
            "gps": gps_state,
            "sats": g.get("sats", 0) if fresh else (last_fix.get("sats", 5) if last_fix else 0),
            "mcu_age": None if age is None else round(age, 1),
            "indoor": bool(g.get("indoor_hold") if fresh else True),
        }
        if gps_state == 1 and (fresh or last_fix):
            src = g if (fresh and g.get("lat") is not None) else last_fix
            # stale = last known position older than the hold time (e.g. saved before a reboot):
            # if a test_location is configured for indoor demo, prefer it over a stale old fix
            stale = src is last_fix and time.time() - last_fix.get("t", 0) >= GPS_FIX_HOLD_S
            if not stale or not test:
                return {**base, "lat": src["lat"], "lon": src["lon"], "speed": src.get("speed", 0.0),
                        "course": src.get("course"), "source": "gps", "stale": stale}
        if test:
            return {**base, "lat": test[0], "lon": test[1], "speed": None, "course": test[2], "source": "test"}
        return {**base, "lat": None, "lon": None, "source": "none"}

    # ------------------------------------------------------------ routing
    def _osrm(self, url, heading, n_others):
        """GET an OSRM url. With a heading, the route must leave the truck in its driving
        direction (so a stop across a divided road costs the real U-turn); if no road
        matches that direction, retry without it."""
        tries = [url]
        if heading is not None:
            bearings = f"{round(heading) % 360},{HEADING_TOL_DEG}" + ";" * n_others
            tries.insert(0, f"{url}&bearings={bearings}")
        last = None
        for u in tries:
            try:
                with urllib.request.urlopen(u, timeout=OSRM_TIMEOUT_S) as r:
                    d = json.load(r)
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                continue
            if d.get("code") == "Ok":
                return d
            last = d.get("code")
        raise RuntimeError(last)

    def _road(self, lat, lon, pts, heading=None):
        """Road (distance, duration, snap_ok) from (lat, lon) to each point via OSRM /table.
        Distances include the off-road gaps at both ends. snap_ok is False (and distance/
        duration None) when the point is farther than SNAP_MAX_M from any known road.
        Returns None if routing is unavailable."""
        coords = ";".join(f"{p[1]:.6f},{p[0]:.6f}" for p in [(lat, lon)] + pts)
        url = (f"{self.osrm}/table/v1/driving/{coords}?sources=0"
               f"&annotations=distance,duration")
        try:
            d = self._osrm(url, heading, len(pts))
            if self.osrm_ok is not True:
                self.log.info("Road routing (OSRM) available")
            self.osrm_ok = True
            # destinations = every coordinate (index 0 is the truck); sources = the truck only
            snaps = [w.get("distance") or 0.0 for w in d.get("destinations", [])]
            srcs = d.get("sources") or []
            src_gap = (srcs[0].get("distance") or 0.0) if srcs else (snaps[0] if snaps else 0.0)
            if src_gap > SNAP_MAX_M:          # the truck itself is off the known roads
                return [(None, None, False)] * len(pts)
            out = []
            for i, (rd, rt) in enumerate(zip(d["distances"][0][1:], d["durations"][0][1:])):
                gap = snaps[i + 1] if i + 1 < len(snaps) else 0.0
                if rd is None or gap > SNAP_MAX_M:
                    out.append((None, None, False))
                else:
                    g = src_gap + gap
                    out.append((rd + g, (rt or 0.0) + g / SNAP_SPEED_MS, True))
            return out
        except Exception as e:
            if self.osrm_ok is not False:
                self.log.info(f"Road routing unavailable ({e}); using straight-line distance")
            self.osrm_ok = False
            return None

    def _route(self, lat, lon, dlat, dlon, heading=None):
        """Road path [[lat, lon], ...] from (lat, lon) to (dlat, dlon) via OSRM /route. None if unavailable."""
        url = (f"{self.osrm}/route/v1/driving/{lon:.6f},{lat:.6f};{dlon:.6f},{dlat:.6f}"
               f"?overview=full&geometries=geojson&steps=false")
        try:
            d = self._osrm(url, heading, 1)
            if not d.get("routes"):
                return None
            # GeoJSON is [lon, lat]; Leaflet wants [lat, lon]. 5 decimals ~ 1 m.
            pts = [[round(c[1], 5), round(c[0], 5)] for c in d["routes"][0]["geometry"]["coordinates"]]
            # the route runs between the snapped road points: join it to the real truck and stop
            return [[round(lat, 5), round(lon, 5)]] + pts + [[round(dlat, 5), round(dlon, 5)]]
        except Exception:
            return None

    def _nearest(self, loc, kind):
        lat, lon = loc["lat"], loc["lon"]
        # GPS heading is only trusted when moving; a test location's course is used as given
        trusted = loc.get("source") == "test" or (loc.get("speed") or 0) >= AHEAD_MIN_KMH
        heading = loc.get("course") if trusted else None
        cands = []
        for km in SEARCH_KM:
            cands = []
            for p in self.index.near(lat, lon, km, kind):
                d = haversine_m(lat, lon, p[0], p[1])
                if d > km * 1000:
                    continue
                b = bearing_deg(lat, lon, p[0], p[1])
                ahead = None if heading is None else angle_diff(b, heading) <= AHEAD_DEG
                score = d * (BEHIND_PENALTY if ahead is False else 1.0)
                cands.append((score, d, b, ahead, p))
            if cands:
                break
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        cands = cands[:CANDIDATES]
        road = self._road(lat, lon, [(c[4][0], c[4][1]) for c in cands], heading)
        best = None
        for i, (score, d, b, ahead, p) in enumerate(cands):
            rd, rt, snap_ok = road[i] if road else (None, None, True)
            if rd is not None and rd < d:     # a road can never be shorter than a straight line
                rt = (rt or 0.0) * d / rd if rd > 0 else d / SNAP_SPEED_MS
                rd = d
            off_road = road is not None and not snap_ok
            side = side_of_road(heading, b, d)
            base = rd if rd is not None else d * (OFF_ROAD_FACTOR if off_road else 1.0)
            key = base * (BEHIND_PENALTY if ahead is False else 1.0)
            if side == "opposite":
                key += OPPOSITE_SIDE_PENALTY_M
            if best is None or key < best[0]:
                best = (key, {"lat": p[0], "lon": p[1], "name": p[3], "type": TYPES[p[2]],
                              "straight_m": round(d), "road_m": None if rd is None else round(rd),
                              "road_s": None if rt is None else round(rt), "off_road": off_road,
                              "bearing": round(b), "ahead": ahead, "side": side})
        n = best[1]
        n["route"] = (self._route(lat, lon, n["lat"], n["lon"], heading)
                      if road and n["road_m"] is not None else None)
        return n

    def state(self):
        loc = self.location()
        out = {"loc": loc, "routing": bool(self.osrm_ok), "pois_updated": self.index.updated}
        if loc["lat"] is None:
            out["nearest"] = {}
            return out
        with self.calc_lock:                      # several browser tabs may poll at once
            now = time.monotonic()
            moved = (self._last_at is None or
                     haversine_m(loc["lat"], loc["lon"], *self._last_at) > RECOMPUTE_M)
            if moved or now - self._last_t > RECOMPUTE_S:
                self.result = {t: self._nearest(loc, k) for k, t in enumerate(TYPES)}
                self._last_at, self._last_t = (loc["lat"], loc["lon"]), now
            out["nearest"] = self.result
        return out

    def pois_in(self, s, w, n, e):
        return self.index.in_bbox(s, w, n, e)
