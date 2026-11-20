"""Extract rest stops (fuel, rest areas/services, cafes) from a Thailand .osm.pbf.

Usage:  python extract_pois.py thailand-latest.osm.pbf maps/pois.json
Needs:  pip install osmium
Output: {"updated", "source", "types": ["fuel","rest_area","cafe"],
         "pois": [[lat, lon, type_idx, name], ...]}
"""
import json
import sys
import time

import osmium

TYPES = ["fuel", "rest_area", "cafe"]


def kind(tags):
    if tags.get("amenity") == "fuel":
        return 0
    if tags.get("highway") in ("rest_area", "services"):
        return 1
    if tags.get("amenity") == "cafe":
        return 2
    return None


def label(tags):
    name = tags.get("name:th") or tags.get("name") or tags.get("name:en") or ""
    brand = tags.get("brand") or ""
    if brand and brand.lower() not in name.lower():
        name = f"{brand} {name}".strip()
    return name[:60]


def main(src, dst):
    pois, seen = [], set()
    fp = (osmium.FileProcessor(src)
          .with_areas()
          .with_filter(osmium.filter.KeyFilter("amenity", "highway")))
    for obj in fp:
        k = kind(obj.tags)
        if k is None:
            continue
        if obj.is_node():
            lat, lon = obj.location.lat, obj.location.lon
        elif obj.is_area():
            # centre of the outer ring's bounding box is good enough for a stop
            lats, lons = [], []
            for ring in obj.outer_rings():
                for n in ring:
                    lats.append(n.lat); lons.append(n.lon)
            if not lats:
                continue
            lat, lon = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
        else:
            continue
        key = (round(lat, 4), round(lon, 4), k)
        if key in seen:
            continue
        seen.add(key)
        pois.append([round(lat, 6), round(lon, 6), k, label(obj.tags)])
    out = {"updated": time.strftime("%Y-%m-%d"), "source": "© OpenStreetMap contributors (ODbL)",
           "types": TYPES, "pois": pois}
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    counts = {t: sum(1 for p in pois if p[2] == i) for i, t in enumerate(TYPES)}
    print("saved", dst, counts)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
