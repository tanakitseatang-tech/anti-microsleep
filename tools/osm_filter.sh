#!/bin/sh
# Make thailand-roads.osm.pbf (drivable roads + turn restrictions only) from thailand-latest.osm.pbf.
# Uses osmium-tool from Debian inside a throwaway container, so nothing is installed on the board.
# Usage: sh ~/osm/osm_filter.sh        Log: ~/osm/filter.log
O=/home/arduino/osm
echo "start $(date +%T)"
df -h / | tail -1
docker run --rm -v $O:/osm debian:trixie-slim sh -c '
  set -e
  apt-get update -qq >/dev/null
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends osmium-tool >/dev/null
  osmium --version | head -1
  # drivable road types only (no footway/path/track/cycleway/steps): far fewer nodes,
  # so osrm-extract needs much less memory. Same set the car profile can route on.
  osmium tags-filter --overwrite -o /osm/thailand-roads.osm.pbf /osm/thailand-latest.osm.pbf \
    w/highway=motorway,motorway_link,trunk,trunk_link,primary,primary_link,secondary,secondary_link,tertiary,tertiary_link,unclassified,residential,living_street,service,road \
    r/type=restriction
  osmium fileinfo -e /osm/thailand-roads.osm.pbf | grep -E "Number of (nodes|ways|relations):"
  chown 1000:1000 /osm/thailand-roads.osm.pbf 2>/dev/null || true
' && echo "FILTER-OK $(date +%T)" || echo "FILTER-FAILED $(date +%T)"
ls -la $O/thailand-roads.osm.pbf 2>&1
df -h / | tail -1
