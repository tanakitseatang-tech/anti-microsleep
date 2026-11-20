#!/bin/bash
# Build OSRM road data for Thailand on the board.
# Usage: osrm_build.sh [full|light|lighter]      Log: ~/osm/osrm_build.log
#   full    = standard car profile: all drivable roads incl. residential, service and
#             frontage roads (closest match to Google Maps). Needs the swap file
#             from add_swap.sh (peak memory is above the board's 3.6 GB RAM).
#   light   = no service roads / living streets
#   lighter = also no residential streets (fits in RAM without extra swap)
# Builds into a new folder and only swaps it in when every step succeeds,
# so a failed build keeps the old road data working.
set -o pipefail
LEVEL=${1:-full}
T=${THREADS:-2}                # fewer threads = lower peak memory (THREADS=1 osrm_build.sh light)
# Input map. thailand-roads.osm.pbf = roads + turn restrictions only (no buildings etc.),
# made with: osmium tags-filter thailand-latest.osm.pbf w/highway r/type=restriction
# It needs far less memory to build, so the full profile fits in the board's RAM.
PBF=${PBF:-/home/arduino/osm/thailand-roads.osm.pbf}
[ -f "$PBF" ] || PBF=/home/arduino/osm/thailand-latest.osm.pbf
IMG=ghcr.io/project-osrm/osrm-backend:v5.27.1
O=/home/arduino/osm
D=$O/osrm
N=$O/osrm_new
P=$O/car_$LEVEL.lua
rm -rf "$N"; mkdir -p "$N"
cp -f $O/car.lua "$P"
if [ "$LEVEL" != full ]; then
  # light profile: drop service + living_street from the routable road types
  sed -i -e '/^ *living_street *= *10,$/d' -e '/^ *service *= *15$/d' \
         -e 's/^\( *residential *= *25\),$/\1/' "$P"
fi
if [ "$LEVEL" = lighter ]; then
  sed -i -e '/^ *residential *= *25$/d' -e 's/^\( *unclassified *= *25\),$/\1/' "$P"
fi
echo "profile $LEVEL:"; sed -n '/speeds = Sequence/,/^    },/p' "$P" | grep '='
echo "input $PBF ($(du -h "$PBF" | cut -f1))"
echo "start $(date +%T)"
# free memory: stop the app and the running router (driver page falls back to straight lines)
arduino-app-cli app stop user:anti-microsleep >/dev/null 2>&1
docker stop osrm >/dev/null 2>&1
sync; free -m | sed -n '2,3p'
RUN="docker run --rm -v $N:/data -v $PBF:/src/thailand.osm.pbf:ro -v $P:/opt/car.lua:ro $IMG"
ok=1
# extract writes next to the input name, so give it /data/thailand.osm.pbf via a symlink
$RUN sh -c "ln -s /src/thailand.osm.pbf /data/thailand.osm.pbf && osrm-extract --threads $T -p /opt/car.lua /data/thailand.osm.pbf" || ok=0
rm -f "$N/thailand.osm.pbf"
[ $ok = 1 ] && echo "extract done $(date +%T)"
[ $ok = 1 ] && { $RUN osrm-partition --threads $T /data/thailand.osrm || ok=0; }
[ $ok = 1 ] && echo "partition done $(date +%T)"
[ $ok = 1 ] && { $RUN osrm-customize --threads $T /data/thailand.osrm || ok=0; }
[ $ok = 1 ] && echo "customize done $(date +%T)"
if [ $ok = 1 ]; then
  rm -rf "$O/osrm_old"; mv "$D" "$O/osrm_old"; mv "$N" "$D"; rm -rf "$O/osrm_old"
  echo "$LEVEL" > "$D/LEVEL"
else
  rm -rf "$N"
fi
docker start osrm >/dev/null 2>&1
arduino-app-cli app start user:anti-microsleep >/dev/null 2>&1
du -sh "$D"
if [ $ok = 1 ]; then echo "OSRM-BUILD-OK $(date +%T)"; else echo "OSRM-BUILD-FAILED $(date +%T)"; dmesg 2>/dev/null | grep -i 'out of memory' | tail -1; fi
