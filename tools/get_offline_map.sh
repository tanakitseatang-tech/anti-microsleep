#!/bin/bash
# Download the official pmtiles CLI (protomaps/go-pmtiles v1.31.2, arm64), then size a Thailand extract.
set -e
cd ~/osm
if [ ! -x ./pmtiles ]; then
  curl -sL -o pmtiles.tgz https://github.com/protomaps/go-pmtiles/releases/download/v1.31.2/go-pmtiles_1.31.2_Linux_arm64.tar.gz
  tar xzf pmtiles.tgz pmtiles
  rm pmtiles.tgz
fi
./pmtiles version
# Thailand bounding box (lon/lat): west,south,east,north
BBOX=97.3,5.6,105.7,20.5
for Z in 13 14; do
  echo "== maxzoom $Z"
  ./pmtiles extract https://build.protomaps.com/20261118.pmtiles /dev/null --bbox=$BBOX --maxzoom=$Z --dry-run 2>&1 | tail -3
done
