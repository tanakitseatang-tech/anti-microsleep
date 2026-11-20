#!/bin/bash
# Run the OSRM routing server permanently (restarts on boot) and test one route in Bangkok.
IMG=ghcr.io/project-osrm/osrm-backend:v5.27.1
docker rm -f osrm >/dev/null 2>&1
docker run -d --name osrm --restart unless-stopped -p 5000:5000 \
  -v /home/arduino/osm/osrm:/data $IMG \
  osrm-routed --algorithm mld --max-table-size 50 /data/thailand.osrm >/dev/null
for i in $(seq 1 60); do
  curl -s -o /dev/null http://localhost:5000/route/v1/driving/100.5018,13.7563%3B100.5078,13.7561 && break
  sleep 2
done
echo "route test:"
curl -s "http://localhost:5000/route/v1/driving/100.5018,13.7563;100.507824,13.756112?overview=false" | head -c 300; echo
docker stats --no-stream --format '{{.Name}} MEM {{.MemUsage}}' osrm
