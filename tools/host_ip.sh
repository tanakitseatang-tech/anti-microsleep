#!/bin/bash
# Runs on the board itself (not in Docker): every 5 s writes the board's Wi-Fi IP and network name
# to data/host_ip.json, so the app (inside Docker, which can't see it) can send it to the company
# website. Also auto-switches to AMS-01 hotspot whenever it appears (no reboot needed).
# Started at boot by cron:  @reboot bash ~/ArduinoApps/anti-microsleep/tools/host_ip.sh
OUT=/home/arduino/ArduinoApps/anti-microsleep/data/host_ip.json
PREF_SSID="AMS-01"
mkdir -p "$(dirname "$OUT")"
exec 9>/tmp/host_ip.lock
flock -n 9 || exit 0          # already running
TICK=0
while true; do
  IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
  if [ -z "$IP" ]; then   # no internet route: take the Wi-Fi address directly
    IP=$(ip -4 -o addr show scope global 2>/dev/null | grep -vE 'docker|br-|veth' | awk '{print $4}' | cut -d/ -f1 | head -1)
  fi
  SSID=$(nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | grep '^yes:' | head -1 | cut -d: -f2-)
  printf '{"ip":"%s","wifi":"%s","t":%s}\n' "$IP" "${SSID//\"/}" "$(date +%s)" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
  TICK=$((TICK + 1))
  if [ "$SSID" != "$PREF_SSID" ] && [ $((TICK % 3)) -eq 0 ]; then
    if nmcli -t -f SSID dev wifi list --rescan yes 2>/dev/null | grep -Fxq "$PREF_SSID"; then
      nmcli con up "$PREF_SSID" >/dev/null 2>&1 || true
    fi
  fi
  sleep 5
done
