#!/bin/bash
# Add a 4 GB swap file on the big /home/arduino partition (15 GB free) so the full
# OSRM road build fits in memory. Run once with sudo:  sudo bash ~/osm/add_swap.sh
# Kept across reboots via /etc/fstab. Low priority, so the faster zram swap is used first.
# NOTE: /home itself is on the small system disk; only /home/arduino is the big partition.
set -e
OLD=/home/swapfile                     # wrong place used by an earlier version: clean it up
if grep -q "^$OLD " /proc/swaps; then swapoff "$OLD"; fi
sed -i "\|^$OLD |d" /etc/fstab
rm -f "$OLD"
F=/home/arduino/swapfile
if ! grep -q "^$F " /proc/swaps; then
  rm -f "$F"
  fallocate -l 4G "$F"
  chmod 600 "$F"
  mkswap "$F" >/dev/null
  swapon -p 10 "$F"
fi
grep -q "^$F " /etc/fstab || echo "$F none swap sw,pri=10 0 0" >> /etc/fstab
df -h / /home/arduino
cat /proc/swaps
echo SWAP-OK
