#!/usr/bin/env bash
set -euo pipefail

ram=$(awk '/^MemTotal:/ {print $2 * 1024}' /proc/meminfo)
target=$((ram * 4))

zram_ready() {
  modprobe zram 2>/dev/null && return 0
  apt-get install -y "linux-modules-extra-$(uname -r)" || return 1
  modprobe zram
}

swapoff -a
rm -f /mnt/swapfile

if zram_ready; then
  echo "$target" > /sys/block/zram0/disksize
  mkswap /dev/zram0
  swapon -p 5 /dev/zram0
  echo "swap: zram0 ${target} bytes"
else
  echo "swap: zram unavailable, falling back to swapfile"
  read -r avail path < <(df -B1 --output=avail,target -t ext4 -t xfs -t btrfs | tail -n +2 | sort -rn | head -n 1)
  cap=$((avail / 4))
  size=$((ram < cap ? ram : cap))
  swapfile="${path%/}/swapfile"
  rm -f "$swapfile"
  fallocate -l "$size" "$swapfile"
  chmod 600 "$swapfile"
  mkswap "$swapfile"
  swapon -p 5 "$swapfile"
  echo "swap: swapfile $swapfile ${size} bytes"
fi

swapon --show
