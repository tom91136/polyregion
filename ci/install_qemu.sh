#!/usr/bin/env bash
set -euo pipefail

arch="${1:?qemu arch (e.g. riscv64) required}"
version="10.0.13+ds-0+deb13u1"

case "$(dpkg --print-architecture)" in
  arm64) deb_arch=arm64; sha256=c73711af02b97cd5e2667e735d9c06890c57165517110ca4e646c95a7083158d ;;
  amd64) deb_arch=amd64; sha256=ca6ede739327a20ae5a3498230c8a3a9a072c586e131bc6bcc1201eb1725e336 ;;
  *) echo "install_qemu: unsupported host $(dpkg --print-architecture)" >&2; exit 1 ;;
esac

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
deb="qemu-user_${version}_${deb_arch}.deb"
curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors --connect-timeout 30 -o "$tmp/$deb" "https://snapshot.debian.org/archive/debian/20260905T205002Z/pool/main/q/qemu/${deb//+/%2B}"
echo "$sha256  $tmp/$deb" | sha256sum -c -
dpkg-deb -x "$tmp/$deb" "$tmp/root"

sudo install -Dm755 "$tmp/root/usr/bin/qemu-$arch" "/usr/bin/qemu-$arch"
sudo mkdir -p /usr/libexec/qemu-binfmt
sudo cp -a "$tmp/root/usr/libexec/qemu-binfmt/$arch-binfmt-P" /usr/libexec/qemu-binfmt/

[ -e /proc/sys/fs/binfmt_misc/register ] || sudo mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc
[ ! -e "/proc/sys/fs/binfmt_misc/qemu-$arch" ] || echo -1 | sudo tee "/proc/sys/fs/binfmt_misc/qemu-$arch" > /dev/null
sudo tee /proc/sys/fs/binfmt_misc/register < "$tmp/root/usr/share/qemu/binfmt.d/qemu-$arch.conf" > /dev/null
# XXX the max cpu enables Ztso, which fences every guest load and store
[ "$arch" != riscv64 ] || echo "QEMU_CPU=rv64" >> "${GITHUB_ENV:-/dev/null}"
"/usr/bin/qemu-$arch" --version | head -1
cat "/proc/sys/fs/binfmt_misc/qemu-$arch"
