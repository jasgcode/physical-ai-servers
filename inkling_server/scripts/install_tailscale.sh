#!/bin/bash
# 一次性安装 tailscale 到用户家目录（不需要 root）
set -e

INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

VERSION="1.78.1"
ARCH="amd64"
TARBALL="tailscale_${VERSION}_${ARCH}.tgz"
URL="https://pkgs.tailscale.com/stable/${TARBALL}"

cd /tmp
echo "[install_tailscale] downloading $URL"
curl -fL -o "$TARBALL" "$URL"
tar -xzf "$TARBALL"
mv "tailscale_${VERSION}_${ARCH}/tailscale" "$INSTALL_DIR/"
mv "tailscale_${VERSION}_${ARCH}/tailscaled" "$INSTALL_DIR/"
rm -rf "tailscale_${VERSION}_${ARCH}" "$TARBALL"

chmod +x "$INSTALL_DIR/tailscale" "$INSTALL_DIR/tailscaled"

echo "[install_tailscale] installed to $INSTALL_DIR/{tailscale,tailscaled}"
"$INSTALL_DIR/tailscale" --version

echo
echo "下一步："
echo "  1. 把 \$HOME/.local/bin 加到 PATH（或加到 ~/.bashrc）："
echo "       export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  2. 在 GPU 节点上跑 scripts/establish_server.sh"
echo "     它会自动起 tailscaled + tailscale up（前提是 ~/.tailscale-bfaq-authkey 已就位）"
