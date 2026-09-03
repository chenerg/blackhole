---
name: byg-mihomo-proxy
description: >
  Configure and debug the local mihomo (Clash Meta) proxy named byg on
  127.0.0.1:17890 beside the existing mac proxy on 10777. Covers subscription
  overlay, bashrc port switch (proxyon mac|byg), grok.com DNS pinning, HTTP/2
  128KB PROTOCOL_ERROR on large POSTs, and TCP buffer/MSS fixes. Use when the
  user says 白月光, byg, mihomo, mixed-port 17890, proxyon, grok 推理断, or
  runs /byg-mihomo-proxy.
metadata:
  short-description: "byg mihomo proxy (17890) setup and failure log"
---

# byg mihomo proxy (17890)

This machine already has a **mac** proxy on `127.0.0.1:10777` (often outside this PID namespace). **byg** is a second mihomo instance for the 白月光 Clash subscription. Never bind, DNAT, NAT-redirect, or dial-out through 10777 when using byg.

Do **not** write subscription URLs, tokens, node passwords, or `/etc/mihomo/secret` into chat, git, or this skill. Read the URL from `$SUB_URL` or `/etc/mihomo/subscription.url` if present.

## Layout (working)

| Role | Value |
|---|---|
| Binary | `/usr/local/bin/mihomo` (Clash Meta, e.g. v1.19.30) |
| Config | `/etc/mihomo/config.yaml` |
| Start | `/usr/local/bin/mihomo-start {start\|stop\|restart\|status}` |
| Refresh sub | `/usr/local/bin/mihomo-update-sub` |
| Log | `/var/log/mihomo.log` |
| byg mixed-port | `127.0.0.1:17890` (HTTP + SOCKS) |
| Controller | `127.0.0.1:19090` (Bearer from `/etc/mihomo/secret`) |
| Local DNS | `127.0.0.1:15353` |
| mac proxy | `127.0.0.1:10777` — do not occupy |

`bind-address: 127.0.0.1`, `allow-lan: false`, `port: 0` (no extra 7890/7891), `ipv6: false`.

Shell switch in `~/.bashrc`: `proxyon mac` → 10777; `proxyon byg` → 17890; last choice in `~/.proxy_port`. Default is mac.

## Overlay after pulling Clash YAML

Fetch with User-Agent `clash.meta/...` or `clash`. Clash-format sub is SS-only (`type: ss`, `aes-128-gcm`). Then patch:

1. Ports/controller/secret/bind as in the table.
2. `tcp-concurrent: true`, `unified-delay: true`, `keep-alive-idle: 30`, `keep-alive-interval: 15`, `routing-mark: 255`.
3. Group **Grok出口** (select): 新加坡 / 美国 / 日本 / 香港 SS nodes only. Default pin **美国 01** after start. No HTTP outbound to 10777.
4. Prepend rules: `DOMAIN-SUFFIX,grok.com`, `DOMAIN-SUFFIX,x.ai`, `DOMAIN-KEYWORD,cli-chat-proxy`, `DOMAIN,auth.x.ai`, `DOMAIN,api.x.ai` → `Grok出口`.
5. Pin Grok/xAI A records in `/etc/hosts` **and** `dns.hosts` to Cloudflare `104.18.28.234` (grok.com, cli-chat-proxy.grok.com) and `104.18.18.80` (api.x.ai, auth.x.ai). `/etc/gai.conf`: `precedence :ffff:0:0/96  100` so IPv4 wins.
6. `sniffer.enable: false` (override-destination on CONNECT broke large HTTP/2).
7. Do **not** send grok.com DoH through `nameserver-policy: ...#♻️ 自动选择` — it deadlocks.

`mihomo-start` also: `sysctl` TCP buffers 16MB, `tcp_mtu_probing=1`, iptables TCPMSS 1360 **only** for `mark 255`.

## Verify (must not use 10777)

Unset all `*_proxy` env vars. Use `-x http://127.0.0.1:17890` only.

```bash
# DNS (system)
getent ahostsv4 cli-chat-proxy.grok.com api.x.ai
# expect 104.18.28.234 and 104.18.18.80

# Clash match
# log: DomainSuffix(grok.com) using Grok出口[🇺🇸 美国 01]

# Small POST
curl --http2 -x http://127.0.0.1:17890 --data '{"model":"x","input":"a"}' \
  https://cli-chat-proxy.grok.com/v1/responses
# expect HTTP 401

# 256KB HTTP/2 POST — grok inference size class
# expect http=401 AND upload=262168, NOT PROTOCOL_ERROR at 131072
```

`grok -p` via byg: `http_proxy`/`https_proxy` = `http://127.0.0.1:17890`, **unset `ALL_PROXY`**. Success = stdout complete + session `turn_ended outcome=completed`, not merely `attempts: 1`.

Confirm 10777 still listens and still has its own exit IP; byg must not NAT 10777.

## Failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `bind: address already in use` on 10777/7890/9090 | mac/host clash already holds those ports; this PID ns cannot see the owner | Use 17890 / 19090 / 15353. Never steal 10777. |
| iptables REDIRECT 10777→17890 | Hijacks mac | Delete the NAT rule. Do not re-add. |
| `grok -p` fails; log `IPCIDR(31.13.64.0/18)` Facebook | No grok.com domain rule; Ali/223 DNS poisons grok/x.ai to Meta/Twitter IPs (IPv6 `2a03:2880:…face:b00c`) | Domain rules to Grok出口; `/etc/hosts` + gai IPv4; do not use polluted nameserver-policy DoH. |
| GET `/v1/models` 401 but `error sending request` on `/v1/responses` | HTTP/2 large POST dies at **131072 bytes** with `PROTOCOL_ERROR (err 1)` on these SS nodes | Kernel `tcp_rmem`/`tcp_wmem` 16MB + MSS 1360 on mark 255; pin Grok出口 to 美国 01. Re-test 256KB POST. |
| `attempts: 1` treated as success | Request never finished (`waiting_for_model`); outer `timeout 45` killed grok | Require `turn_ended` / full stdout. Agent system prompt is hundreds of KB even for a one-word question. |
| `via-mac` HTTP outbound to 10777 | byg traffic leaked to mac | Remove that proxy. byg = SS nodes only. |
| DoH `context deadline exceeded` | `nameserver-policy '+.grok.com' → dns.cloudflare.com#自动选择` chicken-egg | Hosts pin; China DNS only for `proxy-server-nameserver`. |
| Subscription is SS-only, same host different ports | Cannot switch to VLESS/Reality from this sub | Change region port (美国 01 fastest in test) + TCP buffers. |
| `pkill -f 'mihomo -d /etc/mihomo'` kills the agent shell | Pattern matches the wrapper cmdline | `pgrep -x mihomo` then `kill` by PID. |

## Do not

- NAT or bind 10777 when configuring byg.
- Commit or print `$SUB_URL`, node `password`, or controller secret.
- Enable sniffer `override-destination` on this mixed-port.
- Claim grok works from a short GET or from `attempts: 1` without a completed turn.
