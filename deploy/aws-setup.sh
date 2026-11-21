#!/bin/bash
# First-boot setup of the detector's web page (app.py) on an Amazon EC2 instance with Ubuntu Server 24.04.
# Paste this file as "User data" when launching the instance; it runs once, as root. Log: /var/log/detector-setup.log
#
# Result: app.py runs as a service of an unprivileged user on 127.0.0.1:7860 (not reachable from outside).
# Caddy answers on ports 80 and 443 and serves the page over HTTPS at https://<public-ip-with-dashes>.sslip.io
# (sslip.io: a free DNS name that points to the IP address; Let's Encrypt certificate obtained by Caddy), with security
# headers and without a Server header. app.py analyses at most 20 images per visitor (IP address) in 10 minutes and
# shows the detector's answer together with a separate check by the published Ateeqq detector (see "Checker" below).
# The hostname is re-derived from the instance's public IPv4 address at every boot, so after an Elastic IP is
# associated a reboot is enough. No SSH: shell access through AWS Systems Manager Session Manager.
set -euxo pipefail
exec > >(tee -a /var/log/detector-setup.log) 2>&1

REPO=https://github.com/mkk1135/graduation-project.git
# The one commit that is installed (full SHA). Update it when redeploying a newer version of the code.
COMMIT=15468338d47dec4484356b453a9beb81df3cb0c8
APP=/opt/detector    # code and Python environment, owned by root: the service cannot change them
DATA=/var/lib/detector  # the service user's only writable place (model downloads)

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-venv git caddy

useradd --system --home-dir "$DATA" --create-home --shell /usr/sbin/nologin detector
git init -q "$APP"   # only commit $COMMIT, shallow and detached; the installed commit is written to the log:
git -C "$APP" remote add origin "$REPO"
git -C "$APP" fetch --depth 1 origin "$COMMIT"
git -C "$APP" checkout --detach "$COMMIT"
git -C "$APP" log -1 --format='installed commit %H %s'
python3 -m venv "$APP/.venv"
# The versions with which the results were measured (requirements.lock.txt), CPU build of PyTorch.
# torchvision 0.29.2 has no CPU build for Python 3.12; 0.29.1 is used (only for resizing and cropping in commfor.py).
"$APP/.venv/bin/pip" install --no-cache-dir torch==2.15.0 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
"$APP/.venv/bin/pip" install --no-cache-dir transformers==5.23.0 gradio==6.32.0 timm==1.0.31 huggingface_hub==2.4.0 \
    safetensors==0.8.0 pillow==13.0.0 numpy==2.5.4

cat > /etc/systemd/system/detector.service <<'EOF'
[Unit]
Description=AI-generated image detector (app.py of github.com/mkk1135/graduation-project)
After=network-online.target
Wants=network-online.target

[Service]
User=detector
WorkingDirectory=/var/lib/detector
Environment=DETECTOR_PUBLIC=1 HF_HOME=/var/lib/detector/hf GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7860
ExecStart=/opt/detector/.venv/bin/python /opt/detector/app.py
Restart=on-failure
RestartSec=10
# Peak measured for a 2048 x 2048 image: about 2.1 GB. Above this limit the service is restarted, not the machine frozen.
MemoryMax=3G
NoNewPrivileges=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/detector

[Install]
WantedBy=multi-user.target
EOF

# Checker (second_opinion.py): the published Ateeqq detector, run on this server at the same time as the detector.
# The page's answer is the detector's answer alone; below it, as a separate line, the page shows the checker's score
# for "AI-generated" in percent ("Check by the published Ateeqq detector: 97% AI-generated"), which never changes the
# answer. If the check fails, the answer is shown with a note that the check was unavailable. The variable names
# SECOND_OPINION_* and the file second-opinion.conf are kept from the earlier second opinion. second_opinion.py imports
# the anthropic package in any case. Two Claude variants are prepared as checker; they give a class and a reason, not a
# number (swap the Environment lines below, then systemctl daemon-reload and restart the service):
#  - Claude API: key in /etc/detector/anthropic.env as ANTHROPIC_API_KEY=... (root only, mode 600; systemd passes it
#    to the service), plus EnvironmentFile=/etc/detector/anthropic.env and SECOND_OPINION_MODEL=claude-opus-5-5;
#  - Amazon Bedrock: once AWS has enabled Bedrock for the account and the instance role may call the model.
"$APP/.venv/bin/pip" install --no-cache-dir "anthropic[bedrock]==2.12.0"
mkdir -p /etc/systemd/system/detector.service.d
cat > /etc/systemd/system/detector.service.d/second-opinion.conf <<'EOF'
[Service]
# Environment=AWS_REGION=us-east-1 SECOND_OPINION_MODEL=anthropic.claude-opus-5-5 "SECOND_OPINION_NAME=Claude Opus 5.5"
Environment=AWS_REGION=us-east-1 SECOND_OPINION_MODEL=Ateeqq/ai-vs-human-image-detector "SECOND_OPINION_NAME=the published Ateeqq detector"
EOF

cat > /usr/local/bin/detector-caddyfile <<'EOF'
#!/bin/bash
# Writes /etc/caddy/Caddyfile for this instance's current public IPv4 address (IMDSv2).
set -euo pipefail
token=$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
ip=$(curl -sf -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/public-ipv4)
# Security headers on the responses of the page. The CSP restricts only framing, plugins and <base>: Gradio needs inline
# scripts and styles. The Server header (Caddy's and uvicorn's) is removed; deleting a field makes the block deferred.
cat > /etc/caddy/Caddyfile <<CADDY
${ip//./-}.sslip.io {
	request_body {
		max_size 25MB
	}
	header {
		Strict-Transport-Security "max-age=31536000"
		X-Content-Type-Options "nosniff"
		X-Frame-Options "DENY"
		Referrer-Policy "no-referrer"
		Permissions-Policy "camera=(), microphone=(), geolocation=()"
		Content-Security-Policy "frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
		-Server
		defer
	}
	reverse_proxy 127.0.0.1:7860 {
		header_down -Server
	}
}
CADDY
EOF
chmod 755 /usr/local/bin/detector-caddyfile

cat > /etc/systemd/system/detector-caddyfile.service <<'EOF'
[Unit]
Description=Point Caddy at this instance's public IPv4 address
Wants=network-online.target
After=network-online.target
Before=caddy.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/detector-caddyfile

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now detector.service
systemctl enable detector-caddyfile.service
systemctl start detector-caddyfile.service
systemctl restart caddy
echo "setup finished"
