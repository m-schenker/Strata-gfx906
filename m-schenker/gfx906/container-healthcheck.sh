#!/bin/sh
# Container health for the gfx906 image: GET /v1/models with the API key.  Healthy only when the server answers and has
# a model: while the expert cache fills (minutes on an MI50 with every expert in VRAM) the container is "starting".
#
# The key comes from $STRATA_API_KEY / $API_KEY, or from the key file the entrypoint wrote into the volume.
set -eu
APP=${STRATA_APP:-/opt/strata}
KEY_FILE=${STRATA_KEY_FILE:-/data/state/api-key}
KEY=${STRATA_API_KEY:-${API_KEY:-}}
if [ -z "$KEY" ] && [ -r "$KEY_FILE" ]; then
    KEY=$(cat "$KEY_FILE")
fi
exec "$APP/venv/bin/python3" - "http://127.0.0.1:${PORT:-8080}/v1/models" "$KEY" <<'PY'
import json
import sys
import urllib.request

url, key = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key} if key else {})
try:
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.load(r)
        sys.exit(0 if r.status == 200 and data.get("data") else 1)
except Exception as e:                                   # noqa: BLE001 - any failure is "not healthy"
    print("not healthy: %s" % e, file=sys.stderr)
    sys.exit(1)
PY
