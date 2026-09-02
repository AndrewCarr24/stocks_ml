#!/bin/zsh
# Ship the Mac's live world (data/r5_live) to the champion workflow.
#
# The workflow (.github/workflows/champion.yml) carries the world between
# runs in the Actions cache; this seeds that cache when it is empty (first
# run, or GitHub evicted it). The tarball is AES-256-encrypted with
# data/.r5_store_key (the R5_STORE_KEY repo secret) and uploaded as an
# asset of a *draft* release, which only collaborators can see; the
# workflow imports it into the cache and deletes the release.
#
#   ops/r5_seed.sh            # then run the workflow (or wait for Saturday)
#
# Uses git's credential helper for the GitHub token (GH_TOKEN overrides).
set -euo pipefail
REPO=/Users/andrewcarr/Documents/projects/stocks_ml.nosync
OWNER_REPO=AndrewCarr24/stocks_ml
KEY_FILE=data/.r5_store_key
cd "$REPO"

if [ ! -f "$KEY_FILE" ]; then
  openssl rand -hex 32 > "$KEY_FILE"; chmod 600 "$KEY_FILE"
  echo "new store key in $KEY_FILE: put it in the R5_STORE_KEY repo secret before running the workflow"
fi
TOKEN=${GH_TOKEN:-$(printf 'protocol=https\nhost=github.com\n\n' | git credential fill | sed -n 's/^password=//p')}
api() { curl -fsS -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "$@"; }
API=https://api.github.com/repos/$OWNER_REPO

OUT=${TMPDIR:-/tmp}/r5_live.tar.enc
echo "encrypting data/r5_live ..."
tar -C data -cf - r5_live | openssl enc -aes-256-cbc -pbkdf2 -iter 100000 -salt \
    -pass "file:$KEY_FILE" -out "$OUT"
ls -lh "$OUT"

# one seed at a time: drop any earlier draft
for id in $(api "$API/releases?per_page=100" | python3 -c '
import json, sys
print(" ".join(str(r["id"]) for r in json.load(sys.stdin) if r.get("draft") and r.get("name") == "r5 seed"))'); do
  api -X DELETE "$API/releases/$id"
done

upload_url=$(api -X POST "$API/releases" -d '{"tag_name":"r5-seed","name":"r5 seed","draft":true,
  "body":"Encrypted live world for the champion workflow; deleted by the workflow once imported."}' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["upload_url"].split("{")[0])')
echo "uploading ..."
api -X POST -H "Content-Type: application/octet-stream" --data-binary "@$OUT" \
    "$upload_url?name=r5_live.tar.enc" > /dev/null
rm -f "$OUT"
echo "seed uploaded as a draft release; the next champion run imports it"
