#!/bin/bash
set -u
TO="a.otaifi@lmu.de"
STATUS="$1"
JOB="$2"
DETAILS="${3:-}"
SUBJECT="[Chatty] ${STATUS} ${JOB}"
BODY="${STATUS} ${JOB}"
[ -n "$DETAILS" ] && BODY="$BODY — $DETAILS"

# Keep the concise LMU email notification.
printf '%s\n' "$BODY" | /usr/bin/mail -s "$SUBJECT" "$TO"

# Also post to Slack when the protected webhook file exists.
HOOK_FILE="$HOME/.config/chatty/slack_webhook"
if [ -s "$HOOK_FILE" ]; then
  PAYLOAD=$(python3 -c 'import json,sys; print(json.dumps({"text":"[Chatty] "+sys.argv[1]}))' "$BODY")
  curl -fsS -X POST -H 'Content-type: application/json' \
    --data "$PAYLOAD" "$(cat "$HOOK_FILE")" >/dev/null || true
fi
