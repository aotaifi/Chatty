from pathlib import Path
import subprocess, json, time, os

base = Path.home() / "Chatty/tools/local_loop"
inbox = base / "agent_inbox.txt"
outbox = base / "agent_outbox.txt"
seen = None

env = os.environ.copy()
keyfile = base / "keysetup/openai_api_key"
env["OPENAI_API_KEY"] = keyfile.read_text().strip()
env.update({
    "CI":"1",
    "NO_COLOR":"1",
    "OPENCODE_PURE":"1",
    "OPENCODE_FAST_BOOT":"1",
    "OPENCODE_DISABLE_MODELS_FETCH":"1",
    "OPENCODE_DISABLE_PROJECT_CONFIG":"1",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS":"1",
    "OPENCODE_DISABLE_EXTERNAL_SKILLS":"1",
    "OPENCODE_DISABLE_FFF":"1",
    "OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER":"1",
    "OPENCODE_CONFIG_CONTENT":"{}",
})

work = Path("/tmp/opencode-agentb-loop")
work.mkdir(exist_ok=True)

while True:
    try:
        if inbox.exists():
            msg = inbox.read_text(errors="ignore").strip()
            if msg and msg != seen:
                seen = msg
                cmd = [
                    "opencode","run","--format","json","--title","agent-b-loop",
                    "-m","openai/gpt-5.6-sol", msg
                ]
                p = subprocess.run(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True, timeout=120)
                reply = None
                for line in p.stdout.splitlines():
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "text":
                            reply = obj.get("part",{}).get("text")
                    except Exception:
                        pass
                if reply is None:
                    reply = f"[Agent B error exit={p.returncode}] {p.stderr[-1000:]}"
                outbox.write_text(reply.strip() + "\n")
        time.sleep(0.25)
    except Exception as e:
        outbox.write_text(f"[Agent B loop error] {e}\n")
        time.sleep(1)
