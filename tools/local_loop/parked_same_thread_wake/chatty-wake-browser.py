#!/usr/bin/env python3
"""Wake an existing ChatGPT web conversation from a long-running Mac job.

Uses the ChatGPT.app-bundled browser CUA only for short UI operations.
Chrome itself opens the target URL; we bind an existing exact-match tab rather
than calling createBrowserTab(), which is unreliable with the extension backend.
"""
from __future__ import annotations
import hashlib, json, os, plistlib, re, select, subprocess, sys, time, uuid, urllib.parse
from pathlib import Path

APP=Path("/Applications/ChatGPT.app")
RES=APP/"Contents/Resources"
BASE=RES/"cua_node"
NODE=BASE/"bin/node"
SERVER=BASE/"lib/node_modules/@oai/cua-repl/bin/cua-repl.mjs"

if len(sys.argv)<3:
    raise SystemExit("usage: chatty-wake-browser.py THREAD_ID PROMPT...")
thread_id=sys.argv[1].strip()
prompt=" ".join(sys.argv[2:]).strip()
if not thread_id or not prompt:
    raise SystemExit("thread id and prompt must be non-empty")
target_url="https://chatgpt.com/c/"+thread_id
m=re.search(r"\[CHATTY_JOB_DONE:[^\]]+\]",prompt)
wake_token=m.group(0) if m else "[CHATTY_WAKE:"+hashlib.sha256(prompt.encode()).hexdigest()[:16]+"]"
if wake_token not in prompt:
    prompt=wake_token+"\n"+prompt

try:
    with (APP/"Contents/Info.plist").open("rb") as f:
        app_version=plistlib.load(f).get("CFBundleShortVersionString","")
except Exception:
    app_version=""

def build_env():
    e=os.environ.copy()
    e.update({
      "NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS":"1000",
      "NODE_REPL_NODE_MODULE_DIRS":str(BASE/"lib/node_modules"),
      "NODE_REPL_NODE_PATH":str(NODE),
      "NODE_REPL_TRUSTED_CODE_PATHS":os.path.expanduser("~/.codex")+":"+str(BASE/"lib/node_modules"),
      "CODEX_HOME":os.path.expanduser("~/.codex"),
      "BROWSER_USE_AVAILABLE_BACKENDS":"chrome,iab",
      "BROWSER_USE_TINYSKY_ENABLED":"1",
      "NODE_REPL_INSTRUCTIONS_USE_CASE_BROWSER":"Control the browser.",
      "NODE_REPL_INSTRUCTIONS_USE_CASE_CHROME":"Control Chrome.",
      "BROWSER_USE_CODEX_APP_BUILD_FLAVOR":"prod",
      "BROWSER_USE_CODEX_APP_VERSION":app_version,
      "NODE_REPL_TRUSTED_SERVICES":'{"browser":"@oai/browser-desktop/service","sky":"@oai/sky/service"}',
      "SKY_CUA_SERVICE_PATH":os.path.expanduser("~/.codex/computer-use/Codex Computer Use.app"),
      "CODEX_CLI_PATH":str(RES/"codex"),
      "CUA_REPL_NODE_REPL_PATH":str(BASE/"bin/node_repl",
      ),
      "CUA_REPL_ENABLED_SURFACES":"browser",
    })
    return e

class CUA:
    def __init__(self):
        self.p=subprocess.Popen([str(NODE),str(SERVER)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=build_env())
        self.rid=0
        r=self.rpc("initialize",{"protocolVersion":"2025-06-18","capabilities":{"elicitation":{}},"clientInfo":{"name":"chatty-waker","version":"3.0"}},30)
        if "error" in r: raise RuntimeError("MCP initialize failed: "+json.dumps(r["error"]))
        self.notify("notifications/initialized",{})
    def raw(self,o):
        assert self.p.stdin
        self.p.stdin.write(json.dumps(o,separators=(",",":"))+"\n"); self.p.stdin.flush()
    def notify(self,m,params=None):
        o={"jsonrpc":"2.0","method":m}
        if params is not None:o["params"]=params
        self.raw(o)
    def wait(self,i,timeout=55):
        assert self.p.stdout and self.p.stderr
        end=time.time()+timeout
        while time.time()<end:
            rr,_,_=select.select([self.p.stdout,self.p.stderr],[],[],.5)
            for f in rr:
                line=f.readline()
                if not line: continue
                if f is self.p.stderr:
                    print("CUA:",line.rstrip(),file=sys.stderr,flush=True); continue
                try:o=json.loads(line)
                except Exception: continue
                if o.get("method")=="elicitation/create" and "id" in o:
                    par=o.get("params") or {}; meta=par.get("_meta") or {}
                    ok=meta.get("tool_name")=="access_browser_origin" and meta.get("origin")=="https://chatgpt.com"
                    self.raw({"jsonrpc":"2.0","id":o["id"],"result":{"action":"accept" if ok else "cancel","content":{}}})
                    continue
                if o.get("id")==i:return o
        raise TimeoutError(f"timeout waiting for MCP id {i}")
    def rpc(self,m,params=None,timeout=55):
        self.rid+=1; o={"jsonrpc":"2.0","id":self.rid,"method":m}
        if params is not None:o["params"]=params
        self.raw(o); return self.wait(self.rid,timeout)
    def js(self,code,title="Chatty wake step",timeout=55):
        meta=json.dumps({"session_id":thread_id,"thread_id":thread_id,"turn_id":"chatty-wake-"+uuid.uuid4().hex,"model":"gpt-5.6-sol","call_id":uuid.uuid4().hex})
        # Keep each call bounded; long retry loops live in Python.
        r=self.rpc("tools/call",{"name":"js","arguments":{"code":code,"title":title,"timeout_ms":max(1000,(timeout-5)*1000)},"_meta":{"x-codex-turn-metadata":meta}},timeout)
        if "error" in r: raise RuntimeError(json.dumps(r["error"]))
        tr=r.get("result") or {}
        texts=[x.get("text","") for x in tr.get("content",[]) if x.get("type")=="text" and not x.get("text","").startswith("## Computer Use")]
        if tr.get("isError"): raise RuntimeError("\n".join(texts) or json.dumps(tr))
        return texts[-1] if texts else ""
    def close(self):
        try:self.p.terminate(); self.p.wait(timeout=3)
        except Exception:
            try:self.p.kill()
            except Exception:pass

def open_target():
    deep="codex://browser?url="+urllib.parse.quote(target_url,safe="")
    subprocess.run(
        ["/usr/bin/open", deep],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )

def composer_index(ax:str):
    xs=re.findall(r"(\d+)\s+text entry area \(settable\)",ax)
    return int(xs[-1]) if xs else None

c=None
try:
    # Make sure Chrome has a normal, authenticated target tab. This operation is
    # fast and independent of the flaky CUA tab-creation path.
    open_target()
    time.sleep(1)
    c=CUA()

    # Discover every available browser provider and bind an exact-match target
    # tab. The public codex://browser route above bootstraps browser discovery;
    # do not assume a particular Chrome profile/extension instance.
    target=None
    browser_id=None
    seen=[]
    for _ in range(20):
        browser_raw=c.js(
            "nodeRepl.write(JSON.stringify(await cua.listBrowsers({emit:false})))",
            "Discover browsers",
        )
        browsers=json.loads(browser_raw)
        seen=browsers
        candidates=[]
        for b in browsers:
            bid=str(b.get("id",""))
            if not bid:
                continue
            try:
                tabs_raw=c.js(
                    "nodeRepl.write(JSON.stringify(await cua.listTabs({browser:"
                    +json.dumps(bid)+",emit:false})))",
                    "Find ChatGPT tab",
                )
                tabs=json.loads(tabs_raw)
            except Exception:
                continue
            for t in tabs:
                if (t.get("url") or "")==target_url:
                    candidates.append((b,t))
        if candidates:
            # Prefer entries with an explicit recent lastOpened timestamp.
            candidates.sort(
                key=lambda bt:(bt[1].get("lastOpened") or "", str(bt[1].get("id",""))),
                reverse=True,
            )
            browser,target=candidates[0]
            browser_id=str(browser["id"])
            print("WAKE_BROWSER",json.dumps(browser,sort_keys=True),flush=True)
            break
        time.sleep(1)
    if target is None or browser_id is None:
        raise RuntimeError("target ChatGPT tab not found; browsers="+json.dumps(seen))
    c.js(
        "globalThis.__wakeTab=await cua.getTab("
        +json.dumps(str(target["id"]))
        +",{browser:"+json.dumps(browser_id)+"});nodeRepl.write('BOUND')",
        "Bind ChatGPT tab",
        60,
    )
    print("WAKE_TAB",json.dumps(target,sort_keys=True),flush=True)

    # The exact thread can have pending-conversation-input while still accepting
    # a new user interruption. Do not wait on that internal flag. Retry fresh AX
    # observations until a composer exists.
    ax=""
    for attempt in range(20):
        ax=c.js("let s=await globalThis.__wakeTab.getAXState({emit:false,disableDiffing:true});nodeRepl.write(s)","Read ChatGPT composer")
        if wake_token in ax:
            print("WAKE_ALREADY_PRESENT",thread_id,wake_token,flush=True)
            raise SystemExit(0)
        if composer_index(ax) is not None: break
        time.sleep(1)
    else:
        raise RuntimeError("ChatGPT composer not found")

    # React may replace the accessibility element between observation and input.
    # Re-resolve on every attempt. First use setValue; if that fails, focus a
    # fresh element and paste to the focused target (null).
    filled=False; fill_errors=[]
    for attempt in range(8):
        try:
            code=(
              "let a=await globalThis.__wakeTab.getAXState({emit:false,disableDiffing:true});"
              "let es=[...a.matchAll(/(\\d+)\\s+text entry area \\(settable\\)/g)];"
              "if(!es.length)throw new Error('no composer');"
              "let i=Number(es[es.length-1][1]);"
              "try{await globalThis.__wakeTab.setValue(i,"+json.dumps(prompt)+");nodeRepl.write('FILL setValue')}catch(e1){"
              " let a2=await globalThis.__wakeTab.getAXState({emit:false,disableDiffing:true});"
              " let es2=[...a2.matchAll(/(\\d+)\\s+text entry area \\(settable\\)/g)];"
              " if(!es2.length)throw e1;"
              " let i2=Number(es2[es2.length-1][1]);"
              " await globalThis.__wakeTab.click(i2);"
              " await globalThis.__wakeTab.paste(null,"+json.dumps(prompt)+",{format:'text'});"
              " nodeRepl.write('FILL click-paste')}"
            )
            method=c.js(code,"Fill ChatGPT composer",55)
            time.sleep(.35)
            ax=c.js("let s=await globalThis.__wakeTab.getAXState({emit:false,disableDiffing:true});nodeRepl.write(s)","Verify filled composer")
            if wake_token in ax:
                filled=True; print("WAKE_FILLED",attempt+1,method,flush=True); break
            fill_errors.append("token absent")
        except Exception as e:
            fill_errors.append(repr(e))
        time.sleep(1)
    if not filled: raise RuntimeError("fill failed: "+" | ".join(fill_errors[-8:]))

    # Submit. Focused Return is least dependent on a volatile AX id. If that
    # fails, refresh the composer id and send Return to the new id.
    sent_key=False; send_errors=[]
    for attempt in range(5):
        try:
            r=c.js("await globalThis.__wakeTab.pressKey(null,'Return');nodeRepl.write('RETURN focused')","Submit wake prompt",55)
            sent_key=True; print("WAKE_RETURN",attempt+1,r,flush=True); break
        except Exception as e:
            send_errors.append(repr(e))
            try:
                r=c.js(
                  "let a=await globalThis.__wakeTab.getAXState({emit:false,disableDiffing:true});"
                  "let es=[...a.matchAll(/(\\d+)\\s+text entry area \\(settable\\)/g)];"
                  "if(!es.length)throw new Error('no composer');"
                  "await globalThis.__wakeTab.pressKey(Number(es[es.length-1][1]),'Return');"
                  "nodeRepl.write('RETURN indexed')",
                  "Submit wake prompt",55)
                sent_key=True; print("WAKE_RETURN",attempt+1,r,flush=True); break
            except Exception as e2: send_errors.append(repr(e2))
        time.sleep(1)
    if not sent_key: raise RuntimeError("submit failed: "+" | ".join(send_errors[-10:]))

    # Verify the token is present after submission. It may already have been
    # visible in the composer, so also require the composer no longer contains
    # the full prompt when possible; most importantly this step catches no-op
    # Return attempts.
    verified=False
    for _ in range(15):
        time.sleep(.8)
        ax=c.js("let s=await globalThis.__wakeTab.getAXState({emit:false,disableDiffing:true});nodeRepl.write(s)","Verify wake submission")
        if wake_token in ax:
            verified=True; break
    if not verified: raise RuntimeError("wake token not visible after submit")
    print("WAKE_SUBMITTED",thread_id,wake_token,flush=True)
finally:
    if c is not None:c.close()
