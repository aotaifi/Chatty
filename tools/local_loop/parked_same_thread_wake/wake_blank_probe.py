#!/usr/bin/env python3
import json,os,select,subprocess,time,uuid
base='/Applications/ChatGPT.app/Contents/Resources/cua_node'
env=os.environ.copy();env.update({
'NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS':'1000','NODE_REPL_NODE_MODULE_DIRS':base+'/lib/node_modules',
'NODE_REPL_NODE_PATH':base+'/bin/node','NODE_REPL_TRUSTED_CODE_PATHS':os.path.expanduser('~/.codex')+':'+base+'/lib/node_modules',
'CODEX_HOME':os.path.expanduser('~/.codex'),'BROWSER_USE_AVAILABLE_BACKENDS':'chrome,iab','BROWSER_USE_TINYSKY_ENABLED':'1',
'NODE_REPL_INSTRUCTIONS_USE_CASE_BROWSER':'Control browser.','NODE_REPL_INSTRUCTIONS_USE_CASE_CHROME':'Control Chrome.',
'NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE':'Control desktop apps on macOS through Computer Use.',
'BROWSER_USE_CODEX_APP_BUILD_FLAVOR':'prod','BROWSER_USE_CODEX_APP_VERSION':'26.917.71314',
'NODE_REPL_TRUSTED_SERVICES':'{"browser":"@oai/browser-desktop/service","sky":"@oai/sky/service"}',
'SKY_CUA_SERVICE_PATH':os.path.expanduser('~/.codex/computer-use/Codex Computer Use.app'),
'CODEX_CLI_PATH':'/Applications/ChatGPT.app/Contents/Resources/codex','CUA_REPL_NODE_REPL_PATH':base+'/bin/node_repl',
'CUA_REPL_ENABLED_SURFACES':'browser,computer'})
p=subprocess.Popen([base+'/bin/node',base+'/lib/node_modules/@oai/cua-repl/bin/cua-repl.mjs'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=env)
rid=0
def send(o):p.stdin.write(json.dumps(o,separators=(',',':'))+'\n');p.stdin.flush()
def wait(i,t=35):
 end=time.time()+t
 while time.time()<end:
  rr,_,_=select.select([p.stdout,p.stderr],[],[],0.5)
  for f in rr:
   l=f.readline()
   if not l:continue
   if f is p.stderr:print('ERR',l.rstrip(),flush=True);continue
   try:o=json.loads(l)
   except:continue
   if o.get('method')=='elicitation/create' and 'id' in o:
    m=o.get('params',{}).get('_meta',{});ok=m.get('origin')=='https://chatgpt.com'
    send({'jsonrpc':'2.0','id':o['id'],'result':{'action':'accept' if ok else 'cancel','content':{}}});continue
   if o.get('id')==i:return o
 raise TimeoutError(i)
def rpc(method,params=None):
 global rid;rid+=1;o={'jsonrpc':'2.0','id':rid,'method':method}
 if params is not None:o['params']=params
 send(o);return wait(rid)
def js(c,t,tm=25000):
 m=json.dumps({'session_id':'6ab7c0e6-b350-83eb-a1af-44b018656688','turn_id':str(uuid.uuid4()),'model':'gpt-5.6-sol'})
 st=time.time();r=rpc('tools/call',{'name':'js','arguments':{'code':c,'title':t,'timeout_ms':tm},'_meta':{'x-codex-turn-metadata':m}})
 print('STEP',t,'elapsed',time.time()-st,'error',r.get('result',{}).get('isError'),flush=True)
 texts=[x.get('text','') for x in r.get('result',{}).get('content',[]) if x.get('type')=='text']
 print('\n'.join(texts)[-5000:],flush=True);return r
print(rpc('initialize',{'protocolVersion':'2025-06-18','capabilities':{'elicitation':{}},'clientInfo':{'name':'blank-probe','version':'1'}}),flush=True)
send({'jsonrpc':'2.0','method':'notifications/initialized','params':{}})
js("let bs=await cua.listBrowsers({emit:false}); nodeRepl.write('B '+JSON.stringify(bs));",'browsers')
js("globalThis.t=await cua.createBrowserTab('1','about:blank',{sessionName:'Chatty wake blank'}); nodeRepl.write('CREATED '+globalThis.t.id);",'create blank')
js("await globalThis.t.goto('https://chatgpt.com/c/6ab8b825-2e54-83eb-89b4-707520bbbd36'); nodeRepl.write('GOTO_OK');",'goto target')
js("let s=await globalThis.t.getAXState({emit:false,disableDiffing:true}); let m=s.match(/(\\d+)\\s+text entry area \\(settable\\)/); nodeRepl.write('STATE '+JSON.stringify({hasComposer:!!m,stop:/\\bbutton (?:Stoppen|Stop)\\b/.test(s),len:s.length}));",'state')
p.terminate()
