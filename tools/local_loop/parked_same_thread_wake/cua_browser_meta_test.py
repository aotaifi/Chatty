import subprocess,json,os,select,time,uuid
base='/Applications/ChatGPT.app/Contents/Resources/cua_node'
env=os.environ.copy(); env.update({
'NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS':'1000',
'NODE_REPL_NODE_MODULE_DIRS':base+'/lib/node_modules',
'NODE_REPL_NODE_PATH':base+'/bin/node',
'NODE_REPL_TRUSTED_CODE_PATHS':os.path.expanduser('~/.codex')+':'+base+'/lib/node_modules',
'CODEX_HOME':os.path.expanduser('~/.codex'),
'BROWSER_USE_AVAILABLE_BACKENDS':'chrome,iab','BROWSER_USE_TINYSKY_ENABLED':'1',
'NODE_REPL_INSTRUCTIONS_USE_CASE_BROWSER':'Control the in-app browser in conjunction with the Browser Plugin.',
'NODE_REPL_INSTRUCTIONS_USE_CASE_CHROME':'Control Chrome.',
'BROWSER_USE_CODEX_APP_BUILD_FLAVOR':'prod','BROWSER_USE_CODEX_APP_VERSION':'26.917.71314',
'NODE_REPL_TRUSTED_SERVICES':'{"browser":"@oai/browser-desktop/service","sky":"@oai/sky/service"}',
'SKY_CUA_SERVICE_PATH':os.path.expanduser('~/.codex/computer-use/Codex Computer Use.app'),
'CODEX_CLI_PATH':'/Applications/ChatGPT.app/Contents/Resources/codex',
'CUA_REPL_NODE_REPL_PATH':base+'/bin/node_repl','CUA_REPL_ENABLED_SURFACES':'browser'})
p=subprocess.Popen([base+'/bin/node',base+'/lib/node_modules/@oai/cua-repl/bin/cua-repl.mjs'],
 stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=env)
def send(obj):
 p.stdin.write(json.dumps(obj,separators=(',',':'))+'\n'); p.stdin.flush()
def wait(i,timeout=120):
 end=time.time()+timeout; lines=[]
 while time.time()<end:
  rr,_,_=select.select([p.stdout,p.stderr],[],[],0.5)
  for f in rr:
   line=f.readline()
   if not line: continue
   if f is p.stderr: lines.append('ERR '+line.rstrip()); continue
   try:o=json.loads(line)
   except: lines.append('RAW '+line.rstrip()); continue
   if o.get('id')==i:return o,lines
   lines.append('EVENT '+json.dumps(o)[:3000])
 raise TimeoutError(i)
send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-06-18','capabilities':{},'clientInfo':{'name':'chatty-browser-waker','version':'0.2'}}})
r,ls=wait(1); print('INIT',json.dumps(r)[:1200])
send({'jsonrpc':'2.0','method':'notifications/initialized','params':{}})
meta=json.dumps({'session_id':'6ab7c0e6-b350-83eb-a1af-44b018656688','turn_id':str(uuid.uuid4()),'model':'gpt-5.6-sol','call_id':str(uuid.uuid4())})
code='let tab = await cua.createBrowserTab("iab","https://chatgpt.com/c/6ab7c0e6-b350-83eb-a1af-44b018656688",{visible:false}); let s=await tab.getAXState({disableDiffing:true,emit:false}); nodeRepl.write(s);'
send({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'js','arguments':{'code':code,'title':'Same-thread browser probe'},'_meta':{'x-codex-turn-metadata':meta}}})
r,ls=wait(2,120)
for x in ls: print(x)
print('RESULT',json.dumps(r)[:60000])
p.terminate()
