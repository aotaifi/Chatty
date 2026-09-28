import subprocess,json,os,select,sys,time
base='/Applications/ChatGPT.app/Contents/Resources/cua_node'
env=os.environ.copy(); env.update({
'NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS':'1000',
'NODE_REPL_NODE_MODULE_DIRS':base+'/lib/node_modules',
'NODE_REPL_NODE_PATH':base+'/bin/node',
'NODE_REPL_TRUSTED_CODE_PATHS':os.path.expanduser('~/.codex')+':'+base+'/lib/node_modules',
'CODEX_HOME':os.path.expanduser('~/.codex'),
'BROWSER_USE_AVAILABLE_BACKENDS':'chrome,iab','BROWSER_USE_TINYSKY_ENABLED':'1',
'NODE_REPL_INSTRUCTIONS_USE_CASE_BROWSER':'Control the in-app browser in conjunction with the Browser Plugin.',
'NODE_REPL_INSTRUCTIONS_USE_CASE_CHROME':'Control the Chrome browser in conjunction with the Chrome Plugin. Prefer this method of controlling Chrome over alternatives (such as Computer Use) unless the user explicitly mentions an alternative.',
'NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE':'Control desktop apps on macOS through Computer Use.',
'BROWSER_USE_CODEX_APP_BUILD_FLAVOR':'prod','BROWSER_USE_CODEX_APP_VERSION':'26.917.71314',
'NODE_REPL_TRUSTED_SERVICES':'{"browser":"@oai/browser-desktop/service","sky":"@oai/sky/service"}',
'SKY_CUA_SERVICE_PATH':os.path.expanduser('~/.codex/computer-use/Codex Computer Use.app'),
'CODEX_CLI_PATH':'/Applications/ChatGPT.app/Contents/Resources/codex',
'CUA_REPL_NODE_REPL_PATH':base+'/bin/node_repl','CUA_REPL_ENABLED_SURFACES':'browser,computer'})
cmd=[base+'/bin/node',base+'/lib/node_modules/@oai/cua-repl/bin/cua-repl.mjs']
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=env)
next_id=1
def send(method,params=None):
 global next_id
 i=next_id; next_id+=1
 m={'jsonrpc':'2.0','id':i,'method':method}
 if params is not None:m['params']=params
 p.stdin.write(json.dumps(m,separators=(',',':'))+'\n');p.stdin.flush()
 return i
def notify(method,params=None):
 m={'jsonrpc':'2.0','method':method}
 if params is not None:m['params']=params
 p.stdin.write(json.dumps(m,separators=(',',':'))+'\n');p.stdin.flush()
def wait_id(i,timeout=30):
 end=time.time()+timeout
 while time.time()<end:
  rr,_,_=select.select([p.stdout,p.stderr],[],[],0.5)
  for f in rr:
   line=f.readline()
   if not line: continue
   if f is p.stderr:
    print('ERR',line.rstrip(),flush=True); continue
   try:o=json.loads(line)
   except:
    print('RAW',line.rstrip(),flush=True); continue
   if o.get('id')==i:return o
   print('EVENT',json.dumps(o)[:2000],flush=True)
 raise TimeoutError(i)
i=send('initialize',{'protocolVersion':'2025-06-18','capabilities':{},'clientInfo':{'name':'chatty-waker','version':'0.1'}})
print('INIT',json.dumps(wait_id(i))[:3000],flush=True)
notify('notifications/initialized',{})
i=send('tools/call',{'name':'js','arguments':{'code':'let app = await cua.getApp("ChatGPT")','title':'Bind ChatGPT app'}})
print('FIRST',json.dumps(wait_id(i,60))[:20000],flush=True)
print('READY',flush=True)
for line in sys.stdin:
 code=line.rstrip('\n')
 if not code: continue
 i=send('tools/call',{'name':'js','arguments':{'code':code,'title':'Continue ChatGPT UI test'}})
 try:r=wait_id(i,60); print('RESULT',json.dumps(r)[:30000],flush=True)
 except Exception as e: print('CALLERR',repr(e),flush=True)
