import subprocess,json,os,select,time,uuid,sys
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
reqid=1
def send(o):
 p.stdin.write(json.dumps(o,separators=(',',':'))+'\n'); p.stdin.flush()
def accept_elicitation(o):
 if o.get('method')!='elicitation/create' or 'id' not in o: return False
 params=o.get('params') or {}; blob=json.dumps(params).lower()
 print('ELICIT '+json.dumps(params),flush=True)
 ok=('chatgpt.com' in blob and ('browser' in blob or 'access' in blob))
 result={'action':'accept','content':{}} if ok else {'action':'decline','content':{}}
 send({'jsonrpc':'2.0','id':o['id'],'result':result})
 print('ELICIT_RESPONSE '+result['action'],flush=True)
 return True
def wait(i,timeout=120):
 end=time.time()+timeout
 while time.time()<end:
  rr,_,_=select.select([p.stdout,p.stderr],[],[],0.5)
  for f in rr:
   line=f.readline()
   if not line: continue
   if f is p.stderr:
    print('ERR '+line.rstrip(),flush=True); continue
   try:o=json.loads(line)
   except:
    print('RAW '+line.rstrip(),flush=True); continue
   if accept_elicitation(o): continue
   if o.get('id')==i:return o
   print('EVENT '+json.dumps(o),flush=True)
 raise TimeoutError(i)
send({'jsonrpc':'2.0','id':reqid,'method':'initialize','params':{
 'protocolVersion':'2025-06-18','capabilities':{'elicitation':{}},
 'clientInfo':{'name':'chatty-browser-waker','version':'0.5'}}})
r=wait(reqid); print('READY '+json.dumps(r)[:500],flush=True); reqid+=1
send({'jsonrpc':'2.0','method':'notifications/initialized','params':{}})
for raw in sys.stdin:
 code=raw.rstrip('\n')
 if not code: continue
 meta=json.dumps({'session_id':'6ab7c0e6-b350-83eb-a1af-44b018656688','turn_id':str(uuid.uuid4()),'model':'gpt-5.6-sol','call_id':str(uuid.uuid4())})
 i=reqid; reqid+=1
 send({'jsonrpc':'2.0','id':i,'method':'tools/call','params':{
  'name':'js','arguments':{'code':code,'title':'Chatty same-thread wake'},
  '_meta':{'x-codex-turn-metadata':meta}}})
 try:
  o=wait(i,120)
  content=o.get('result',{}).get('content',[])
  useful=[]
  for c in content:
   if c.get('type')!='text': continue
   t=c.get('text','')
   if t.startswith('## Computer Use'): continue
   useful.append(t)
  print('CALL_RESULT '+json.dumps({'isError':o.get('result',{}).get('isError'),'text':useful}),flush=True)
 except Exception as e:
  print('CALL_ERROR '+repr(e),flush=True)
