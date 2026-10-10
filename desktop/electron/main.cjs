const {app,BrowserWindow,ipcMain,dialog,session,safeStorage}=require('electron');
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');
const readline=require('node:readline');
const crypto=require('node:crypto');
const runtime=fs.existsSync(path.join(__dirname,'runtime.json'))?JSON.parse(fs.readFileSync(path.join(__dirname,'runtime.json'),'utf8')):{};
const root=process.env.SPARTADY_PROJECT_ROOT||path.resolve(__dirname,'../..');
const resourceDir=app.isPackaged?path.join(process.resourcesPath,'runtime'):path.join(root,'desktop/runtime');
if(process.env.SPARTADY_DATA_DIR)app.setPath('userData',path.resolve(process.env.SPARTADY_DATA_DIR));
let window,worker;let next=0;const pending=new Map();let quitting=false;
const methods=new Set(['bootstrap','settings','save_exam','delete_exam','budget','plan','note','notebook','save_notebook','transcribe','tutor','quiz','grade','enrich','capture_records','install_model','setup_asr','learning_session','sessions']);
function trusted(event){if(!window||event.sender!==window.webContents||event.senderFrame!==window.webContents.mainFrame)throw Error('허용되지 않은 창입니다.');}
function request(method,args){return new Promise((resolve,reject)=>{
  if(!worker||worker.exitCode!==null)return reject(Error('학습 엔진이 종료되었습니다. 앱을 다시 열어주세요.'));
  const id=++next;const timer=setTimeout(()=>{pending.delete(id);reject(Error('처리 시간이 초과되었습니다. 진행 중인 작업은 보존됩니다.'));},60*60*1000);
  pending.set(id,{resolve,reject,timer});worker.stdin.write(JSON.stringify({id,method,args})+'\n');
});}
function startWorker(){
  const bundled=path.join(resourceDir,'backend',process.platform==='win32'?'spartady-engine.exe':'spartady-engine');
  const python=process.env.SPARTADY_PYTHON||path.join(root,process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
  const executable=fs.existsSync(bundled)?bundled:python;
  if(!fs.existsSync(executable))throw Error('학습 엔진을 찾지 못했습니다. 앱을 다시 설치해주세요.');
  let key=process.env.OPENAI_API_KEY;const keyFile=path.join(app.getPath('userData'),'api-key.enc');
  if(fs.existsSync(keyFile)&&safeStorage.isEncryptionAvailable())key=safeStorage.decryptString(fs.readFileSync(keyFile));
  worker=spawn(executable,fs.existsSync(bundled)?['--data',app.getPath('userData')]:[path.join(root,'desktop/backend/server.py'),'--data',app.getPath('userData')],{cwd:app.getPath('userData'),detached:process.platform!=='win32',stdio:['pipe','pipe','pipe'],env:{...process.env,PYTHONUNBUFFERED:'1',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8',SPARTADY_RESOURCE_DIR:resourceDir,...(key?{OPENAI_API_KEY:key}:{})}});
  readline.createInterface({input:worker.stdout}).on('line',line=>{try{const r=JSON.parse(line);if(r.event){window?.webContents.send('progress',r);return;}const p=pending.get(r.id);if(p){clearTimeout(p.timer);pending.delete(r.id);r.error?p.reject(Error(r.error)):p.resolve(r.result);}}catch{console.error('Unexpected engine output');}});
  worker.stderr.on('data',data=>{fs.appendFileSync(path.join(app.getPath('userData'),'engine.log'),data);});
  const failed=()=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(Error('학습 엔진 연결이 끊겼습니다. 앱을 다시 열어주세요.'));}pending.clear();};
  worker.on('error',failed);worker.on('exit',failed);
}
app.whenReady().then(()=>{
  fs.mkdirSync(app.getPath('userData'),{recursive:true});startWorker();
  window=new BrowserWindow({width:1460,height:960,minWidth:1000,minHeight:700,title:'SPARTADY',backgroundColor:'#111111',titleBarStyle:process.platform==='darwin'?'hiddenInset':'default',trafficLightPosition:{x:18,y:20},webPreferences:{preload:path.join(__dirname,'preload.cjs'),nodeIntegration:false,contextIsolation:true,sandbox:true}});
  window.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  window.webContents.on('will-navigate',(event)=>event.preventDefault());
  session.defaultSession.setPermissionRequestHandler((contents,permission,callback,details)=>callback(contents===window.webContents&&permission==='media'&&details.mediaTypes?.every(t=>t==='audio')));
  session.defaultSession.setPermissionCheckHandler((contents,permission,origin,details)=>contents===window.webContents&&permission==='media'&&details.mediaType==='audio');
  ipcMain.handle('engine',(event,method,args)=>{trusted(event);if(!methods.has(method))throw Error('지원하지 않는 요청입니다.');return request(method,args);});
  ipcMain.handle('save-api-key',async(event,key)=>{trusted(event);if(typeof key!=='string'||key.length>1000)throw Error('유효한 API 키를 입력해주세요.');if(!safeStorage.isEncryptionAvailable())throw Error('이 기기의 안전한 자격증명 저장소를 사용할 수 없습니다.');fs.writeFileSync(path.join(app.getPath('userData'),'api-key.enc'),safeStorage.encryptString(key),{mode:0o600});await request('set_api_key',{key});return {saved:true};});
  ipcMain.handle('import-pdf' ,async event=>{trusted(event);const {canceled,filePaths}=await dialog.showOpenDialog(window,{title:'교안 추가',properties:['openFile','multiSelections'],filters:[{name:'PDF 교안',extensions:['pdf']}]});if(canceled)return null;return request('import',{paths:filePaths});});
  ipcMain.handle('read-pdf',async(event,id)=>{trusted(event);const result=await request('pdf_path',{id});return new Uint8Array(await fs.promises.readFile(result.path));});
  ipcMain.handle('save-audio',async(event,bytes)=>{trusted(event);const buffer=Buffer.from(bytes);if(buffer.length>16*1024*1024||buffer.toString('ascii',0,4)!=='RIFF'||buffer.toString('ascii',8,12)!=='WAVE')throw Error('유효한 WAV 녹음이 아닙니다.');const id=crypto.randomUUID()+'.wav';const folder=path.join(app.getPath('userData'),'audio');await fs.promises.mkdir(folder,{recursive:true});await fs.promises.writeFile(path.join(folder,id),buffer);return id;});
  ipcMain.handle('read-audio',async(event,id)=>{trusted(event);if(typeof id!=='string'||!/^[-\w]+\.wav$/.test(id))throw Error('유효한 녹음이 아닙니다.');return new Uint8Array(await fs.promises.readFile(path.join(app.getPath('userData'),'audio',id)));});
  window.loadFile(path.join(__dirname,'../dist/index.html'));
  if(process.env.SPARTADY_TEST)window.webContents.on('did-finish-load',()=>console.log('SPARTADY_READY'));
}).catch(error=>{dialog.showErrorBox('SPARTADY를 열 수 없습니다',error.message);app.quit();});
function stopWorker(){try{if(process.platform!=='win32')process.kill(-worker.pid,'SIGTERM');else worker.kill();}catch{}}
app.on('before-quit',event=>{if(worker&&!quitting){event.preventDefault();quitting=true;request('shutdown',{}).finally(()=>{stopWorker();app.quit();});setTimeout(()=>{stopWorker();app.quit();},6000).unref();}});
app.on('window-all-closed',()=>app.quit());
