const {contextBridge,ipcRenderer}=require('electron');
contextBridge.exposeInMainWorld('spartady',Object.freeze({
  call:(method,args={})=>ipcRenderer.invoke('engine',method,args),
  saveAPIKey:(key)=>ipcRenderer.invoke('save-api-key',key),
  importPDF:()=>ipcRenderer.invoke('import-pdf'),
  pdf:(id)=>ipcRenderer.invoke('read-pdf',id),
  saveAudio:(bytes)=>ipcRenderer.invoke('save-audio',bytes),
  audio:(id)=>ipcRenderer.invoke('read-audio',id),
  onProgress:(callback)=>{const handler=(_e,data)=>callback(data);ipcRenderer.on('progress',handler);return()=>ipcRenderer.removeListener('progress',handler);}
}));
