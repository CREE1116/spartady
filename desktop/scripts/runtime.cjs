const fs=require('node:fs'),path=require('node:path');
fs.writeFileSync(path.join(__dirname,'../electron/runtime.json'),JSON.stringify({projectRoot:path.resolve(__dirname,'../..')},null,2));
