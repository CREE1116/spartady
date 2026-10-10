const fs=require('node:fs'),{Resvg}=require('@resvg/resvg-js'),{execFileSync}=require('node:child_process');
const graphic=fs.readFileSync('src/assets/spartan.svg','utf8').replace(/<title[^>]*>.*?<\/title>/,'');
const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 256 256"><rect x="5" y="5" width="246" height="246" rx="53" fill="#171717"/><svg x="32" y="23" width="192" height="204" viewBox="0 0 256 256">${graphic.replace(/<svg[^>]*>|<\/svg>/g,'')}</svg></svg>`;
fs.writeFileSync('resources/app-icon.svg',svg);fs.writeFileSync('resources/icon.png',new Resvg(svg).render().asPng());fs.mkdirSync('resources/icon.iconset',{recursive:true});
for(const size of [16,32,128,256,512])for(const scale of [1,2])execFileSync('sips',['-z',String(size*scale),String(size*scale),'resources/icon.png','--out',`resources/icon.iconset/icon_${size}x${size}${scale===2?'@2x':''}.png`],{stdio:'ignore'});
execFileSync('iconutil',['-c','icns','resources/icon.iconset','-o','resources/icon.icns']);
