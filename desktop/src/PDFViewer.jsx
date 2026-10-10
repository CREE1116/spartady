import React,{useEffect,useRef,useState} from 'react';
import {getDocument,GlobalWorkerOptions} from 'pdfjs-dist';
import workerURL from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
GlobalWorkerOptions.workerSrc=workerURL;
export default function PDFViewer({document,page,zoom}){
  const canvas=useRef(),host=useRef();const [pdf,setPdf]=useState(null),[error,setError]=useState(''),[width,setWidth]=useState(600),[loading,setLoading]=useState(false);
  useEffect(()=>{const observer=new ResizeObserver(([e])=>setWidth(Math.max(250,e.contentRect.width-64)));observer.observe(host.current);return()=>observer.disconnect();},[]);
  useEffect(()=>{let active=true,task;setPdf(null);setError('');if(!document)return;
    window.spartady.pdf(document.id).then(bytes=>{if(!active)return;task=getDocument({data:new Uint8Array(bytes),isEvalSupported:false});return task.promise;}).then(value=>{if(active&&value)setPdf(value);}).catch(e=>active&&setError(e.message));
    return()=>{active=false;task?.destroy();};},[document?.id]);
  useEffect(()=>{if(!pdf||!canvas.current)return;let active=true,render;setLoading(true);
    pdf.getPage(page).then(p=>{if(!active)return;const base=p.getViewport({scale:1});const viewport=p.getViewport({scale:width/base.width*zoom});const ratio=window.devicePixelRatio||1;
      canvas.current.width=Math.floor(viewport.width*ratio);canvas.current.height=Math.floor(viewport.height*ratio);canvas.current.style.width=viewport.width+'px';canvas.current.style.height=viewport.height+'px';
      render=p.render({canvasContext:canvas.current.getContext('2d'),viewport,transform:[ratio,0,0,ratio,0,0]});return render.promise;
    }).then(()=>active&&setLoading(false)).catch(e=>{if(active&&e.name!=='RenderingCancelledException')setError(e.message);});
    return()=>{active=false;render?.cancel();};},[pdf,page,width,zoom]);
  return <div className="pdf-surface" ref={host}>{error?<div className="empty">{error}</div>:!document?<div className="empty">PDF 교안을 추가해주세요.</div>:<>{(!pdf||loading)&&<span className="pdf-loading">교안 불러오는 중…</span>}<canvas ref={canvas}/></>}</div>;
}
