import React,{useEffect,useRef,useState} from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {Check,Loader2} from 'lucide-react';
export default function Notebook({document,page,elapsed,sessionId}){
  const [text,setText]=useState(''),[preview,setPreview]=useState(false),[status,setStatus]=useState('불러오는 중'),[error,setError]=useState('');
  const state=useRef({text:'',revision:0,dirty:false,saving:false,ready:false}),timer=useRef(),mounted=useRef(true);
  const draftKey='spartady-notebook:'+document?.id;
  const save=async()=>{
    const s=state.current;if(!document||!s.ready||!s.dirty||s.saving)return;
    const snapshot=s.text;s.saving=true;if(mounted.current){setStatus('저장 중');setError('');}
    let failed=false;
    try{
      const result=await window.spartady.call('save_notebook',{document_id:document.id,text:snapshot,revision:s.revision,...s.context});
      s.revision=result.revision;if(s.text===snapshot){s.dirty=false;localStorage.removeItem(draftKey);if(mounted.current)setStatus('저장됨');}
      else{localStorage.setItem(draftKey,JSON.stringify({text:s.text,revision:s.revision}));if(mounted.current)setStatus('변경됨');}
    }catch(e){failed=true;if(mounted.current){setError(e.message);setStatus('저장 실패');}}
    finally{s.saving=false;if(s.dirty&&!failed)timer.current=setTimeout(save,1800);}
  };
  useEffect(()=>{
    mounted.current=true;if(!document){setStatus('교안을 선택해주세요');return;}
    window.spartady.call('notebook',{document_id:document.id}).then(result=>{
      if(!mounted.current)return;
      const s=state.current;s.revision=result.revision;s.text=result.text;s.ready=true;
      try{const draft=JSON.parse(localStorage.getItem(draftKey)||'null');if(draft&&draft.revision===result.revision){s.text=draft.text;s.dirty=draft.text!==result.text;}}catch{}
      s.context={page,time_ms:elapsed,session_id:sessionId};setText(s.text);setStatus(s.dirty?'변경됨':'저장됨');if(s.dirty)timer.current=setTimeout(save,1800);
    }).catch(e=>{setError(e.message);setStatus('불러오기 실패');});
    return()=>{mounted.current=false;clearTimeout(timer.current);save();};
  },[document?.id]);
  const change=value=>{
    const s=state.current;s.text=value;s.dirty=true;s.context={page,time_ms:elapsed,session_id:sessionId};setText(value);setStatus('변경됨');
    localStorage.setItem(draftKey,JSON.stringify({text:value,revision:s.revision}));clearTimeout(timer.current);timer.current=setTimeout(save,1600);
  };
  return <section className="notebook"><header><strong>강의 필기</strong><div><button className={!preview?'active':''} onClick={()=>setPreview(false)}>작성</button><button className={preview?'active':''} onClick={()=>setPreview(true)}>미리보기</button></div></header>
    {preview?<div className="markdown-body"><Markdown remarkPlugins={[remarkGfm]} skipHtml>{text||'*아직 작성한 필기가 없습니다.*'}</Markdown></div>:<textarea aria-label="강의 마크다운 필기" spellCheck={false} placeholder={'# 강의 필기\n\n페이지에 나누지 않고, 강의 전체를 이어서 기록하세요.\n\n## 중요한 내용\n- \n\n## 질문\n- '} value={text} disabled={!state.current.ready} onChange={e=>change(e.target.value)} onKeyDown={e=>{if((e.metaKey||e.ctrlKey)&&e.key==='s'){e.preventDefault();save();}if(e.key==='Tab'){e.preventDefault();const el=e.currentTarget,start=el.selectionStart,end=el.selectionEnd;change(text.slice(0,start)+'  '+text.slice(end));requestAnimationFrame(()=>{el.selectionStart=el.selectionEnd=start+2;});}}}/>}
    <footer><span>{status==='저장 중'?<Loader2 size={12} className="spin"/>:status==='저장됨'?<Check size={12}/>:null}{error||status}</span><small>Markdown · 자동 저장</small></footer>
  </section>;
}
