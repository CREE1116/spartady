import React,{useEffect,useRef,useState} from 'react';
import {useEditor,EditorContent} from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import {Markdown} from '@tiptap/markdown';
import {TableKit} from '@tiptap/extension-table';
import TaskList from '@tiptap/extension-task-list';
import TaskItem from '@tiptap/extension-task-item';
import {Check,Loader2} from 'lucide-react';
export default function Notebook({document,page,elapsed,sessionId}){
  const [status,setStatus]=useState('불러오는 중'),[error,setError]=useState('');
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
  const change=value=>{
    const s=state.current;s.text=value;s.dirty=true;s.context={page,time_ms:elapsed,session_id:sessionId};setStatus('변경됨');
    localStorage.setItem(draftKey,JSON.stringify({text:value,revision:s.revision}));clearTimeout(timer.current);timer.current=setTimeout(save,1600);
  };
  const context=useRef();context.current={page,time_ms:elapsed,session_id:sessionId};
  const changeRef=useRef();changeRef.current=change;
  const editor=useEditor({extensions:[StarterKit.configure({link:{openOnClick:false}}),TableKit,TaskList,TaskItem.configure({nested:true}),Markdown],content:'',contentType:'markdown',editable:false,
    editorProps:{attributes:{role:'textbox','aria-label':'강의 마크다운 필기','aria-multiline':'true',spellcheck:'false'},handleKeyDown:(_,event)=>{if((event.metaKey||event.ctrlKey)&&event.key==='s'){event.preventDefault();save();return true;}return false;}},
    onUpdate:({editor})=>{state.current.context=context.current;changeRef.current(editor.getMarkdown());}});
  useEffect(()=>{
    mounted.current=true;if(!document){setStatus('교안을 선택해주세요');return;}
    window.spartady.call('notebook',{document_id:document.id}).then(result=>{
      if(!mounted.current)return;
      const s=state.current;s.revision=result.revision;s.text=result.text;s.ready=true;
      try{const draft=JSON.parse(localStorage.getItem(draftKey)||'null');if(draft&&draft.revision===result.revision){s.text=draft.text;s.dirty=draft.text!==result.text;}}catch{}
      s.context={page,time_ms:elapsed,session_id:sessionId};editor?.commands.setContent(s.text,{contentType:'markdown',emitUpdate:false});editor?.setEditable(true);setStatus(s.dirty?'변경됨':'저장됨');if(s.dirty)timer.current=setTimeout(save,1800);
    }).catch(e=>{setError(e.message);setStatus('불러오기 실패');});
    return()=>{mounted.current=false;clearTimeout(timer.current);save();};
  },[document?.id,editor]);
  return <section className="notebook"><header><strong>강의 필기</strong><small># 제목 · **굵게** · - 목록</small></header>
    <EditorContent editor={editor} className="notebook-editor markdown-body"/>
    <footer><span>{status==='저장 중'?<Loader2 size={12} className="spin"/>:status==='저장됨'?<Check size={12}/>:null}{error||status}</span><small>Markdown · 자동 저장</small></footer>
  </section>;
}
