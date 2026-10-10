"""Download pinned HF checkpoints and convert them with our bundled native importer."""
import json,os,re,shutil,subprocess,sys
from pathlib import Path

def resources():return Path(os.environ.get('SPARTADY_RESOURCE_DIR',Path(__file__).resolve().parents[2]))
def native_binary():
 override=os.environ.get('SPARTADY_LMX_BINARY')
 return Path(override) if override else resources()/('spartady-local-lmx.exe' if sys.platform=='win32' else 'spartady-local-lmx')
def install_model(directory,args,progress):
 directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
 from huggingface_hub import HfApi,hf_hub_download,snapshot_download
 repo=args.get('repo','').strip();revision=args.get('revision','main').strip()
 if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo):raise ValueError('Hugging Face의 조직/모델 이름을 입력해주세요.')
 info=HfApi().model_info(repo,revision=revision);commit=info.sha
 folder=Path(directory)/'model-downloads'/repo.replace('/','--')/commit
 progress({'event':'model','stage':'config','repo':repo})
 config_path=hf_hub_download(repo,'config.json',revision=commit,local_dir=folder)
 config=json.loads(Path(config_path).read_text())
 supported={'qwen2','qwen3','qwen3_moe','llama','mistral'}
 if config.get('model_type') not in supported or not isinstance(config.get('quantization') or config.get('quantization_config'),dict):
  raise ValueError('지원되는 MLX 양자화 체크포인트를 선택해주세요. Qwen2/3, Llama, Mistral 계열을 지원합니다.')
 target=Path(directory)/'models'/(repo.replace('/','--')+'--'+commit[:12]+'.lmx');target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():return {'name':target.stem,'path':str(target),'repo':repo,'revision':commit,'cached':True}
 needed=sum(s.size or 0 for s in HfApi().model_info(repo,revision=commit,files_metadata=True).siblings if s.rfilename.endswith('.safetensors'))
 if needed and shutil.disk_usage(directory).free<needed*2+512*1024**2:raise ValueError('다운로드와 변환에 필요한 저장 공간이 부족합니다.')
 progress({'event':'model','stage':'download','repo':repo,'bytes':needed,'revision':commit})
 snapshot_download(repo,revision=commit,local_dir=folder,allow_patterns=['*.json','*.safetensors','*.jinja','*.model','LICENSE*','README.md'],max_workers=4)
 progress({'event':'model','stage':'convert','repo':repo})
 process=subprocess.run([str(native_binary()),'convert',str(folder),str(target)],capture_output=True,text=True,timeout=3600)
 if process.returncode:raise RuntimeError('LMX 변환 실패: '+process.stderr[-1200:])
 result=json.loads(process.stdout);result.update(name=target.stem,repo=repo,revision=commit)
 (target.with_suffix('.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
 progress({'event':'model','stage':'done','repo':repo})
 return result

def setup_asr(directory,progress):
 folder=Path(directory)/'asr-runtime';python=folder/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
 uv=resources()/('uv.exe' if sys.platform=='win32' else 'uv')
 progress({'event':'asr_setup','stage':'python'})
 subprocess.run([str(uv),'venv','--python','3.12','--managed-python',str(folder)],check=True,capture_output=True,text=True,timeout=900)
 packages=['mlx-audio==0.3.1'] if sys.platform=='darwin' else ['qwen-asr==0.0.6','torch','soundfile']
 progress({'event':'asr_setup','stage':'packages'})
 result=subprocess.run([str(uv),'pip','install','--python',str(python),*packages],capture_output=True,text=True,timeout=1800)
 if result.returncode:raise RuntimeError('음성 전사 환경 설치 실패: '+result.stderr[-1200:])
 (folder/'.ready').write_text('1')
 progress({'event':'asr_setup','stage':'done'})
 return {'installed':True}
