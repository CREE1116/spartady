"""Bundle Python engine, native LMX converter/runtime and optional ASR installer."""
import os,shutil,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];runtime=root/'desktop/runtime';runtime.mkdir(exist_ok=True)
exe='.exe' if os.name=='nt' else ''
subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onedir','--name','spartady-engine','--distpath',str(runtime),'--workpath',str(root/'build/pyinstaller'),'--paths',str(root),'--collect-all','sklearn','--collect-all','ortools','--collect-all','tzdata','--hidden-import','huggingface_hub','--hidden-import','PIL.Image','--hidden-import','desktop.backend.models',str(root/'desktop/backend/server.py')],cwd=root,check=True)
source=runtime/'spartady-engine';dest=runtime/'backend'
if dest.exists():shutil.rmtree(dest)
source.rename(dest)
shutil.copy2(root/f'integrations/local_lmx/target/release/spartady-local-lmx{exe}',runtime/f'spartady-local-lmx{exe}')
import uv
shutil.copy2(uv.find_uv_bin(),runtime/f'uv{exe}')
shutil.copy2(root/'desktop/backend/asr.py',runtime/'asr.py')
