"""Real HF download -> LMX conversion -> CPU inference on each installer platform."""
import os,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from desktop.backend.models import install_model
from engine.second_brain.providers import LocalLMX
root=Path(__file__).resolve().parents[1]
binary=root/'integrations/local_lmx/target/release'/('spartady-local-lmx.exe' if os.name=='nt' else 'spartady-local-lmx')
os.environ['SPARTADY_LMX_BINARY']=str(binary);os.environ['SPARTADY_LMX_GPU']='0'
with tempfile.TemporaryDirectory(prefix='spartady-native-') as data:
 result=install_model(data,{'repo':'unsloth/Qwen3.5-0.8B-GGUF','filename':'Qwen3.5-0.8B-Q4_K_M.gguf'},print)
 backend=LocalLMX(result['path'],binary=binary)
 try:
  answer=backend.complete('Answer briefly.','Say hello.',64)
  assert answer['text'].strip() and answer['provider']=='local_lmx'
  print('NATIVE_CPU_CONVERSION_INFERENCE_PASS')
 finally:backend.close()
