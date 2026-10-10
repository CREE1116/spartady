"""Qwen3-ASR worker; timestamps identify capture intervals, not word alignment."""
import contextlib,json,sys

def main():
 request=json.load(sys.stdin)
 with contextlib.redirect_stdout(sys.stderr):
  if sys.platform=='darwin':
   from mlx_audio.stt.utils import load_model
   name='mlx-community/Qwen3-ASR-0.6B-4bit';model=load_model(name)
   result=model.generate(request['path'],language='Korean',max_tokens=512,verbose=False);text=result.text
  else:
   import torch
   from qwen_asr import Qwen3ASRModel
   name='Qwen/Qwen3-ASR-0.6B';device='cuda:0' if torch.cuda.is_available() else 'cpu'
   model=Qwen3ASRModel.from_pretrained(name,dtype=torch.float16 if device!='cpu' else torch.float32,device_map=device,max_inference_batch_size=1,max_new_tokens=512)
   text=model.transcribe(audio=request['path'],language='Korean')[0].text
 print(json.dumps({'text':text,'model':name,'timestamp_basis':'CAPTURE_INTERVAL_NOT_WORD_ALIGNMENT'},ensure_ascii=False))
if __name__=='__main__':main()
