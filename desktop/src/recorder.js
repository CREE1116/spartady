// PCM WAV capture. A page turn flushes the previous page's audio before updating
// its anchor. Intervals use captured sample counts, independent of ASR latency.
export class LectureRecorder{
  constructor(onSegment,onLevel){this.onSegment=onSegment;this.onLevel=onLevel;this.parts=[];this.samples=0;this.offset=0;this.session=crypto.randomUUID();}
  async start(anchor){
    this.anchor=anchor;this.stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true},video:false});
    try{
      this.context=new AudioContext();await this.context.resume();
      this.source=this.context.createMediaStreamSource(this.stream);this.node=this.context.createScriptProcessor(4096,1,1);
      this.mute=this.context.createGain();this.mute.gain.value=0;this.source.connect(this.node);this.node.connect(this.mute);this.mute.connect(this.context.destination);
      this.node.onaudioprocess=e=>{const pcm=new Float32Array(e.inputBuffer.getChannelData(0));this.parts.push(pcm);this.samples+=pcm.length;
        this.onLevel(Math.sqrt(pcm.reduce((v,n)=>v+n*n,0)/pcm.length));if(this.samples>=this.context.sampleRate*12)this.flush();};
    }catch(error){this.stream.getTracks().forEach(t=>t.stop());throw error;}
  }
  page(anchor){this.flush();this.anchor=anchor;}
  flush(){
    if(!this.samples)return;
    const pcm=new Float32Array(this.samples);let at=0;for(const part of this.parts){pcm.set(part,at);at+=part.length;}
    const start=this.offset;this.offset+=this.samples/this.context.sampleRate*1000;
    this.parts=[];this.samples=0;
    if(pcm.length<this.context.sampleRate*.35)return;
    const wav=encodeWAV(pcm,this.context.sampleRate);
    this.onSegment({bytes:wav,...this.anchor,start_ms:Math.floor(start),end_ms:Math.ceil(this.offset),session_id:this.session});
  }
  async stop(){if(this.stopped||!this.node)return;this.stopped=true;this.node.onaudioprocess=null;this.flush();this.node.disconnect();this.source.disconnect();this.mute.disconnect();this.stream.getTracks().forEach(t=>t.stop());await this.context.close();}
}
function encodeWAV(input,rate){
  const target=16000;const count=Math.floor(input.length*target/rate);const buffer=new ArrayBuffer(44+count*2),v=new DataView(buffer);
  const ascii=(offset,s)=>[...s].forEach((c,i)=>v.setUint8(offset+i,c.charCodeAt(0)));
  ascii(0,'RIFF');v.setUint32(4,36+count*2,true);ascii(8,'WAVE');ascii(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,target,true);v.setUint32(28,target*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);ascii(36,'data');v.setUint32(40,count*2,true);
  for(let i=0;i<count;i++){const position=i*rate/target,index=Math.floor(position),fraction=position-index;const sample=input[index]*(1-fraction)+(input[Math.min(index+1,input.length-1)]||0)*fraction;v.setInt16(44+i*2,Math.max(-1,Math.min(1,sample))*32767,true);}
  return new Uint8Array(buffer);
}
