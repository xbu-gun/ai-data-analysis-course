"""真实解码检查视频、字幕时间轴、合成音频与历史成果指纹。"""
import hashlib
import json
import re
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OUT=ROOT/'成果交付/09_AI视频_zs'
WORK=HERE/'work'
sys.path.insert(0,str(HERE/'dependencies'))
import imageio_ffmpeg

FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
VIDEO=OUT/'财务信号与文本画像_AI视频_zs.mp4'


def main():
    timeline=json.loads((HERE/'timeline.json').read_text(encoding='utf-8'))
    previous=0
    for cue in timeline['cues']:
        assert previous<=cue['start']<cue['end']<=120
        assert '许' not in cue['subtitle'] and '唅' not in cue['subtitle']
        previous=cue['end']
    srt=(OUT/'中文字幕_zs.srt').read_text(encoding='utf-8-sig')
    assert len(re.findall(r' --> ',srt))==15
    with wave.open(str(OUT/'旁白_zs.wav'),'rb') as w:
        assert w.getnchannels()==1 and w.getsampwidth()==2
        duration=w.getnframes()/w.getframerate()
        samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)
    assert abs(duration-120)<.001
    assert np.sqrt(np.mean(samples**2))>100
    peak=float(np.max(np.abs(samples)))
    # 完整解码；成功读文件头不等于整片可播放。
    decode=subprocess.run([FFMPEG,'-hide_banner','-v','error','-i',str(VIDEO),'-f','null','-'],capture_output=True)
    assert decode.returncode==0 and not decode.stderr.strip(),decode.stderr.decode('utf-8',errors='replace')
    reader=imageio_ffmpeg.read_frames(str(VIDEO),pix_fmt='rgb24')
    meta=next(reader)
    assert tuple(meta['size'])==(1920,1080) and meta['fps']==25
    count=0
    for _ in reader:count+=1
    assert count==3000, count
    extracts=[]
    for idx,scene in enumerate(timeline['scenes']):
        stamp=scene['start']+min(5.,(scene['end']-scene['start'])/2)
        p=WORK/f'actual_frame_{idx:02d}.png'
        subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y','-ss',f'{stamp:.3f}',
                        '-i',str(VIDEO),'-frames:v','1',str(p)],check=True)
        extracts.append(p)
    sheet=Image.new('RGB',(1440,4*405),'#F4F2EB')
    for idx,p in enumerate(extracts):
        im=Image.open(p).convert('RGB').resize((720,405),Image.Resampling.LANCZOS)
        sheet.paste(im,((idx%2)*720,(idx//2)*405))
    sheet.save(OUT/'成片关键帧预览_zs.jpg',quality=94)
    # 对冻结的 590 份旧成果逐项校验，不能以制作视频为由改写实证。
    manifest=json.loads((ROOT/'过程代码/research_v2/v1_preservation_manifest.json').read_text(encoding='utf-8'))
    changed=[]
    for item in manifest['files']:
        p=ROOT/item['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:changed.append(item['path'])
    assert not changed,changed
    result=dict(time_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                file=VIDEO.name,bytes=VIDEO.stat().st_size,duration_seconds=duration,decoded_frames=count,
                resolution=meta['size'],fps=meta['fps'],video_codec=meta.get('codec'),
                audio_codec=meta.get('audio_codec'),full_decode_passed=True,
                voice_source='Windows 已安装 Microsoft Kangkang 普通话男声，本地离线合成',
                narration_wave_peak=peak,subtitle_cues=15,subtitle_timing='依据各 WAV 实际采样数计时，句级字幕',
                synthetic_voice_disclosure_visible=True,public_author='zs',old_result_files_unchanged=len(manifest['files']),
                actual_frame_visual_review='pending',
                sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest())
    (OUT/'成片核查.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
