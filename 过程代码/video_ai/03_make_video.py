"""本地合成 1080p 动态信息图、实测时间轴、句级字幕与普通话旁白。"""
import hashlib
import json
import math
import os
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / '成果交付/09_AI视频_zs'
WORK = HERE / 'work'
sys.path.insert(0, str(HERE / 'dependencies'))
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CFG = json.loads((HERE / 'storyboard.json').read_text(encoding='utf-8'))
W, H = CFG['resolution']
FPS = CFG['fps']
BG = '#F4F2EB'
INK = '#172F37'
MUTED = '#647575'
TEAL = '#087E78'
ORANGE = '#C45432'
GOLD = '#B28A35'
PAPER = '#FFFEFA'
FONT = 'C:/Windows/Fonts/msyh.ttc'
BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
LATIN = 'C:/Windows/Fonts/bahnschrift.ttf'
FONTS = {}


def font(size, bold=False, latin=False):
    key = (size, bold, latin)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(LATIN if latin else BOLD if bold else FONT, size)
    return FONTS[key]


def text(d, xy, value, size=36, fill=INK, bold=False, latin=False):
    d.text(xy, str(value), font=font(size, bold, latin), fill=fill,
           anchor='lt', stroke_width=0)


def wrap(value, size, width, bold=False):
    result = []
    line = ''
    f = font(size, bold)
    for char in value:
        if f.getlength(line + char) > width and line:
            result.append(line)
            line = char
        else:
            line += char
    if line:
        result.append(line)
    return result


def paragraph(d, xy, value, size=32, width=600, fill=MUTED, line_height=None, bold=False):
    x, y = xy
    for line in wrap(value, size, width, bold):
        text(d, (x, y), line, size, fill, bold)
        y += line_height or round(size * 1.55)
    return y


def pill(d, box, label, color=TEAL, size=27, light='#E1EDE6'):
    d.rounded_rectangle(box, radius=(box[3]-box[1])//2, fill=light)
    f = font(size, True)
    tw = f.getlength(label)
    text(d, ((box[0]+box[2]-tw)/2, box[1]+((box[3]-box[1])-size)/2-2), label, size, color, True)


def card(d, box, color=PAPER, radius=24, outline=None):
    d.rounded_rectangle(box, radius=radius, fill=color, outline=outline, width=2 if outline else 1)


def line(d, points, color=TEAL, width=4):
    d.line(points, fill=color, width=width, joint='curve')


def arrow(d, start, end, color=TEAL, width=4):
    line(d, [start,end], color, width)
    a = math.atan2(end[1]-start[1], end[0]-start[0])
    for delta in [-0.45,0.45]:
        p = (end[0]-18*math.cos(a+delta),end[1]-18*math.sin(a+delta))
        line(d, [p,end], color, width)


def base(scene, idx):
    im = Image.new('RGB', (W,H), BG)
    d = ImageDraw.Draw(im)
    for x in range(56,W,80):
        for y in range(60,920,80):
            d.ellipse((x,y,x+2,y+2), fill='#D5DDD3')
    d.rectangle((0,0,16,H), fill=TEAL)
    text(d,(80,42),'zs  /  财务信号与文本画像',28,INK,True)
    text(d,(1392,45),'AI 合成男声 · 中文字幕',25,MUTED)
    line(d,[(80,102),(1840,102)],'#D0D9D0',2)
    text(d,(82,136),scene['eyebrow'],29,TEAL,True)
    text(d,(78,193),scene['title'],70,INK,True)
    text(d,(80,862),'课程研究演示  ·  非真实概率或审计结论',24,MUTED)
    text(d,(1720,862),f'{idx+1:02d} / 07',28,TEAL,latin=True)
    return im


def opening(d):
    text(d,(84,308),'用财务指标发现线索',45,INK,True)
    text(d,(84,380),'用年报文字理解企业',45,INK,True)
    paragraph(d,(87,490),'从数据探索、模型比较，到可查询的网站与可核查的证据。',34,735)
    pill(d,(86,650,358,710),'公开署名  zs',TEAL,29)
    text(d,(90,757),'本片使用合成旁白，未模拟作者本人声音或形象。',25,MUTED)
    # 抽象数据与文字图形，仅作视觉引导，不绘制无来源数据曲线。
    card(d,(1040,311,1786,776),INK)
    text(d,(1090,350),'FINANCIAL SIGNALS',31,'#D5E2D6',latin=True)
    text(d,(1090,402),'财务  ×  文本',54,'#FFFFFF',True)
    for i,label in enumerate(['数据','模型','解释','证据']):
        x = 1090 + i*160
        card(d,(x,550,x+124,653),'#28484D',16)
        text(d,(x+21,585),label,39,'#E6EDE3',True)
        if i<3: arrow(d,(x+128,602),(x+155,602),'#63BBA5',3)
    text(d,(1094,702),'课程项目说明  /  约 2 分钟',28,'#A7C9BE')


def data_scene(d):
    specs=[(80,'舞弊面板','42,511','条记录','按年份前推','训练 2007—2019；测试 2020—2021'),
           (680,'破产数据','6,819','条记录','分层留出','训练 / 验证 / 测试分开'),
           (1280,'年报 MD&A','1,123','篇文本','225 家公司','2018—2022 年')]
    for x,label,value,unit,tag,note in specs:
        card(d,(x,326,x+560,667))
        text(d,(x+33,360),label,37,INK,True)
        text(d,(x+28,425),value,94,TEAL,latin=True)
        text(d,(x+35,530),unit,28,MUTED)
        text(d,(x+35,588),tag,30,INK,True)
        text(d,(x+35,635),note,22,MUTED)
    for i,label in enumerate(['训练集聚类探索','比较四类模型','锁定参数与阈值','留出测试终评']):
        x=110+i*444
        text(d,(x,735),label,30,INK,True)
        if i<3:arrow(d,(x+304,755),(x+399,755),TEAL,3)
    text(d,(83,808),'预处理只在训练部分拟合，测试集不参与调参。',28,TEAL,True)


def model_scene(d):
    f = CFG['facts']['fraud']; b=CFG['facts']['bankruptcy']
    for x,label,v,ap,baseline,fp,fn in [(80,'舞弊预测',f['roc_auc'],f['pr_auc_ap'],f['positive_baseline'],3414,313),
                                      (985,'破产预测',b['roc_auc'],b['pr_auc_ap'],b['positive_baseline'],39,14)]:
        card(d,(x,323,x+855,764))
        text(d,(x+35,353),label,39,INK,True)
        pill(d,(x+590,347,x+817,400),'XGBoost',TEAL,26)
        text(d,(x+35,427),'ROC-AUC',29,MUTED,latin=True)
        text(d,(x+32,475),f'{v:.3f}',101,TEAL,latin=True)
        text(d,(x+362,465),f'AP  {ap:.3f}',34,INK,latin=True)
        text(d,(x+362,523),f'正类基线  {baseline:.3f}',28,MUTED)
        line(d,[(x+35,600),(x+817,600)],'#DCE2D9',2)
        text(d,(x+35,640),f'误报  {fp:,}',36,ORANGE,True)
        text(d,(x+430,640),f'漏报  {fn:,}',36,ORANGE,True)
        text(d,(x+35,709),'固定阈值下的测试集计数',25,MUTED)
    text(d,(84,802),'测试集：舞弊 8,724 条；破产 1,023 条（其中正类 33 条）。',29,INK)


def website_scene(d):
    card(d,(80,324,1840,775))
    text(d,(122,362),'5,574',109,TEAL,latin=True)
    text(d,(126,487),'家公司 · 风险排序',36,INK,True)
    pill(d,(126,560,490,617),'另用可解释决策树',TEAL,27)
    text(d,(128,661),'深度 5  ·  31 个叶节点',28,MUTED)
    line(d,[(578,365),(578,731)],'#DCE2D9',2)
    text(d,(635,363),'网站查询流程（功能示意）',31,MUTED)
    labels=[('查询公司','公司代码 / 名称'),('财务与评分','查看相对风险线索'),('判定路径','追踪树模型的分支')]
    for i,(label,note) in enumerate(labels):
        x=633+i*380
        card(d,(x,442,x+324,604),'#E7EEE5',18)
        text(d,(x+25,475),label,36,INK,True)
        text(d,(x+25,543),note,25,MUTED)
        if i<2:arrow(d,(x+331,523),(x+371,523),TEAL,3)
    paragraph(d,(635,652),'网页应用模型与上页测试模型不同；不能借用 XGBoost 成绩。',28,1100,ORANGE)
    text(d,(84,809),'加权分数 ≠ 真实造假概率；应用集无标签，实际准确率未知。',32,ORANGE,True)


def text_scene(d):
    card(d,(80,323,805,770))
    text(d,(116,354),'五维文化指标',39,INK,True)
    for i,label in enumerate(['创新','诚信','质量','尊重','团队']):
        x=119+(i%3)*209;y=443+(i//3)*82
        pill(d,(x,y,x+180,y+57),label,TEAL,31)
    text(d,(118,634),'784',75,TEAL,latin=True)
    text(d,(298,662),'个词  ·  经确认删词后保留',29,MUTED)
    text(d,(118,723),'衡量管理层如何描述自己',28,INK)
    card(d,(846,323,1840,770))
    text(d,(882,354),'全文分段 → 汇总向量 → BERTopic',36,INK,True)
    vals=[('41,099','全文分段'),('12','描述性主题'),('110','离群文档')]
    for i,(value,label) in enumerate(vals):
        x=883+i*308
        text(d,(x,452),value,64,TEAL,latin=True)
        text(d,(x,540),label,29,MUTED)
    text(d,(887,624),'主随机种子 42；按既定规则合并后为 12 主题。',27,INK)
    text(d,(887,674),'保留首 / 中 / 尾旧方案对照；主题含混合业务。',26,MUTED)
    text(d,(887,721),'主题 ≠ 行业分类  ·  覆盖完整 ≠ 准确率提升',27,ORANGE,True)
    text(d,(84,810),'文化得分衡量自我披露，不能直接代表实际行为。',30,ORANGE,True)


def evidence_scene(d):
    labels=[('01','固定划分','数据指纹与训练 / 测试边界'),('02','错误明细','把误报和漏报留给读者核查'),('03','原文定位','文本例证保留来源与位置')]
    for i,(num,label,note) in enumerate(labels):
        x=80+i*600
        card(d,(x,330,x+560,568))
        text(d,(x+28,354),num,44,TEAL,latin=True)
        text(d,(x+28,422),label,43,INK,True)
        text(d,(x+28,513),note,27,MUTED)
    card(d,(80,612,1840,785),'#F2E5D9')
    text(d,(116,638),'仍待核实：收入分母、accruals 与历史更正版本',38,ORANGE,True)
    text(d,(118,712),'保留课堂数值和证据清单；不作真实事前预警的强表述。',33,INK)


def closing(d):
    card(d,(80,319,1840,811),INK)
    text(d,(124,354),'作者确认',30,'#9DBFB6',True)
    text(d,(124,405),'样本  ·  权重  ·  删词  ·  展示方案',40,'#FFFFFF',True)
    text(d,(124,482),'AI 协助',30,'#9DBFB6',True)
    text(d,(124,534),'编程  ·  计算  ·  证据核查',40,'#FFFFFF',True)
    line(d,[(124,614),(1788,614)],'#4B6266',2)
    text(d,(126,644),'网站  +  代码仓库  +  一页结果说明',37,'#E5EFDE',True)
    text(d,(126,709),'github.com/xbu-gun/ai-data-analysis-course',31,'#A7D5C5',latin=True)
    text(d,(126,761),'financial-signals-zs.satiny-grebe-2478.chatgpt.site',27,'#A7D5C5',latin=True)


DRAWERS=[opening,data_scene,model_scene,website_scene,text_scene,evidence_scene,closing]


def make_slides():
    slides=[]
    slide_dir=WORK/'slides'; slide_dir.mkdir(exist_ok=True)
    for idx,scene in enumerate(CFG['scenes']):
        bg=base(scene,idx)
        content=Image.new('RGBA',(W,H),(0,0,0,0))
        d=ImageDraw.Draw(content)
        DRAWERS[idx](d)
        whole=bg.convert('RGBA');whole.alpha_composite(content)
        whole.convert('RGB').save(slide_dir/f'{idx:02d}_{scene["id"]}.png')
        slides.append((bg,content,whole.convert('RGB')))
    # 静态联络表供版面核查，字幕另从实际成片抽帧核查。
    sheet=Image.new('RGB',(1440,4*405),BG)
    for idx,(_,_,im) in enumerate(slides):
        sheet.paste(im.resize((720,405),Image.Resampling.LANCZOS),((idx%2)*720,(idx//2)*405))
    sheet.save(WORK/'contact_sheet.png')
    cover=slides[0][2].copy()
    d=ImageDraw.Draw(cover)
    text(d,(84,938),'AI 合成男声  /  动态图表  /  中文字幕',34,TEAL,True)
    cover.save(OUT/'视频封面_zs.png')
    return slides


def seconds_stamp(value):
    ms=round(value*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def make_timeline():
    params=None;samples=[];timeline=[];cues=[];offset=0
    def pause(seconds):
        nonlocal offset
        n=round(seconds*params.framerate)
        samples.append(bytes(n*params.nchannels*params.sampwidth));offset+=n
    for idx,scene in enumerate(CFG['scenes']):
        st=dict(id=scene['id'],index=idx,title=scene['title'],start=offset/(params.framerate if params else 22050))
        if idx==0:
            first=WORK/'narration'/f"{scene['cues'][0]['id']}.wav"
            with wave.open(str(first),'rb') as w:params=w.getparams()
            pause(.6)
        elif idx>0:
            pause(.18)
        for c in scene['cues']:
            with wave.open(str(WORK/'narration'/f"{c['id']}.wav"),'rb') as w:
                p=w.getparams();assert p[:4]==params[:4] or (p.nchannels,p.sampwidth,p.framerate)==(params.nchannels,params.sampwidth,params.framerate)
                raw=w.readframes(w.getnframes())
            start=offset/params.framerate
            samples.append(raw);offset+=len(raw)//(params.sampwidth*params.nchannels)
            end=offset/params.framerate
            cues.append(dict(**c,start=start,end=end,scene=scene['id']))
            if c['id']!=CFG['scenes'][-1]['cues'][-1]['id']:pause(.08)
        st['end']=offset/params.framerate
        timeline.append(st)
    target=CFG['target_seconds']
    assert offset/params.framerate<target-1, '需要先调整旁白语速；不可强行裁断。'
    pause(target-offset/params.framerate)
    timeline[-1]['end']=target
    narration=OUT/'旁白_zs.wav'
    with wave.open(str(narration),'wb') as w:
        w.setnchannels(params.nchannels);w.setsampwidth(params.sampwidth);w.setframerate(params.framerate)
        w.writeframes(b''.join(samples))
    assert len(samples)>15
    # WAV 全长与字幕使用同一采样计数时间轴，不用估计字数定时。
    srt=[]
    for i,c in enumerate(cues):
        lines=wrap(c['subtitle'],40,1650)
        assert len(lines)<=2, (c['id'],lines)
        srt.append(f'{i+1}\n{seconds_stamp(c["start"])} --> {seconds_stamp(c["end"])}\n'+ '\n'.join(lines)+'\n')
    (OUT/'中文字幕_zs.srt').write_text('\n'.join(srt),encoding='utf-8-sig')
    data=dict(duration=target,scenes=timeline,cues=cues,voice=CFG['voice'],speech_rate=CFG['voice_rate'],sample_rate=params.framerate)
    (HERE/'timeline.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    return data,narration


def subtitle_image(value):
    im=Image.new('RGBA',(W,166),(0,0,0,0));d=ImageDraw.Draw(im)
    d.rounded_rectangle((65,2,1855,152),radius=18,fill=(23,47,55,244))
    lines=wrap(value,40,1650)
    assert len(lines)<=2
    y=39 if len(lines)==1 else 17
    for v in lines:
        tw=font(40).getlength(v)
        text(d,((W-tw)/2,y),v,40,'#FFFFFF')
        y+=57
    return im


def render(slides,timeline,narration):
    output=OUT/'财务信号与文本画像_AI视频_zs.mp4'
    captions={c['id']:subtitle_image(c['subtitle']) for c in timeline['cues']}
    duration=timeline['duration'];frames=round(duration*FPS)
    cmd=[FFMPEG,'-hide_banner','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
         '-r',str(FPS),'-i','pipe:0','-i',str(narration),'-map','0:v:0','-map','1:a:0',
         '-c:v','libx264','-preset','veryfast','-crf','20','-threads','4','-pix_fmt','yuv420p',
         '-af','loudnorm=I=-16:TP=-1.5:LRA=11','-c:a','aac','-b:a','160k','-ar','48000',
         '-movflags','+faststart','-t',str(duration),'-metadata','title=财务信号与文本画像｜zs',
         '-metadata','artist=zs','-metadata','comment=AI synthetic Mandarin narration and motion graphics; no voice or likeness cloning.',
         str(output)]
    log=WORK/'encoding.log'
    with log.open('wb') as stderr:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=stderr)
        try:
            for frame_number in range(frames):
                t=frame_number/FPS
                idx=next((s['index'] for s in timeline['scenes'] if s['start']<=t<s['end']),len(slides)-1)
                s=timeline['scenes'][idx]; bg,content,whole=slides[idx]
                rel=t-s['start']
                enter=min(1.,max(0.,rel/.65));ease=1-(1-enter)**3
                if enter<1:
                    im=bg.copy()
                    layer=content.copy()
                    alpha=layer.getchannel('A').point(lambda x:int(x*ease))
                    layer.putalpha(alpha)
                    im.paste(layer,(0,round(20*(1-ease))),layer)
                else: im=whole.copy()
                d=ImageDraw.Draw(im)
                # 阶段进度轨道只表示视频播放位置，不表示模型精度。
                d.rectangle((80,910,1840,914),fill='#D9E0D5')
                d.rectangle((80,910,80+round(1760*t/duration),914),fill=TEAL)
                cue=next((c for c in timeline['cues'] if c['start']<=t<c['end']),None)
                if cue:
                    cap=captions[cue['id']]
                    im.paste(cap,(0,927),cap)
                # 稍微淡出，结尾不切断旁白。
                remaining=duration-t
                if remaining<.6:
                    im=Image.blend(Image.new('RGB',(W,H),BG),im,max(0.,remaining/.6))
                process.stdin.write(im.tobytes())
                if frame_number%(FPS*10)==0:
                    print(f'已编码 {t:.0f} / {duration} 秒',flush=True)
        finally:
            process.stdin.close()
        rc=process.wait()
        if rc:raise RuntimeError(f'视频编码失败：{rc}，见 {log}')
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    slides=make_slides()
    timeline,narration=make_timeline()
    print('分镜与实测字幕时间轴已生成。',flush=True)
    if '--slides-only' in sys.argv:return
    output=render(slides,timeline,narration)
    receipt=dict(time_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                 author='zs',duration_seconds=timeline['duration'],resolution=[W,H],fps=FPS,
                 voice=CFG['voice'],narration_disclosure=CFG['disclosure'],
                 output=output.name,bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                 encoder='imageio-ffmpeg 0.6.0 / libx264 + AAC',
                 scene_count=len(slides),subtitle_cues=len(timeline['cues']))
    (OUT/'视频制作回执.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
