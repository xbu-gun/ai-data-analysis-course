"""下载已定位的UCI官方数据，仅作来源核对，不替换课程输入。"""
import urllib.request,zipfile
from study_common import *

def main():
    setup();root=WORK/'external_sources/uci572';root.mkdir(parents=True,exist_ok=True)
    archive=root/'taiwanese_bankruptcy_prediction.zip'
    url=PROTOCOL['bankruptcy_source_check']['official_download']
    if not archive.exists():
        with urllib.request.urlopen(url,timeout=60) as response:
            content=response.read()
        archive.write_bytes(content)
    with zipfile.ZipFile(archive) as z:
        member=next(n for n in z.namelist() if n.rsplit('/',1)[-1]=='data.csv')
        content=z.read(member)
    target=root/'data.csv'
    if target.exists():assert target.read_bytes()==content
    else:target.write_bytes(content)
    dump(root/'retrieval_manifest.json',{'retrieved_beijing':now(),'landing_page':PROTOCOL['bankruptcy_source_check']['official_landing_page'],
         'download_url':url,'archive_sha256':sha(archive),'csv_sha256':sha(target),'csv_bytes':target.stat().st_size,
         'use':'source equality check only; do not replace course data'})
    print(target)

if __name__=='__main__':main()
