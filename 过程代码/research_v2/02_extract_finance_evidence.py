"""只从大型原始XLSX流式提取预设公司的年报候选行，保留原始行号。"""
import html,re,shutil,zipfile,json
from lxml import etree
from study_common import *

NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def rows_from_xml(stream):
    buffer=b''
    pattern=re.compile(rb'<row\b[^>]*>.*?</row>',re.S)
    while True:
        chunk=stream.read(1024*1024)
        buffer+=chunk
        end=0
        for match in pattern.finditer(buffer):
            end=match.end();yield match.group()
        buffer=buffer[end:]
        if not chunk:break

def cell_value(attrs,body,strings):
    match=re.search(rb'<v[^>]*>(.*?)</v>',body,re.S)
    if b't="s"' in attrs:
        return strings[int(match.group(1))] if match else ''
    if b't="inlineStr"' in attrs:
        return ''.join(html.unescape(x.decode('utf-8')) for x in re.findall(rb'<t\b[^>]*>(.*?)</t>',body,re.S))
    return html.unescape(match.group(1).decode('utf-8')) if match else ''

def parse_row(row,strings):
    values={}
    for attrs,body in re.findall(rb'<c\b([^>]*?)>(.*?)</c>',row,re.S):
        address=re.search(rb'\br="([A-Z]+)\d+"',attrs)
        if address:values[address.group(1).decode()]=cell_value(attrs,body,strings)
    return values

def extract(archive_name,book_name):
    root=WORK/'raw_finance';root.mkdir(parents=True,exist_ok=True)
    target=root/book_name
    archive=DATA/'舞弊多年份/原始数据'/archive_name
    with zipfile.ZipFile(archive) as outer:
        size=outer.getinfo(book_name).file_size
        if target.exists():assert target.stat().st_size==size
        else:
            with outer.open(book_name) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    print('Extracted',book_name,size,flush=True)
    wanted=set(PROTOCOL['raw_finance_spotcheck']['company_codes'])
    years={2005,2006,2007,2010,2011,2012,2017,2018,2019,2023,2024,2025}
    selected=[];header=None;scanned=0
    with zipfile.ZipFile(target) as inner:
        strings=[]
        if 'xl/sharedStrings.xml' in inner.namelist():
            with inner.open('xl/sharedStrings.xml') as f:
                for _,element in etree.iterparse(f,events=('end',),tag=NS+'si'):
                    strings.append(''.join(element.itertext()));element.clear()
                    while element.getprevious() is not None:del element.getparent()[0]
        sheet=next(n for n in inner.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml'))
        with inner.open(sheet) as f:
            for row in rows_from_xml(f):
                scanned+=1
                if scanned==1:
                    header=parse_row(row,strings);print('HEADERS',book_name,list(header.items())[:8],flush=True);continue
                first=re.search(rb'<c\b([^>]*\br="A\d+"[^>]*)>(.*?)</c>',row,re.S)
                if not first:continue
                code=cell_value(first.group(1),first.group(2),strings)
                if code not in wanted:continue
                vals=parse_row(row,strings)
                record={name:vals.get(col,'') for col,name in header.items()}
                date=record.get('Accper','')
                if not date.endswith('12-31') or int(date[:4]) not in years or record.get('Typrep')!='A':continue
                record.update(source_archive=archive_name,source_workbook=book_name,source_sheet=sheet,
                              source_excel_row=int(re.search(rb'<row\b[^>]*\br="(\d+)"',row).group(1)))
                selected.append(record)
                if len(selected)%10==0:print('Selected',book_name,len(selected),'at physical row',scanned,flush=True)
    dump(root/(book_name+'.selected.json'),{'source_archive_sha256':sha(archive),'xlsx_sha256':sha(target),
         'scanned_rows':scanned,'selected_rows':selected,'selected_count':len(selected)})
    print('DONE',book_name,'scanned',scanned,'selected',len(selected),flush=True)

def main():
    setup()
    for args in [('资产负债表.zip','FS_Combas.xlsx'),('利润表.zip','FS_Comins.xlsx')]:extract(*args)

if __name__=='__main__':main()
