"""Read-only lists and side-by-side diagnostic sheets for deformation audits."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def category(row):
    status=row['status']
    if status=='no_global_deformation_detected' and row.get('local_structure',{}).get('status')=='inconclusive':
        return 'local_unverified'
    return status


def priority(row):
    local=row.get('local_structure',{})
    if category(row)=='global_deformation_suspected':return (0,-row['shape']['anisotropy_ratio'])
    if category(row)=='local_structure_change_suspected':return (1,-local.get('fraction_over_3px',0)*local.get('p90_displacement_px',0))
    return (2,-row.get('shape',{}).get('anisotropy_ratio',1))


def make_report(output):
    manifest=json.loads((output/'manifest.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((output/'measurements').glob('*.json'))]
    for row in rows:
        row['artists']=[a.get('name','') if isinstance(a,dict) else a for a in row['artists']]
    if len(rows)!=len(manifest['rows']):raise ValueError('Finish the complete manifest before reporting')
    if {r['id'] for r in rows}!={r['id'] for r in manifest['rows']}:raise ValueError('Manifest/result mismatch')
    counts={label:dict(Counter(category(r) for r in rows if r['dataset']==label)) for label in ['sumbul_approved','nilufer_all']}
    flagged=sorted([r for r in rows if category(r) in ['global_deformation_suspected','local_structure_change_suspected']],key=priority)
    fields=['dataset','id','title','artists','model','aspect_ratio','category','anisotropy_ratio','shear_degrees',
            'local_p90_displacement_px','local_fraction_over_3px','local_valid_textured_fraction','source_file','raw_file']
    with (output/'all-results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in sorted(rows,key=lambda r:(r['dataset'],priority(r))):
            shape=r.get('shape') or {};local=r.get('local_structure') or {}
            writer.writerow({'dataset':r['dataset'],'id':r['id'],'title':r['title'],'artists':', '.join(r['artists']),
                'model':r['model'],'aspect_ratio':r.get('aspect_ratio'),'category':category(r),
                'anisotropy_ratio':shape.get('anisotropy_ratio'),'shear_degrees':shape.get('shear_deviation_degrees'),
                'local_p90_displacement_px':local.get('p90_displacement_px'),
                'local_fraction_over_3px':local.get('fraction_over_3px'),
                'local_valid_textured_fraction':local.get('valid_textured_fraction'),
                'source_file':r['source_file'],'raw_file':r['raw_file']})
    with (output/'all-results.csv').open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.DictReader(stream))
    for filename,categories in [('suspects.csv',{'global_deformation_suspected','local_structure_change_suspected'}),
                                ('uncertain.csv',{'borderline','inconclusive','local_unverified'})]:
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows(row for row in records if row['category'] in categories)
    report={'scope':len(rows),'counts':counts,'suspect_count':len(flagged),'suspects':flagged,
            'notes':['Candidates are not automatic rejections. Human decisions were preserved.',
                     'No detected deformation is not a guarantee of anatomy, text, or complete source fidelity.',
                     'Local structure changes can be redraws as well as bends. Visual review is needed.',
                     'Flow fractions are only of textured, visible, bidirectionally consistent pixels, not the whole cover.']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    text=['# Bükülme / esneme denetimi','',f'{len(rows)} çıktı tarandı; {len(flagged)} görsel yeniden inceleme adayı.','',
          'Konum kayması, dönme ve eşit ölçek bozulma sayılmadı. Yerel farklar yeniden çizim de olabilir; bu liste otomatik ret değildir.',
          'Belirsizler temiz kabul edilmedi. İnsan seçimleri ve görseller değiştirilmedi.','',
          '| Set | Genel esneme | Yerel yapı farkı | Sınırda | Ölçülemeyen | Yerel test belirsiz | Belirgin fark saptanmadı |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name,c in counts.items():
        keys=['global_deformation_suspected','local_structure_change_suspected','borderline','inconclusive','local_unverified','no_global_deformation_detected']
        text.append('| '+name+' | '+' | '.join(str(c.get(k,0)) for k in keys)+' |')
    text+=['','## Yeniden incelenecek adaylar','']
    for r in flagged:
        text.append(f"- **{r['title']}** — {', '.join(r['artists'])}; {r['dataset']}, {r['id']} ({category(r)}). [Kaynak]({r['source_file']}) · [Ham çıktı]({r['raw_file']})")
    (output/'report.md').write_text('\n'.join(text)+'\n')
    return flagged


def sheets(output, rows):
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    cv2.setNumThreads(1)
    font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
    font=ImageFont.truetype(font_path,14)
    pages=[]
    for start in range(0,len(rows),4):
        subset=rows[start:start+4];sheet=Image.new('RGB',(600,len(subset)*285),(24,26,24));draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(subset):
            matrix=np.asarray(row['matrix_source_to_analysis'],float)
            u,s,vt=np.linalg.svd(matrix[:,:2])
            # Closest uniform-scale rotation: this intentionally does not undo
            # unequal scale/shear/local bends in the diagnostic teacher crop.
            rotation=u@vt;linear=rotation*float(s.mean())
            center=matrix@np.array([256,256,1]);transform=np.column_stack((linear,center-linear@np.array([256,256])))
            with Image.open(row['source_file']) as image:source=image.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
            with Image.open(row['raw_file']) as image:
                working=np.asarray(image.convert('RGB').resize(tuple(row['analysis_dimensions']),Image.Resampling.LANCZOS))
            candidate=cv2.warpAffine(working,transform,(512,512),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            y=j*285;label=f"{start+j+1}. {row['title']} | {row['id']}"
            draw.text((10,y+5),label[:90],font=font,fill=(240,240,230))
            draw.text((18,y+28),'Kaynak',font=font,fill=(200,215,190))
            draw.text((309,y+28),'Öğretmen bölgesi',font=font,fill=(255,185,100))
            sheet.paste(source.resize((225,225)),(18,y+51));sheet.paste(Image.fromarray(candidate).resize((225,225)),(309,y+51))
        path=output/f'comparison-{start//4+1:02}.jpg';sheet.save(path,quality=90);pages.append(str(path.resolve()))
    (output/'comparison-index.json').write_text(json.dumps(pages,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--sheets',action='store_true');args=parser.parse_args()
    rows=make_report(args.output)
    if args.sheets:sheets(args.output,rows)
