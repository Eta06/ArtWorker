"""Read-only local texture diagnostics; diagnostic ROIs are not an algorithm mask."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/Users/emir/Documents/ChatGPT/ArtWorker')
OUT=Path(__file__).parent
RAW=ROOT/'experiments/qwen/controlnet_runs/2026-10-05/track3-saved-top32-outpaint/tangent-canny/top32/raw.png'
SOURCE=ROOT/'experiments/evaluation/inputs/track3/source_512.png'
PROFILE=ROOT/'experiments/qwen/boundary_compositor_runs/2026-10-05/track3-top32-profile16/direct-profile32/composite.png'
cv2.setNumThreads(1)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lum(a):
    return np.einsum('...c,c->...',a.astype(np.float64),[.2126,.7152,.0722])

def detrend(z):
    y,x=np.indices(z.shape)
    A=np.stack([np.ones(z.size),x.ravel(),y.ravel()],axis=1)
    coef=np.linalg.lstsq(A,z.ravel(),rcond=None)[0]
    return z-(A@coef).reshape(z.shape)

def statistics(a):
    z=lum(a); zz=detrend(z)
    gx=np.diff(z,axis=1); gy=np.diff(z,axis=0)
    tensor=np.cov(np.stack([gx[:-1].ravel(),gy[:,:-1].ravel()]))
    eigen=np.linalg.eigvalsh(tensor)
    stats={'luma_std':float(z.std()),'detrended_luma_std':float(zz.std()),
           'dx_rms':float(np.sqrt(np.mean(gx*gx))),
           'dy_rms':float(np.sqrt(np.mean(gy*gy))),
           'gradient_tensor_anisotropy':float((eigen[-1]-eigen[0])/max(eigen.sum(),1e-12))}
    sigmas=[0,.7,1.4,2.8,5.6,11.2]
    blurs=[z]+[cv2.GaussianBlur(z,(0,0),s,borderType=cv2.BORDER_REFLECT101) for s in sigmas[1:]]
    stats['laplacian_rms']={f'{sigmas[i]}-{sigmas[i+1]}':float(np.sqrt(np.mean((blurs[i]-blurs[i+1])**2))) for i in range(len(blurs)-1)}
    window=np.hanning(z.shape[0])[:,None]*np.hanning(z.shape[1])[None,:]
    power=np.abs(np.fft.fft2(zz*window))**2/(np.sum(window*window)*z.size)
    fy=np.fft.fftfreq(z.shape[0])[:,None]; fx=np.fft.fftfreq(z.shape[1])[None,:]
    freq=np.sqrt(fx*fx+fy*fy)
    stats['radial_power']={}
    for lo,hi in ((2,4),(4,8),(8,16),(16,32),(32,64)):
        mask=(freq>=1/hi)&(freq<1/lo)
        p=float(power[mask].sum())
        stats['radial_power'][f'{lo}-{hi}px']={'power':p,'fraction':p/max(float(power.sum()),1e-12),'bins':int(mask.sum())}
    stats['axis_power']={}
    for axis,f in [('horizontal',np.abs(fx)),('vertical',np.abs(fy))]:
        stats['axis_power'][axis]={f'{lo}-{hi}px':float(power[(np.broadcast_to(f,power.shape)>=1/hi)&(np.broadcast_to(f,power.shape)<1/lo)].sum()) for lo,hi in ((2,4),(4,8),(8,16),(16,32))}
    return stats

def main():
    before={str(p):sha(p) for p in (RAW,SOURCE,PROFILE)}
    raw=np.array(Image.open(RAW).convert('RGB')); source=np.array(Image.open(SOURCE).convert('RGB'))
    base=raw.copy(); base[320:832]=source
    profile=np.array(Image.open(PROFILE).convert('RGB'))
    # All same-material paired diagnostic ROIs. No rail, text or person included.
    rois={
      'water48':{'source':(8,320,104,368),'generated':(8,272,104,320)},
      'water32':{'source':(8,320,136,352),'generated':(8,288,136,320)},
      'asphalt64':{'source':(8,768,136,832),'generated':(8,832,136,896)},
      'asphalt-right64':{'source':(368,768,496,832),'generated':(368,832,496,896)},
    }
    report={'purpose':'read-only same-material texture statistics; no success criterion',
            'source_rect':[0,320,512,832],'inputs_sha256':before,'rois':{},
            'spectral_notes':['Each ROI is independently plane detrended then Hann-windowed; power is luma^2.',
             'Laplacian uses reflect101 within each ROI; coarse levels have boundary contamination. They are diagnostic, not a gain recommendation.',
             'Radial PSD conflates orientation. Axis power and tensor anisotropy are included.',
             'Small 32/48px ROI height poorly resolves wavelengths above16px; no perceptual quality inferred from ratios.']}
    panels=[]
    for name,rects in rois.items():
        patches={}
        for side,rect in rects.items():
            x0,y0,x1,y1=rect; patches[side]=base[y0:y1,x0:x1]
        x0,y0,x1,y1=rects['generated']; patches['profile-generated']=profile[y0:y1,x0:x1]
        stats={k:statistics(v) for k,v in patches.items()}
        stats['generated_to_source_laplacian_gain_oracle']={k:stats['source']['laplacian_rms'][k]/max(stats['generated']['laplacian_rms'][k],1e-9) for k in stats['source']['laplacian_rms']}
        report['rois'][name]={'rects_xyxy':rects,'statistics':stats}
        # Save actual1x montage: rows source/generated/profile, no scaling.
        w=patches['source'].shape[1]; h=patches['source'].shape[0]
        panel=Image.new('RGB',(w+175,3*(h+22)),(25,25,25)); d=ImageDraw.Draw(panel)
        for i,(k,p) in enumerate(patches.items()):
            d.text((3,i*(h+22)+4),f'{name} / {k}',fill='white')
            panel.paste(Image.fromarray(p),(175,i*(h+22)))
        panels.append(panel)
    w=max(p.width for p in panels); h=sum(p.height+12 for p in panels)
    montage=Image.new('RGB',(w,h),(20,20,20)); y=0
    for p in panels: montage.paste(p,(0,y)); y+=p.height+12
    montage.save(OUT/'roi-comparison-1x.png')
    # Seam crop image remains actual1x; source paste is exact.
    Image.fromarray(base[272:384]).save(OUT/'top-boundary-1x.png')
    Image.fromarray(base[768:896]).save(OUT/'bottom-boundary-1x.png')
    report['input_hashes_unchanged']=all(sha(Path(p))==s for p,s in before.items())
    (OUT/'statistics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{'lap_gain':v['statistics']['generated_to_source_laplacian_gain_oracle'],
          'source_radial':v['statistics']['source']['radial_power'],
          'generated_radial':v['statistics']['generated']['radial_power'],
          'source_anisotropy':v['statistics']['source']['gradient_tensor_anisotropy'],
          'generated_anisotropy':v['statistics']['generated']['gradient_tensor_anisotropy']} for k,v in report['rois'].items()},indent=2))

if __name__=='__main__': main()
