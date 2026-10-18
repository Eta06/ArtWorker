// The registered original polygon is immutable evidence. This is a separate
// choice of which teacher-output pixels become the student's conditioning input.
const cropDrafts=new Map();
const cropPreviews=new Map();
function reviewCropExtensions(r){
  let choice=cropDrafts.get(r.id+'-'+r.raw_sha256);
  if(!choice)try{choice=JSON.parse(localStorage.getItem('artworker-crop-'+r.id+'-'+r.raw_sha256)||'null')}catch{}
  if(!choice){const saved=trainingReviews[r.id];choice=saved?.raw_sha256===r.raw_sha256?saved.crop_extensions:null}
  return {top:choice?.top===true,bottom:choice?.bottom===true};
}
function trainingCrop(r,extensions=reviewCropExtensions(r)){
  if(r.audit_status!=='reliable'||r.region?.polygon?.length!==4||r.dimensions?.length!==2)return null;
  const [w,h]=r.dimensions,points=r.region.polygon;
  if(!(w>0&&h>0)||!points.every(p=>p.length===2&&p.every(Number.isFinite)))return null;
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),clamp=(v,max)=>Math.max(0,Math.min(max,v));
  const box=[clamp(Math.floor(Math.min(...xs)),w),extensions.top?0:clamp(Math.floor(Math.min(...ys)),h),clamp(Math.ceil(Math.max(...xs)),w),extensions.bottom?h:clamp(Math.ceil(Math.max(...ys)),h)];
  if(box[2]<=box[0]||box[3]<=box[1])return null;
  return {source:'teacher_output_crop',box_px:box,target_dimensions:[w,h],extensions,
    remaining_target_pixels:w*h-(box[2]-box[0])*(box[3]-box[1])};
}
function cropControls(r,prefix){
  const crop=trainingCrop(r),ext=reviewCropExtensions(r);
  return `<section class="crop-controls"><strong>Eğitim girdisine dahil et</strong><div class="detail-controls">${[['top','Üstü de dahil et'],['bottom','Altı da dahil et']].map(([side,label])=>`<button id="${prefix}-crop-${side}" aria-pressed="${ext[side]}" class="${ext[side]?'active':''}" ${!crop||!reviewReady?'disabled':''}>${ext[side]?'✓ ':''}${label}</button>`).join('')}</div><p class="subtle" id="${prefix}-crop-info">${cropInfo(crop)}</p></section>`;
}
function cropInfo(crop,preview=false){
  if(!crop)return 'Güvenilir kaynak alanı bulunamadı; eğitim girdisi sınırı seçilemiyor.';
  const [x0,y0,x1,y1]=crop.box_px;
  return `${x1-x0} × ${y1-y0} px eğitim girdisi · turuncu: bulunan kaynak, ${preview?'mor: tıklama sonrası kesilecek alan':'mavi: teacher çıktısından kesilecek alan'}.${crop.remaining_target_pixels===0?' Doldurulacak dış alan kalmadı; bu seçim eğitim dışında bekletilir.':''}`;
}
function trainingCropPoints(r,extensions=reviewCropExtensions(r)){
  const crop=trainingCrop(r,extensions);if(!crop)return '';
  const [l,t,right,b]=crop.box_px;return [[l,t],[right,t],[right,b],[l,b]].map(p=>p.join(',')).join(' ');
}
function previewTrainingExtent(r,side,prefix){
  if(!reviewReady||reviewBusy||fastReview.busy||!trainingCrop(r))return;
  const overlay=$(prefix==='fast'?'#fast-boundary':'#boundary');
  const polygon=$('#'+prefix+'-training-crop'),info=$('#'+prefix+'-crop-info');
  if(!overlay||!polygon||!info)return;
  const ext=reviewCropExtensions(r);ext[side]=!ext[side];
  const crop=trainingCrop(r,ext);if(!crop)return;
  cropPreviews.set(prefix,{id:r.id,sha:r.raw_sha256,overlay,display:overlay.style.display});
  overlay.style.display='block';
  polygon.setAttribute('points',trainingCropPoints(r,ext));
  polygon.classList.toggle('crop-preview',true);
  info.classList.toggle('crop-preview-label',true);
  info.textContent='ÖNİZLEME · tıklarsan '+(side==='top'?'üst taraf ':'alt taraf ')+(ext[side]?'dahil edilecek. ':'çıkarılacak. ')+cropInfo(crop,true)+' Henüz kaydedilmedi.';
}
function clearTrainingPreview(r,prefix){
  const preview=cropPreviews.get(prefix);
  if(!preview||preview.id!==r.id||preview.sha!==r.raw_sha256)return;
  cropPreviews.delete(prefix);
  const overlay=$(prefix==='fast'?'#fast-boundary':'#boundary');
  if(overlay!==preview.overlay)return; // The user may already be on another cover.
  updateCropView(r,prefix);
  $('#'+prefix+'-training-crop')?.classList.toggle('crop-preview',false);
  $('#'+prefix+'-crop-info')?.classList.toggle('crop-preview-label',false);
  const checkbox=$('#boundary-toggle');
  overlay.style.display=prefix==='fast'?(fastReview.boundary?'block':'none'):(checkbox?(checkbox.checked?'':'none'):preview.display);
}
function updateCropView(r,prefix){
  const ext=reviewCropExtensions(r),crop=trainingCrop(r);
  for(const [side,label] of [['top','Üstü de dahil et'],['bottom','Altı da dahil et']]){
    const button=$('#'+prefix+'-crop-'+side);if(!button)continue;
    button.setAttribute('aria-pressed',String(ext[side]));button.classList.toggle('active',ext[side]);button.textContent=(ext[side]?'✓ ':'')+label;
  }
  const polygon=$('#'+prefix+'-training-crop');if(polygon)polygon.setAttribute('points',trainingCropPoints(r));
  const info=$('#'+prefix+'-crop-info');if(info)info.textContent=cropInfo(crop);
}
async function toggleTrainingExtent(r,side,prefix){
  if(!reviewReady||reviewBusy||fastReview.busy||!trainingCrop(r))return;
  cancelNoteAutosave();
  const ext=reviewCropExtensions(r);ext[side]=!ext[side];
  cropDrafts.set(r.id+'-'+r.raw_sha256,ext);
  try{localStorage.setItem('artworker-crop-'+r.id+'-'+r.raw_sha256,JSON.stringify(ext))}catch{}
  const fast=prefix==='fast';
  const reason=fast?$('#fast-note').value:$('#review-reason').value;
  if(fast){rememberFastNote();fastReview.busy=true;fastReview.boundary=true;$('#fast-boundary').style.display='block';$('#fast-region-toggle').setAttribute('aria-pressed','true');$('#fast-region-toggle').textContent='Kaynak bölgesini gizle [Space]';fastButtons()}
  else{reviewBusy=true;if($('#boundary'))$('#boundary').style.display='';if($('#boundary-toggle'))$('#boundary-toggle').checked=true}
  updateCropView(r,prefix);
  for(const s of ['top','bottom'])$('#'+prefix+'-crop-'+s).disabled=true;
  try{
    await persistReview(r,()=>reviewDecision(r),reason,ext);
    render();
  }catch(error){autosaveStatus(error.message,true)}
  finally{
    if(fast){fastReview.busy=false;fastButtons()}else reviewBusy=false;
    for(const s of ['top','bottom'])$('#'+prefix+'-crop-'+s).disabled=!reviewReady;
    if(fast&&$('#fast-dialog').open)$('#fast-content').focus({preventScroll:true});
  }
}
function bindCropControls(r,prefix){
  cropPreviews.delete(prefix);
  for(const side of ['top','bottom']){
    const button=$('#'+prefix+'-crop-'+side);
    button.setAttribute('aria-describedby',prefix+'-crop-info');
    button.onmouseenter=()=>previewTrainingExtent(r,side,prefix);
    button.onfocus=()=>previewTrainingExtent(r,side,prefix);
    button.onmouseleave=()=>clearTrainingPreview(r,prefix);
    button.onblur=()=>clearTrainingPreview(r,prefix);
    button.onclick=()=>{clearTrainingPreview(r,prefix);toggleTrainingExtent(r,side,prefix)};
  }
}
