// The registered original polygon is immutable evidence. This is a separate
// choice of which teacher-output pixels become the student's conditioning input.
const cropDrafts=new Map();
const cropPreviews=new Map();
const targetTrimDrafts=new Map();
function reviewTargetTrim(r){
  let choice=targetTrimDrafts.get(r.id+'-'+r.raw_sha256);
  if(!choice)try{choice=JSON.parse(localStorage.getItem('artworker-trim-'+r.id+'-'+r.raw_sha256)||'null')}catch{}
  if(!choice){const saved=trainingReviews[r.id];choice=saved?.raw_sha256===r.raw_sha256?saved.target_trim:null}
  return {top:choice?.top===true,bottom:choice?.bottom===true};
}
function reviewCropExtensions(r){
  let choice=cropDrafts.get(r.id+'-'+r.raw_sha256);
  if(!choice)try{choice=JSON.parse(localStorage.getItem('artworker-crop-'+r.id+'-'+r.raw_sha256)||'null')}catch{}
  if(!choice){const saved=trainingReviews[r.id];choice=saved?.raw_sha256===r.raw_sha256?saved.crop_extensions:null}
  return {top:choice?.top===true,bottom:choice?.bottom===true};
}
function trainingCrop(r,extensions=reviewCropExtensions(r),trim=reviewTargetTrim(r)){
  if(r.audit_status!=='reliable'||r.region?.polygon?.length!==4||r.dimensions?.length!==2)return null;
  const [w,h]=r.dimensions,points=r.region.polygon;
  if(!(w>0&&h>0)||!points.every(p=>p.length===2&&p.every(Number.isFinite)))return null;
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),clamp=(v,max)=>Math.max(0,Math.min(max,v));
  const base=[clamp(Math.floor(Math.min(...xs)),w),clamp(Math.floor(Math.min(...ys)),h),clamp(Math.ceil(Math.max(...xs)),w),clamp(Math.ceil(Math.max(...ys)),h)];
  const trimmed=trim.top||trim.bottom;
  const target=[trimmed?base[0]:0,trim.top?base[1]:0,trimmed?base[2]:w,trim.bottom?base[3]:h];
  const box=[clamp(Math.floor(Math.min(...xs)),w),extensions.top?0:clamp(Math.floor(Math.min(...ys)),h),clamp(Math.ceil(Math.max(...xs)),w),extensions.bottom?h:clamp(Math.ceil(Math.max(...ys)),h)];
  box[0]=Math.max(box[0],target[0]);box[1]=Math.max(box[1],target[1]);box[2]=Math.min(box[2],target[2]);box[3]=Math.min(box[3],target[3]);
  if(box[2]<=box[0]||box[3]<=box[1])return null;
  return {source:'teacher_output_crop',box_px:box,raw_dimensions:[w,h],target_box_px:target,target_dimensions:[target[2]-target[0],target[3]-target[1]],input_box_in_target_px:[box[0]-target[0],box[1]-target[1],box[2]-target[0],box[3]-target[1]],extensions,target_trim:trim,
    remaining_target_pixels:(target[2]-target[0])*(target[3]-target[1])-(box[2]-box[0])*(box[3]-box[1])};
}
function cropControls(r,prefix){
  const crop=trainingCrop(r),ext=reviewCropExtensions(r),trim=reviewTargetTrim(r);
  return `<section class="crop-controls"><strong>Eğitim girdisine dahil et</strong><div class="detail-controls">${[['top','Üstü de dahil et'],['bottom','Altı da dahil et']].map(([side,label])=>`<button id="${prefix}-crop-${side}" aria-pressed="${ext[side]}" class="${ext[side]?'active':''}" ${!crop||!reviewReady||trim[side]?'disabled':''}>${ext[side]?'✓ ':''}${label}</button>`).join('')}</div><strong>Kötü genişlemeyi eğitim hedefinden çıkar</strong><div class="detail-controls">${[['top','Üstü sil · sadece alta genişlet'],['bottom','Altı sil · sadece üste genişlet']].map(([side,label])=>`<button id="${prefix}-trim-${side}" aria-pressed="${trim[side]}" class="trim-control ${trim[side]?'active':''}" ${!crop||!reviewReady?'disabled':''}>${trim[side]?'✓ ':''}${label}</button>`).join('')}</div><p class="subtle" id="${prefix}-crop-info">${cropInfo(crop)}</p></section>`;
}
function cropInfo(crop,preview=false){
  if(!crop)return 'Güvenilir kaynak alanı bulunamadı; eğitim girdisi sınırı seçilemiyor.';
  const [x0,y0,x1,y1]=crop.box_px;
  return `${x1-x0} × ${y1-y0} px girdi → ${crop.target_dimensions.join(' × ')} px hedef · turuncu: kaynak, ${preview?'mor: önizlenen girdi':'mavi: eğitim girdisi'}, yeşil: kalan hedef, kırmızı: çıkarılan alan.${crop.remaining_target_pixels===0?' Doldurulacak dış alan kalmadı; bu seçim eğitim dışında bekletilir.':''}`;
}
function targetTrimMarks(r,crop=trainingCrop(r)){
  if(!crop||!Object.values(crop.target_trim).some(Boolean))return '';
  const [w,h]=r.dimensions,[l,t,right,b]=crop.target_box_px;
  return [[0,0,w,t],[0,b,w,h-b],[0,t,l,b-t],[right,t,w-right,b-t]].filter(a=>a[2]>0&&a[3]>0).map(([x,y,width,height])=>`<rect class="trim-discard" x="${x}" y="${y}" width="${width}" height="${height}"/>`).join('')+`<polygon class="trim-target" points="${[[l,t],[right,t],[right,b],[l,b]].map(p=>p.join(',')).join(' ')}"/>`;
}
function trainingCropPoints(r,extensions=reviewCropExtensions(r),trim=reviewTargetTrim(r)){
  const crop=trainingCrop(r,extensions,trim);if(!crop)return '';
  const [l,t,right,b]=crop.box_px;return [[l,t],[right,t],[right,b],[l,b]].map(p=>p.join(',')).join(' ');
}
function previewTrainingExtent(r,side,prefix,type='crop'){
  if(!reviewReady||reviewBusy||fastReview.busy||!trainingCrop(r))return;
  if(type==='crop'&&reviewTargetTrim(r)[side])return;
  const overlay=$(prefix==='fast'?'#fast-boundary':'#boundary');
  const polygon=$('#'+prefix+'-training-crop'),info=$('#'+prefix+'-crop-info');
  if(!overlay||!polygon||!info)return;
  const ext=reviewCropExtensions(r),trim=reviewTargetTrim(r);
  if(type==='trim'){trim[side]=!trim[side];if(trim[side])ext[side]=false}
  else ext[side]=!ext[side];
  const crop=trainingCrop(r,ext,trim);if(!crop)return;
  cropPreviews.set(prefix,{id:r.id,sha:r.raw_sha256,overlay,display:overlay.style.display});
  overlay.style.display='block';
  polygon.setAttribute('points',trainingCropPoints(r,ext,trim));
  const marks=$('#'+prefix+'-trim-zones');if(marks)marks.innerHTML=targetTrimMarks(r,crop);
  polygon.classList.toggle('crop-preview',true);
  info.classList.toggle('crop-preview-label',true);
  info.textContent='ÖNİZLEME · tıklarsan '+(side==='top'?'üst taraf ':'alt taraf ')+(type==='trim'?(trim[side]?'eğitim hedefinden çıkarılacak. ':'hedefe geri alınacak. '):(ext[side]?'girdiye dahil edilecek. ':'girdiden çıkarılacak. '))+cropInfo(crop,true)+' Henüz kaydedilmedi.';
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
  const trim=reviewTargetTrim(r);
  for(const [side,label] of [['top','Üstü sil · sadece alta genişlet'],['bottom','Altı sil · sadece üste genişlet']]){
    const button=$('#'+prefix+'-trim-'+side);if(button){button.setAttribute('aria-pressed',String(trim[side]));button.classList.toggle('active',trim[side]);button.textContent=(trim[side]?'✓ ':'')+label}
  }
  const marks=$('#'+prefix+'-trim-zones');if(marks)marks.innerHTML=targetTrimMarks(r);
}
async function toggleTrainingExtent(r,side,prefix){
  if(!reviewReady||reviewBusy||fastReview.busy||!trainingCrop(r)||reviewTargetTrim(r)[side])return;
  const ext=reviewCropExtensions(r);ext[side]=!ext[side];
  return saveTrainingGeometry(r,prefix,ext,reviewTargetTrim(r));
}
async function toggleTargetTrim(r,side,prefix){
  if(!reviewReady||reviewBusy||fastReview.busy||!trainingCrop(r))return;
  const ext=reviewCropExtensions(r),trim=reviewTargetTrim(r);trim[side]=!trim[side];
  if(trim[side])ext[side]=false;
  return saveTrainingGeometry(r,prefix,ext,trim);
}
async function saveTrainingGeometry(r,prefix,ext,trim){
  cancelNoteAutosave();
  cropDrafts.set(r.id+'-'+r.raw_sha256,ext);
  targetTrimDrafts.set(r.id+'-'+r.raw_sha256,trim);
  try{localStorage.setItem('artworker-crop-'+r.id+'-'+r.raw_sha256,JSON.stringify(ext))}catch{}
  try{localStorage.setItem('artworker-trim-'+r.id+'-'+r.raw_sha256,JSON.stringify(trim))}catch{}
  const fast=prefix==='fast';
  const reason=fast?$('#fast-note').value:$('#review-reason').value;
  if(fast){rememberFastNote();fastReview.busy=true;fastReview.boundary=true;$('#fast-boundary').style.display='block';$('#fast-region-toggle').setAttribute('aria-pressed','true');$('#fast-region-toggle').textContent='Kaynak bölgesini gizle [Space]';fastButtons()}
  else{reviewBusy=true;if($('#boundary'))$('#boundary').style.display='';if($('#boundary-toggle'))$('#boundary-toggle').checked=true}
  updateCropView(r,prefix);
  for(const s of ['top','bottom'])for(const type of ['crop','trim'])$('#'+prefix+'-'+type+'-'+s).disabled=true;
  try{
    await persistReview(r,()=>reviewDecision(r),reason,ext,trim);
    render();
  }catch(error){autosaveStatus(error.message,true)}
  finally{
    if(fast){fastReview.busy=false;fastButtons()}else reviewBusy=false;
    for(const s of ['top','bottom']){ $('#'+prefix+'-crop-'+s).disabled=!reviewReady||trim[s];$('#'+prefix+'-trim-'+s).disabled=!reviewReady; }
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
    const remove=$('#'+prefix+'-trim-'+side);
    remove.setAttribute('aria-describedby',prefix+'-crop-info');
    remove.onmouseenter=()=>previewTrainingExtent(r,side,prefix,'trim');
    remove.onfocus=()=>previewTrainingExtent(r,side,prefix,'trim');
    remove.onmouseleave=()=>clearTrainingPreview(r,prefix);
    remove.onblur=()=>clearTrainingPreview(r,prefix);
    remove.onclick=()=>{clearTrainingPreview(r,prefix);toggleTargetTrim(r,side,prefix)};
  }
}
