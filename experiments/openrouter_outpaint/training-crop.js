// The registered original polygon is immutable evidence. This is a separate
// choice of which teacher-output pixels become the student's conditioning input.
const cropDrafts=new Map();
const cropPreviews=new Map();
const targetTrimDrafts=new Map();
const squareFitDrafts=new Map();
function reviewSquareFit(r){
  const key=r.id+'-'+r.raw_sha256;let value=squareFitDrafts.get(key);
  if(value===undefined)try{value=JSON.parse(localStorage.getItem('artworker-square-'+key)||'null')}catch{}
  if(typeof value!=='boolean'){const saved=trainingReviews[r.id];value=saved?.raw_sha256===r.raw_sha256?saved.square_fit:false}
  return value===true;
}
function cropSideLabels(r){return (r.dimensions?.[0]>r.dimensions?.[1])?
  [['left','Solu da dahil et','Solu sil · sadece sağa genişlet'],['right','Sağı da dahil et','Sağı sil · sadece sola genişlet']]:
  [['top','Üstü de dahil et','Üstü sil · sadece alta genişlet'],['bottom','Altı da dahil et','Altı sil · sadece üste genişlet']];}
function reviewTargetTrim(r){
  let choice=targetTrimDrafts.get(r.id+'-'+r.raw_sha256);
  if(!choice)try{choice=JSON.parse(localStorage.getItem('artworker-trim-'+r.id+'-'+r.raw_sha256)||'null')}catch{}
  if(!choice){const saved=trainingReviews[r.id];choice=saved?.raw_sha256===r.raw_sha256?saved.target_trim:null}
  const sides={top:choice?.top===true,bottom:choice?.bottom===true};
  if(r.dimensions?.[0]>r.dimensions?.[1]||choice?.left!==undefined||choice?.right!==undefined)Object.assign(sides,{left:choice?.left===true,right:choice?.right===true});
  return sides;
}
function reviewCropExtensions(r){
  let choice=cropDrafts.get(r.id+'-'+r.raw_sha256);
  if(!choice)try{choice=JSON.parse(localStorage.getItem('artworker-crop-'+r.id+'-'+r.raw_sha256)||'null')}catch{}
  if(!choice){const saved=trainingReviews[r.id];choice=saved?.raw_sha256===r.raw_sha256?saved.crop_extensions:null}
  const sides={top:choice?.top===true,bottom:choice?.bottom===true};
  if(r.dimensions?.[0]>r.dimensions?.[1]||choice?.left!==undefined||choice?.right!==undefined)Object.assign(sides,{left:choice?.left===true,right:choice?.right===true});
  return sides;
}
function referenceBox(r,squareFit=false){
  if(!registrationIsReady(r)||r.region?.polygon?.length!==4||r.dimensions?.length!==2)return null;
  const [w,h]=r.dimensions,points=r.region.polygon;
  if(!(w>0&&h>0)||!points.every(p=>p.length===2&&p.every(Number.isFinite)))return null;
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),clamp=(v,max)=>Math.max(0,Math.min(max,v));
  const base=[clamp(Math.floor(Math.min(...xs)),w),clamp(Math.floor(Math.min(...ys)),h),clamp(Math.ceil(Math.max(...xs)),w),clamp(Math.ceil(Math.max(...ys)),h)];
  if(base[2]<=base[0]||base[3]<=base[1])return null;
  if(squareFit){
    const side=Math.min(w,h);if(base[2]-base[0]>side||base[3]-base[1]>side)return null;
    const x=Math.max(Math.max(0,base[2]-side),Math.min(Math.min(w-side,base[0]),Math.floor((base[0]+base[2]-side)/2)));
    const y=Math.max(Math.max(0,base[3]-side),Math.min(Math.min(h-side,base[1]),Math.floor((base[1]+base[3]-side)/2)));
    return [x,y,x+side,y+side];
  }
  return base;
}
function referenceRegionPoints(r,squareFit=reviewSquareFit(r)){
  const box=squareFit?referenceBox(r,true):null;
  return (box?[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]:r.region?.polygon||[]).map(p=>p.join(',')).join(' ');
}
function trainingCrop(r,extensions=reviewCropExtensions(r),trim=reviewTargetTrim(r),squareFit=reviewSquareFit(r)){
  const base=referenceBox(r,squareFit);if(!base)return null;
  const [w,h]=r.dimensions;
  const vertical=trim.top||trim.bottom,horizontal=trim.left||trim.right;
  const target=[trim.left||(vertical&&!horizontal)?base[0]:0,trim.top||(horizontal&&!vertical)?base[1]:0,
    trim.right||(vertical&&!horizontal)?base[2]:w,trim.bottom||(horizontal&&!vertical)?base[3]:h];
  const box=[extensions.left?0:base[0],extensions.top?0:base[1],extensions.right?w:base[2],extensions.bottom?h:base[3]];
  box[0]=Math.max(box[0],target[0]);box[1]=Math.max(box[1],target[1]);box[2]=Math.min(box[2],target[2]);box[3]=Math.min(box[3],target[3]);
  if(box[2]<=box[0]||box[3]<=box[1])return null;
  return {source:'teacher_output_crop',box_px:box,square_fit:squareFit,reference_box_px:base,raw_dimensions:[w,h],target_box_px:target,target_dimensions:[target[2]-target[0],target[3]-target[1]],input_box_in_target_px:[box[0]-target[0],box[1]-target[1],box[2]-target[0],box[3]-target[1]],extensions,target_trim:trim,
    remaining_target_pixels:(target[2]-target[0])*(target[3]-target[1])-(box[2]-box[0])*(box[3]-box[1])};
}
function cropControls(r,prefix){
  const crop=trainingCrop(r),ext=reviewCropExtensions(r),trim=reviewTargetTrim(r);
  return `<section class="crop-controls"><div class="detail-controls"><button id="${prefix}-square-fit" aria-pressed="${reviewSquareFit(r)}" class="${reviewSquareFit(r)?'active':''}" ${!reviewReady||(!reviewSquareFit(r)&&!referenceBox(r,true))?'disabled':''} title="${referenceBox(r,true)?'Referansı kapsayan ve görselin kısa kenarını dolduran kareyi kullan':'Kaynak alanı kırpılmadan bu kareye sığmıyor'}">${reviewSquareFit(r)?'✓ ':''}Kareyi kenarlara oturt</button></div><strong>Eğitim girdisine dahil et · ${r.dimensions?.[0]>r.dimensions?.[1]?'yatay / sol–sağ':'dikey / üst–alt'}</strong><div class="detail-controls">${cropSideLabels(r).map(([side,label])=>`<button id="${prefix}-crop-${side}" aria-pressed="${ext[side]}" class="${ext[side]?'active':''}" ${!crop||!reviewReady||trim[side]?'disabled':''}>${ext[side]?'✓ ':''}${label}</button>`).join('')}</div><strong>Kötü genişlemeyi eğitim hedefinden çıkar</strong><div class="detail-controls">${cropSideLabels(r).map(([side,_,label])=>`<button id="${prefix}-trim-${side}" aria-pressed="${trim[side]}" class="trim-control ${trim[side]?'active':''}" ${!crop||!reviewReady?'disabled':''}>${trim[side]?'✓ ':''}${label}</button>`).join('')}</div><p class="subtle" id="${prefix}-crop-info">${cropInfo(crop)}</p></section>`;
}
function cropInfo(crop,preview=false){
  if(!crop)return 'Güvenilir kaynak alanı bulunamadı; eğitim girdisi sınırı seçilemiyor.';
  const [x0,y0,x1,y1]=crop.box_px;
  return `${crop.square_fit?'Kare referans: '+(crop.reference_box_px[2]-crop.reference_box_px[0])+' px. ':''}${x1-x0} × ${y1-y0} px girdi → ${crop.target_dimensions.join(' × ')} px hedef · turuncu: ${crop.square_fit?'kare referans':'kaynak'}, ${preview?'mor: önizlenen girdi':'mavi: eğitim girdisi'}, yeşil: kalan hedef, kırmızı: çıkarılan alan.${crop.remaining_target_pixels===0?' Doldurulacak dış alan kalmadı; bu seçim eğitim dışında bekletilir.':''}`;
}
function targetTrimMarks(r,crop=trainingCrop(r)){
  if(!crop||!Object.values(crop.target_trim).some(Boolean))return '';
  const [w,h]=r.dimensions,[l,t,right,b]=crop.target_box_px;
  return [[0,0,w,t],[0,b,w,h-b],[0,t,l,b-t],[right,t,w-right,b-t]].filter(a=>a[2]>0&&a[3]>0).map(([x,y,width,height])=>`<rect class="trim-discard" x="${x}" y="${y}" width="${width}" height="${height}"/>`).join('')+`<polygon class="trim-target" points="${[[l,t],[right,t],[right,b],[l,b]].map(p=>p.join(',')).join(' ')}"/>`;
}
function trainingCropPoints(r,extensions=reviewCropExtensions(r),trim=reviewTargetTrim(r),squareFit=reviewSquareFit(r)){
  const crop=trainingCrop(r,extensions,trim,squareFit);if(!crop)return '';
  const [l,t,right,b]=crop.box_px;return [[l,t],[right,t],[right,b],[l,b]].map(p=>p.join(',')).join(' ');
}
function previewTrainingExtent(r,side,prefix,type='crop'){
  if(!reviewReady||reviewBusy||fastReview.busy||!trainingCrop(r))return;
  if(type==='crop'&&reviewTargetTrim(r)[side])return;
  const overlay=$(prefix==='fast'?'#fast-boundary':'#boundary');
  const polygon=$('#'+prefix+'-training-crop'),info=$('#'+prefix+'-crop-info');
  if(!overlay||!polygon||!info)return;
  const ext=reviewCropExtensions(r),trim=reviewTargetTrim(r);
  const squareFit=type==='square'?!reviewSquareFit(r):reviewSquareFit(r);
  if(type==='trim'){trim[side]=!trim[side];if(trim[side])ext[side]=false}
  else if(type==='crop')ext[side]=!ext[side];
  const crop=trainingCrop(r,ext,trim,squareFit);if(!crop)return;
  cropPreviews.set(prefix,{id:r.id,sha:r.raw_sha256,overlay,display:overlay.style.display});
  overlay.style.display='block';
  polygon.setAttribute('points',trainingCropPoints(r,ext,trim,squareFit));
  $('#'+prefix+'-source-boundary')?.setAttribute('points',referenceRegionPoints(r,squareFit));
  const marks=$('#'+prefix+'-trim-zones');if(marks)marks.innerHTML=targetTrimMarks(r,crop);
  polygon.classList.toggle('crop-preview',true);
  info.classList.toggle('crop-preview-label',true);
  if(type==='square'){info.textContent='ÖNİZLEME · '+(squareFit?'Referans kısa kenarı dolduran kareye oturacak. ':'Tespit edilen kaynak alanına dönülecek. ')+cropInfo(crop,true)+' Henüz kaydedilmedi.';return}
  info.textContent='ÖNİZLEME · tıklarsan '+({top:'üst taraf ',bottom:'alt taraf ',left:'sol taraf ',right:'sağ taraf '}[side])+(type==='trim'?(trim[side]?'eğitim hedefinden çıkarılacak. ':'hedefe geri alınacak. '):(ext[side]?'girdiye dahil edilecek. ':'girdiden çıkarılacak. '))+cropInfo(crop,true)+' Henüz kaydedilmedi.';
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
  const square=reviewSquareFit(r),squareButton=$('#'+prefix+'-square-fit');
  if(squareButton){squareButton.setAttribute('aria-pressed',String(square));squareButton.classList.toggle('active',square);squareButton.textContent=(square?'✓ ':'')+'Kareyi kenarlara oturt'}
  $('#'+prefix+'-source-boundary')?.setAttribute('points',referenceRegionPoints(r));
  for(const [side,label] of cropSideLabels(r)){
    const button=$('#'+prefix+'-crop-'+side);if(!button)continue;
    button.setAttribute('aria-pressed',String(ext[side]));button.classList.toggle('active',ext[side]);button.textContent=(ext[side]?'✓ ':'')+label;
  }
  const polygon=$('#'+prefix+'-training-crop');if(polygon)polygon.setAttribute('points',trainingCropPoints(r));
  const info=$('#'+prefix+'-crop-info');if(info)info.textContent=cropInfo(crop);
  const trim=reviewTargetTrim(r);
  for(const [side,_,label] of cropSideLabels(r)){
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
async function toggleSquareFit(r,prefix){
  if(!reviewReady||reviewBusy||fastReview.busy)return;
  const squareFit=!reviewSquareFit(r);if(!referenceBox(r,squareFit))return;
  return saveTrainingGeometry(r,prefix,reviewCropExtensions(r),reviewTargetTrim(r),squareFit);
}
async function saveTrainingGeometry(r,prefix,ext,trim,squareFit=reviewSquareFit(r)){
  cancelNoteAutosave();
  cropDrafts.set(r.id+'-'+r.raw_sha256,ext);
  targetTrimDrafts.set(r.id+'-'+r.raw_sha256,trim);
  squareFitDrafts.set(r.id+'-'+r.raw_sha256,squareFit);
  try{localStorage.setItem('artworker-crop-'+r.id+'-'+r.raw_sha256,JSON.stringify(ext))}catch{}
  try{localStorage.setItem('artworker-trim-'+r.id+'-'+r.raw_sha256,JSON.stringify(trim))}catch{}
  try{localStorage.setItem('artworker-square-'+r.id+'-'+r.raw_sha256,JSON.stringify(squareFit))}catch{}
  const fast=prefix==='fast';
  const reason=fast?$('#fast-note').value:$('#review-reason').value;
  if(fast){rememberFastNote();fastReview.busy=true;fastReview.boundary=true;$('#fast-boundary').style.display='block';$('#fast-region-toggle').setAttribute('aria-pressed','true');$('#fast-region-toggle').textContent='Kaynak bölgesini gizle [Space]';fastButtons()}
  else{reviewBusy=true;if($('#boundary'))$('#boundary').style.display='';if($('#boundary-toggle'))$('#boundary-toggle').checked=true}
  updateCropView(r,prefix);
  $('#'+prefix+'-square-fit').disabled=true;
  for(const [s] of cropSideLabels(r))for(const type of ['crop','trim'])$('#'+prefix+'-'+type+'-'+s).disabled=true;
  try{
    await persistReview(r,()=>reviewDecision(r),reason,ext,trim,undefined,squareFit);
    render();
  }catch(error){autosaveStatus(error.message,true)}
  finally{
    if(fast){fastReview.busy=false;fastButtons()}else reviewBusy=false;
    for(const [s] of cropSideLabels(r)){ $('#'+prefix+'-crop-'+s).disabled=!reviewReady||trim[s];$('#'+prefix+'-trim-'+s).disabled=!reviewReady; }
    $('#'+prefix+'-square-fit').disabled=!reviewReady||(!reviewSquareFit(r)&&!referenceBox(r,true));
    if(fast&&$('#fast-dialog').open)$('#fast-content').focus({preventScroll:true});
  }
}
function bindCropControls(r,prefix){
  cropPreviews.delete(prefix);
  const square=$('#'+prefix+'-square-fit');
  square.setAttribute('aria-describedby',prefix+'-crop-info');
  square.onmouseenter=()=>previewTrainingExtent(r,'square',prefix,'square');
  square.onfocus=()=>previewTrainingExtent(r,'square',prefix,'square');
  square.onmouseleave=()=>clearTrainingPreview(r,prefix);
  square.onblur=()=>clearTrainingPreview(r,prefix);
  square.onclick=()=>{clearTrainingPreview(r,prefix);toggleSquareFit(r,prefix)};
  for(const [side] of cropSideLabels(r)){
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
