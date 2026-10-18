// A frozen queue avoids reordering/skipping cards when filters change after a save.
const fastReview={rows:[],index:0,busy:false,ready:false,imageError:false,token:0,drafts:new Map(),boundary:false};
function persistFastSession(){
  try{localStorage.setItem('artworker-fast-session-v2',JSON.stringify({rows:fastReview.rows.map(r=>({id:r.id,sha:r.raw_sha256})),index:fastReview.index,drafts:[...fastReview.drafts],filters:fastFilters()}))}catch{}
}
function fastFilters(){return ['search','model','audit','review-filter','sort'].map(id=>$('#'+id)?.value||'').join('|')}
function restoreFastSession(){
  try{
    const saved=JSON.parse(localStorage.getItem('artworker-fast-session-v2')||'null');
    if(!saved||saved.filters!==fastFilters()||saved.index>=saved.rows.length)return false;
    const rows=saved.rows.map(key=>data.rows.find(r=>r.id===key.id&&r.raw_sha256===key.sha));
    if(rows.some(r=>!r))return false;
    fastReview.rows=rows;fastReview.index=saved.index;fastReview.drafts=new Map(saved.drafts);
    return true;
  }catch{return false}
}
function rememberFastNote() {
  const row=fastReview.rows[fastReview.index], input=$('#fast-note');
  if(row&&input){fastReview.drafts.set(row.id+'-'+row.raw_sha256,input.value);stashReviewNote(row,input.value)}
  persistFastSession();
}
function fastQueue(rows) {
  rememberFastNote();
  fastReview.rows=rows.slice();fastReview.index=0;
  persistFastSession();
  if($('#dialog').open)$('#dialog').close();
  renderFastReview();
  if(!$('#fast-dialog').open)$('#fast-dialog').showModal();
  $('#fast-content').focus({preventScroll:true});
}
function startFastReview() {
  if(!reviewReady||reviewBusy||fastReview.busy){
    $('#review-connection').textContent='Önce karar kayıt bağlantısının hazır olmasını bekle; gerekirse Yenile’ye bas.';return;
  }
  // Default: resume unreviewed. Explicit accepted/rejected filters allow re-review.
  const explicit=['accepted','rejected'].includes($('#review-filter').value);
  if(restoreFastSession()){
    if($('#dialog').open)$('#dialog').close();
    renderFastReview();$('#fast-dialog').showModal();$('#fast-content').focus({preventScroll:true});return;
  }
  fastQueue(selected().filter(row=>explicit||reviewDecision(row)==='pending'));
}
function fastButtons() {
  document.querySelectorAll('[data-fast-decision]').forEach(b=>{
    b.disabled=fastReview.busy||!reviewReady||(!fastReview.ready&&!(fastReview.imageError&&b.dataset.fastDecision==='pending'));
  });
  if($('#fast-previous'))$('#fast-previous').disabled=fastReview.busy||fastReview.index===0;
  if($('#fast-close'))$('#fast-close').disabled=fastReview.busy;
  if($('#fast-detail'))$('#fast-detail').disabled=fastReview.busy;
  if($('#fast-note'))$('#fast-note').disabled=fastReview.busy;
  const row=fastReview.rows[fastReview.index];
  for(const side of ['top','bottom']){
    const button=$('#fast-crop-'+side);if(button)button.disabled=fastReview.busy||!reviewReady||!row||!trainingCrop(row);
  }
}
function renderFastReview() {
  persistFastSession();
  const r=fastReview.rows[fastReview.index], token=++fastReview.token;
  fastReview.ready=false;fastReview.imageError=false;fastReview.boundary=false;
  if(!r){
    const skipped=fastReview.rows.filter(row=>reviewDecision(row)==='pending');
    $('#fast-content').innerHTML=`<div class="fast-finished"><div class="eyebrow">Hızlı inceleme</div><h2 id="fast-title">${fastReview.rows.length?'Bu tur tamamlandı.':'Bu filtrede incelenecek görsel yok.'}</h2><p>Kararlar dosyaya kaydedildi. ${skipped.length} görsel emin olunmadığı için eğitim dışında bekliyor.</p><div class="detail-controls"><button id="fast-retry" ${skipped.length?'':'disabled'}>Atladıklarımı tekrar göster</button><button id="fast-close">Galeriye dön</button><button id="fast-export">Kararları yedekle</button><a href="training-manifest" download="artworker-egitim-adaylari.json">Son eğitim adayı listesini indir</a></div></div>`;
    $('#fast-close').onclick=()=>$('#fast-dialog').close();
    $('#fast-retry').onclick=()=>fastQueue(skipped);
    $('#fast-export').onclick=exportReviews;return;
  }
  const saved=trainingReviews[r.id],key=r.id+'-'+r.raw_sha256;
  const note=reviewNote(r);
  $('#fast-content').innerHTML=`<div class="fast-heading"><div><span class="eyebrow">Hızlı inceleme · ${fastReview.index+1} / ${fastReview.rows.length}</span>${modelBanner(r)}<h2 id="fast-title">${esc(r.title)}</h2><p>${esc(r.artists.join(' · '))} · ${r.model==='sunburst'?'Sunburst 2.5':'Nano Banana 2.1'} ${reviewBadge(r)}</p></div><button id="fast-close" aria-label="Hızlı incelemeyi kapat">✕</button></div><div class="fast-pair"><figure><figcaption>ÖNCE · ORİJİNAL</figcaption><div class="fast-image"><img id="fast-before" src="${r.asset}${r.original_file}" alt="${esc(r.title)} orijinal kapak" decoding="async"></div></figure><figure><figcaption>SONRA · GENİŞLETİLMİŞ</figcaption><div class="fast-image fast-output"><img id="fast-after" src="${r.asset}raw.png" alt="${esc(r.title)} genişletilmiş kapak" decoding="async">${fastRegion(r)?`<svg id="fast-boundary" style="display:none" viewBox="0 0 ${r.dimensions.join(' ')}" preserveAspectRatio="xMidYMid meet" aria-hidden="true"><polygon points="${r.region.polygon.map(p=>p.join(',')).join(' ')}"/><polygon id="fast-training-crop" class="training-crop-outline" points="${trainingCropPoints(r)}"/></svg>`:''}</div></figure></div><div class="fast-region-control"><button id="fast-region-toggle" aria-pressed="false" ${fastRegion(r)?'':'disabled'}>Kaynak bölgesini göster <kbd>Space</kbd></button>${fastRegion(r)?'':'<span class="subtle">Güvenilir kaynak bölgesi bulunamadı</span>'}</div>${cropControls(r,'fast')}<div class="fast-actions"><p id="fast-save-status" class="autosave-inline" role="status">${esc(autosaveMessage)}</p><label class="fast-note-label" for="fast-note">Not otomatik kaydedilir · Enter: kısayollara dön</label><input id="fast-note" maxlength="500" placeholder="Bir not ekle…" value="${esc(note)}"><div class="fast-choices"><button data-fast-decision="rejected" class="fast-bad" disabled>← Kötü <kbd>A</kbd></button><button data-fast-decision="pending" disabled>Emin değilim / Atla <kbd>S</kbd></button><button data-fast-decision="accepted" class="fast-good" disabled>İyi → <kbd>D</kbd></button></div><div class="fast-meta"><button id="fast-previous">Önceki</button><span id="fast-status" role="status" aria-live="polite">İki görsel yükleniyor…</span><button id="fast-detail">Eşleşme detayları ↗</button></div></div>`;
  $('#fast-close').onclick=()=>{rememberFastNote();$('#fast-dialog').close()};
  $('#fast-previous').onclick=()=>{if(fastReview.busy||!fastReview.index)return;rememberFastNote();fastReview.index--;renderFastReview()};
  $('#fast-detail').onclick=()=>{rememberFastNote();$('#fast-dialog').close();openDetail(r)};
  $('#fast-note').onkeydown=event=>{
    if(event.key==='Enter'&&!event.isComposing){event.preventDefault();$('#fast-content').focus({preventScroll:true})}
  };
  $('#fast-note').oninput=()=>{rememberFastNote();queueNoteAutosave(r,$('#fast-note').value)};
  $('#fast-note').title='Not yazdıktan sonra Enter ile karar kısayollarına dön.';
  document.querySelectorAll('[data-fast-decision]').forEach(button=>button.onclick=()=>chooseFastReview(button.dataset.fastDecision));
  const images=[$('#fast-before'),$('#fast-after')];
  const loaded=()=>{
    if(token!==fastReview.token)return;
    fastReview.ready=images.every(img=>img.complete&&img.naturalWidth>0);
    if(fastReview.ready)$('#fast-status').textContent='← / A kötü · → / D iyi · S atla · Space kaynak bölgesi. Kaydedilince ilerler.';
    fastButtons();
  };
  for(const img of images){img.onload=loaded;img.onerror=()=>{
    if(token!==fastReview.token)return;
    fastReview.imageError=true;$('#fast-status').textContent='Bir görsel yüklenemedi. S ile atlayabilir veya galeriye dönebilirsin.';fastButtons();
  }}
  loaded();
  $('#fast-region-toggle').onclick=toggleFastRegion;
  bindCropControls(r,'fast');
  // Only the next pair is prefetched; no unbounded decoded-image queue.
  const next=fastReview.rows[fastReview.index+1];
  if(next)for(const file of [next.original_file,'raw.png']){const image=new Image();image.src=next.asset+file;}
  if($('#fast-dialog').open)$('#fast-content').focus({preventScroll:true});
}
async function chooseFastReview(decision) {
  const r=fastReview.rows[fastReview.index];
  if(!r||fastReview.busy||reviewBusy||!reviewReady||(!fastReview.ready&&!(fastReview.imageError&&decision==='pending')))return;
  rememberFastNote();
  cancelNoteAutosave();
  fastReview.busy=true;fastButtons();$('#fast-status').textContent='Kaydediliyor…';
  try {
    await persistReview(r,decision,$('#fast-note').value);
    render();fastReview.index++;persistFastSession();fastReview.busy=false;renderFastReview();
  }catch(error){
    $('#fast-status').textContent=error.message||'Kaydedilemedi; bu görselde kaldık. Tekrar dene.';
  }finally{fastReview.busy=false;fastButtons()}
}
function fastKeydown(event) {
  if(!$('#fast-dialog').open)return;
  if(fastReview.busy&&event.key==='Escape'){event.preventDefault();return;}
  if(event.repeat||event.isComposing||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)return;
  const target=event.target;
  if(target?.closest?.('input,textarea,select,[contenteditable="true"],a,button'))return;
  if(event.key===' '){event.preventDefault();toggleFastRegion();return;}
  const decision=({'ArrowLeft':'rejected',a:'rejected',A:'rejected',ArrowRight:'accepted',d:'accepted',D:'accepted',s:'pending',S:'pending'})[event.key];
  if(!decision)return;
  event.preventDefault();chooseFastReview(decision);
}
function fastRegion(r){
  return r.audit_status==='reliable'&&r.region?.polygon?.length===4;
}
function toggleFastRegion(){
  const r=fastReview.rows[fastReview.index];
  if(!r||!fastRegion(r)){if($('#fast-status'))$('#fast-status').textContent='Güvenilir kaynak bölgesi bulunamadı; tahmini sınır gösterilmiyor.';return;}
  fastReview.boundary=!fastReview.boundary;
  $('#fast-boundary').style.display=fastReview.boundary?'block':'none';
  $('#fast-region-toggle').setAttribute('aria-pressed',String(fastReview.boundary));
  $('#fast-region-toggle').textContent=fastReview.boundary?'Kaynak bölgesini gizle [Space]':'Kaynak bölgesini göster [Space]';
}
function setupFastReview() {
  $('#fast-start').onclick=startFastReview;
  document.addEventListener('keydown',fastKeydown);
  $('#fast-dialog').addEventListener('cancel',event=>{if(fastReview.busy)event.preventDefault();else rememberFastNote()});
}
