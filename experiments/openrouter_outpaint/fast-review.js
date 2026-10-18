// A frozen queue avoids reordering/skipping cards when filters change after a save.
const fastReview={rows:[],index:0,busy:false,ready:false,imageError:false,token:0,drafts:new Map()};
function rememberFastNote() {
  const row=fastReview.rows[fastReview.index], input=$('#fast-note');
  if(row&&input)fastReview.drafts.set(row.id+'-'+row.raw_sha256,input.value);
}
function fastQueue(rows) {
  rememberFastNote();
  fastReview.rows=rows.slice();fastReview.index=0;
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
}
function renderFastReview() {
  const r=fastReview.rows[fastReview.index], token=++fastReview.token;
  fastReview.ready=false;fastReview.imageError=false;
  if(!r){
    const skipped=fastReview.rows.filter(row=>reviewDecision(row)==='pending');
    $('#fast-content').innerHTML=`<div class="fast-finished"><div class="eyebrow">Hızlı inceleme</div><h2 id="fast-title">${fastReview.rows.length?'Bu tur tamamlandı.':'Bu filtrede incelenecek görsel yok.'}</h2><p>Kararlar dosyaya kaydedildi. ${skipped.length} görsel emin olunmadığı için eğitim dışında bekliyor.</p><div class="detail-controls"><button id="fast-retry" ${skipped.length?'':'disabled'}>Atladıklarımı tekrar göster</button><button id="fast-close">Galeriye dön</button><button id="fast-export">Kararları yedekle</button><a href="training-manifest" download="artworker-egitim-adaylari.json">Son eğitim adayı listesini indir</a></div></div>`;
    $('#fast-close').onclick=()=>$('#fast-dialog').close();
    $('#fast-retry').onclick=()=>fastQueue(skipped);
    $('#fast-export').onclick=exportReviews;return;
  }
  const saved=trainingReviews[r.id],key=r.id+'-'+r.raw_sha256;
  const note=fastReview.drafts.get(key)??(saved?.raw_sha256===r.raw_sha256?saved.reason:'')??'';
  $('#fast-content').innerHTML=`<div class="fast-heading"><div><span class="eyebrow">Hızlı inceleme · ${fastReview.index+1} / ${fastReview.rows.length}</span><h2 id="fast-title">${esc(r.title)}</h2><p>${esc(r.artists.join(' · '))} · ${r.model==='sunburst'?'Sunburst 2.5':'Nano Banana 2.1'} ${reviewBadge(r)}</p></div><button id="fast-close" aria-label="Hızlı incelemeyi kapat">✕</button></div><div class="fast-pair"><figure><figcaption>ÖNCE · ORİJİNAL</figcaption><div class="fast-image"><img id="fast-before" src="${r.asset}${r.original_file}" alt="${esc(r.title)} orijinal kapak" decoding="async"></div></figure><figure><figcaption>SONRA · GENİŞLETİLMİŞ</figcaption><div class="fast-image"><img id="fast-after" src="${r.asset}raw.png" alt="${esc(r.title)} genişletilmiş kapak" decoding="async"></div></figure></div><div class="fast-actions"><label class="fast-note-label" for="fast-note">İsteğe bağlı not · kararınla kaydedilir · Enter: kısayollara dön</label><input id="fast-note" maxlength="500" placeholder="Bir not ekle…" value="${esc(note)}"><div class="fast-choices"><button data-fast-decision="rejected" class="fast-bad" disabled>← Kötü <kbd>A</kbd></button><button data-fast-decision="pending" disabled>Emin değilim / Atla <kbd>Space</kbd></button><button data-fast-decision="accepted" class="fast-good" disabled>İyi → <kbd>D</kbd></button></div><div class="fast-meta"><button id="fast-previous">Önceki</button><span id="fast-status" role="status" aria-live="polite">İki görsel yükleniyor…</span><button id="fast-detail">Eşleşme detayları ↗</button></div></div>`;
  $('#fast-close').onclick=()=>{rememberFastNote();$('#fast-dialog').close()};
  $('#fast-previous').onclick=()=>{if(fastReview.busy||!fastReview.index)return;rememberFastNote();fastReview.index--;renderFastReview()};
  $('#fast-detail').onclick=()=>{rememberFastNote();$('#fast-dialog').close();openDetail(r)};
  $('#fast-note').onkeydown=event=>{
    if(event.key==='Enter'&&!event.isComposing){event.preventDefault();$('#fast-content').focus({preventScroll:true})}
  };
  $('#fast-note').title='Not yazdıktan sonra Enter ile karar kısayollarına dön.';
  document.querySelectorAll('[data-fast-decision]').forEach(button=>button.onclick=()=>chooseFastReview(button.dataset.fastDecision));
  const images=[$('#fast-before'),$('#fast-after')];
  const loaded=()=>{
    if(token!==fastReview.token)return;
    fastReview.ready=images.every(img=>img.complete&&img.naturalWidth>0);
    if(fastReview.ready)$('#fast-status').textContent='← / A kötü · → / D iyi · Space atla. Kaydedilince otomatik ilerler.';
    fastButtons();
  };
  for(const img of images){img.onload=loaded;img.onerror=()=>{
    if(token!==fastReview.token)return;
    fastReview.imageError=true;$('#fast-status').textContent='Bir görsel yüklenemedi. Space ile atlayabilir veya galeriye dönebilirsin.';fastButtons();
  }}
  loaded();
  // Only the next pair is prefetched; no unbounded decoded-image queue.
  const next=fastReview.rows[fastReview.index+1];
  if(next)for(const file of [next.original_file,'raw.png']){const image=new Image();image.src=next.asset+file;}
  if($('#fast-dialog').open)$('#fast-content').focus({preventScroll:true});
}
async function chooseFastReview(decision) {
  const r=fastReview.rows[fastReview.index];
  if(!r||fastReview.busy||reviewBusy||!reviewReady||(!fastReview.ready&&!(fastReview.imageError&&decision==='pending')))return;
  rememberFastNote();
  fastReview.busy=true;fastButtons();$('#fast-status').textContent='Kaydediliyor…';
  try {
    await persistReview(r,decision,$('#fast-note').value);
    render();fastReview.index++;fastReview.busy=false;renderFastReview();
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
  const decision=({'ArrowLeft':'rejected',a:'rejected',A:'rejected',ArrowRight:'accepted',d:'accepted',D:'accepted',' ':'pending'})[event.key];
  if(!decision)return;
  event.preventDefault();chooseFastReview(decision);
}
function setupFastReview() {
  $('#fast-start').onclick=startFastReview;
  document.addEventListener('keydown',fastKeydown);
  $('#fast-dialog').addEventListener('cancel',event=>{if(fastReview.busy)event.preventDefault();else rememberFastNote()});
}
