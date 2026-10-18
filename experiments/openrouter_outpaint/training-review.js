// Human decisions are separate from numerical source-fidelity measurements.
let trainingReviews = {}, reviewReady = false, reviewBusy = false;
const reviewLabels = {accepted:'Eğitime al', rejected:'Eğitimden çıkar', pending:'Kararsız / incelenmedi'};
function reviewDecision(r) {
  const saved = trainingReviews[r.id];
  return saved?.raw_sha256 === r.raw_sha256 ? saved.decision : 'pending';
}
function reviewBadge(r) {
  const decision = reviewDecision(r);
  return `<span class="review-badge ${decision}">${decision === 'accepted' ? '✓ Eğitim adayı' : decision === 'rejected' ? '× Eğitim dışında' : '○ İncelenmedi / kararsız'}</span>`;
}
function reviewControls(r) {
  const decision = reviewDecision(r), saved = trainingReviews[r.id];
  return `<section class="review-box" aria-label="Eğitim kararı"><div class="review-top"><strong>Bu çıktı eğitime girsin mi?</strong>${reviewBadge(r)}</div><div class="detail-controls">${Object.entries(reviewLabels).map(([value,label])=>`<button data-decision="${value}" aria-pressed="${decision===value}" class="${decision===value?'active':''}" ${!reviewReady||reviewBusy?'disabled':''}>${label}</button>`).join('')}<button id="next-review">Sonraki incelenmemiş →</button></div><label class="subtle" for="review-reason">Not / eleme nedeni (isteğe bağlı)</label><div class="review-note"><input id="review-reason" maxlength="500" placeholder="Örn. yüz değişmiş, renk geçişi kötü…" value="${esc(saved?.raw_sha256===r.raw_sha256?saved.reason:'')}"><button id="save-review-note" ${!reviewReady||reviewBusy?'disabled':''}>Notu kaydet</button></div><p class="subtle" id="review-save" role="status">${reviewReady?'Kararlar anında bu Mac’te dosyaya kaydedilir. Kararsız olanlar eğitim listesine alınmaz.':'Kayıt bağlantısı yükleniyor; karar düğmeleri henüz açık değil.'}</p>${decision==='accepted'&&r.audit_status!=='reliable'?'<p class="error">Kabul ettin; ancak güvenilir kaynak eşleşmesi eksik. Bu çıktı ayrıca kontrol edilene kadar eğitim adayı listesinde bekletiliyor.</p>':''}</section>`;
}
function updateReviewSummary() {
  const counts = {accepted:0,rejected:0,pending:0};
  for (const r of data.rows) counts[reviewDecision(r)]++;
  $('#review-summary').textContent = `${counts.accepted} kabul · ${counts.rejected} eğitim dışında · ${counts.pending} incelenmedi / kararsız`;
}
function bindReviewControls(r) {
  document.querySelectorAll('[data-decision]').forEach(button=>button.onclick=()=>saveReview(r,button.dataset.decision));
  $('#save-review-note').onclick=()=>saveReview(r,reviewDecision(r));
  $('#next-review').onclick=()=>{
    const rows=selected(), index=rows.findIndex(x=>x.id===r.id);
    const ordered=[...rows.slice(index+1),...rows.slice(0,index+1)];
    const next=ordered.find(x=>x.id!==r.id&&reviewDecision(x)==='pending') || data.rows.find(x=>x.id!==r.id&&reviewDecision(x)==='pending');
    if(next) {openDetail(next);$('#dialog').scrollTop=0;}
    else $('#review-save').textContent='İncelenmemiş başka çıktı kalmadı.';
  };
}
async function fetchReviews() {
  const response=await fetch('review-api',{cache:'no-store'});
  if(!response.ok) throw Error('Review server unavailable');
  const snapshot=await response.json();
  trainingReviews=snapshot.decisions;reviewReady=true;
  return snapshot;
}
async function startReviews() {
  try {
    await fetchReviews();
    $('#review-connection').textContent='✓ Kararlar bu Mac’te dosyaya kaydediliyor';
    render();if(current&&$('#dialog').open)openDetail(current);
  } catch {
    reviewReady=false;
    $('#review-connection').textContent='Kayıt sunucusuna bağlanılamadı. Karar vermeden önce Yenile’ye bas.';
    if(current&&$('#dialog').open){
      openDetail(current);
      $('#review-save').textContent='Kayıt bağlantısı yok. Yenile’ye basarak tekrar bağlan.';
    }
  }
}
async function saveReview(r, decision) {
  if(reviewBusy||!reviewReady)return;
  const reason=$('#review-reason').value;
  reviewBusy=true;
  document.querySelectorAll('[data-decision],#save-review-note').forEach(b=>b.disabled=true);
  $('#review-save').textContent='Kaydediliyor…';
  try {
    await persistReview(r,decision,reason);
    reviewBusy=false;render();
    if(current?.id===r.id&&$('#dialog').open){openDetail(r);$('#review-save').textContent='✓ Dosyaya kaydedildi. '+(decision==='rejected'?'Bu çıktı eğitim verisine alınmayacak.':decision==='accepted'?'Eğitim adayı olarak işaretlendi.':'Bu çıktı karar verilene kadar eğitim dışında.');}
  } catch(error) {
    if(current?.id===r.id)$('#review-save').textContent=error.message;
  } finally {
    reviewBusy=false;
    document.querySelectorAll('[data-decision],#save-review-note').forEach(b=>b.disabled=!reviewReady);
  }
}
async function persistReview(r,decision,reason) {
  const saved=trainingReviews[r.id];
  const revision=saved?.raw_sha256===r.raw_sha256?saved.revision:0;
  const response=await fetch('review-api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:r.id,raw_sha256:r.raw_sha256,decision,reason,revision})});
  const result=await response.json();
  if(!response.ok) {
    if(response.status===409)await fetchReviews();
    throw Error(response.status===409?'Başka bir sekme bu kararı değiştirmiş. Yenile ve kararını tekrar seç.':'Kayıt başarısız. Karar kaydedilmedi; tekrar dene.');
  }
  trainingReviews=result.decisions;
  try{localStorage.setItem('artworker-training-review-backup',JSON.stringify(trainingReviews))}catch{}
}
function exportReviews() {
  const selections=data.rows.map(r=>({id:r.id,cover_id:r.cover_id,model:r.model,raw_sha256:r.raw_sha256,title:r.title,decision:reviewDecision(r),reason:trainingReviews[r.id]?.raw_sha256===r.raw_sha256?trainingReviews[r.id].reason:''}));
  const blob=new Blob([JSON.stringify({schema_version:2,policy:'explicit_accept_and_reliable_registration',selections},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='artworker-egitim-secimleri.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
