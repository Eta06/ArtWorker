// Human decisions are separate from numerical source-fidelity measurements.
let trainingReviews = {}, reviewReady = false, reviewBusy = false;
const reviewOutboxKey='artworker-review-outbox-v1';
let reviewOutbox={}, mutationChain=Promise.resolve(), noteTimer=null, autosaveReplay=false;
let autosaveMessage='✓ Otomatik kayıt açık';
try{reviewOutbox=JSON.parse(localStorage.getItem(reviewOutboxKey)||'{}')}catch{}
function autosaveStatus(message,failed=false){
  autosaveMessage=message;
  for(const selector of ['#autosave-status','#fast-save-status']){
    const el=$(selector);if(el){el.textContent=message;el.classList.toggle('save-error',failed)}
  }
}
function stashReviewOutbox(){localStorage.setItem(reviewOutboxKey,JSON.stringify(reviewOutbox))}
function reviewNote(r){
  try{const draft=localStorage.getItem('artworker-note-'+r.id+'-'+r.raw_sha256);if(draft!==null)return draft}catch{}
  const saved=trainingReviews[r.id];return saved?.raw_sha256===r.raw_sha256?saved.reason||'':'';
}
function stashReviewNote(r,value){try{localStorage.setItem('artworker-note-'+r.id+'-'+r.raw_sha256,value)}catch{}}
function cancelNoteAutosave(){clearTimeout(noteTimer);noteTimer=null}
function queueNoteAutosave(r,reason){
  cancelNoteAutosave();
  autosaveStatus('Not taslağı cihazda; otomatik kaydediliyor…');
  noteTimer=setTimeout(()=>{
    noteTimer=null;
    persistReview(r,()=>reviewDecision(r),reason).catch(error=>autosaveStatus(error.message,true));
  },700);
}
function modelBanner(r){return r.model==='nano'?'<div class="nano-banner">🍌 NANO BANANA <span>Bu görsel Nano Banana ile üretildi</span></div>':''}
function reviewRegistrationConfirmation(r){
  const saved=trainingReviews[r.id];
  return saved?.raw_sha256===r.raw_sha256?saved.registration_confirmation||null:null;
}
function registrationIsReady(r){return r.registration_recovery?reviewRegistrationConfirmation(r)===r.registration_recovery.id:r.audit_status==='reliable'}
function reviewMetrics(r){return reviewCenterSquare(r)?null:r.registration_recovery?(registrationIsReady(r)?r.registration_recovery.metrics:null):r.metrics}
function recoveryControls(r,prefix){
  const recovery=r.registration_recovery;if(!recovery)return '';
  const kind=recovery.evidence.classification,ready=registrationIsReady(r);
  const warning=kind==='manual_center_reference'?'Bu otomatik eşleşme değil. Görselin ortasına kısa kenarı dolduran tam kare yerleştirildi. Son alan seçimi ve iyi/kötü kararı sende.':kind==='partial_reference'?`Referans alanının %${(100*recovery.evidence.visible_source_fraction).toFixed(1)} kadarı çıktıda görünüyor; geri kalanı kırpılmış.`:kind==='edited_reference'?'İçerik yeniden çizilmiş. Bu sınır temkinli bir konum adayıdır; piksel korunması onayı değildir.':'Seyrek kaynak özellikleri, yoğun piksel karşılaştırmasıyla ayrıca kontrol edildi. Bu bir sınır önerisidir.';
  return `<section class="recovery-controls"><strong>${ready?'✓ Kaynak alanını onayladın':kind==='manual_center_reference'?'Ortadan tam kare · seçimin bekleniyor':'Yeni bulunan aday kaynak alanı · onayın bekleniyor'}</strong><p>${esc(warning)}</p><button id="${prefix}-confirm-registration" ${ready||!reviewReady?'disabled':''}>Bu kaynak alanını kullan</button><span class="subtle">Turuncu sınırı incele. Bu düğme iyi/kötü kararını değiştirmez.</span></section>`;
}
function bindRecoveryControls(r,fast){
  const selector=fast?'#fast-confirm-registration':'#detail-confirm-registration';
  const button=$(selector);if(!r.registration_recovery||!button)return;
  button.onclick=async()=>{
    if(reviewBusy||fastReview.busy||!reviewReady)return;
    cancelNoteAutosave();reviewBusy=true;if(fast){fastReview.busy=true;fastButtons()}button.disabled=true;
    try{
      await persistReview(r,()=>reviewDecision(r),fast?$('#fast-note').value:$('#review-reason').value,null,null,r.registration_recovery.id);
      reviewBusy=false;render();
      if(fast){fastReview.busy=false;renderFastReview();toggleFastRegion()}
      else openDetail(r);
    }catch(error){autosaveStatus(error.message,true)}
    finally{reviewBusy=false;if(fast){fastReview.busy=false;fastButtons()}if($(selector))$(selector).disabled=registrationIsReady(r)}
  };
}
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
  return `<section class="review-box" aria-label="Eğitim kararı"><div class="review-top"><strong>Bu çıktı eğitime girsin mi?</strong>${reviewBadge(r)}</div><div class="detail-controls">${Object.entries(reviewLabels).map(([value,label])=>`<button data-decision="${value}" aria-pressed="${decision===value}" class="${decision===value?'active':''}" ${!reviewReady||reviewBusy?'disabled':''}>${label}</button>`).join('')}<button id="next-review">Sonraki incelenmemiş →</button></div><label class="subtle" for="review-reason">Not / eleme nedeni (isteğe bağlı)</label><div class="review-note"><input id="review-reason" maxlength="500" placeholder="Örn. yüz değişmiş, renk geçişi kötü…" value="${esc(reviewNote(r))}"><button id="save-review-note" ${!reviewReady||reviewBusy?'disabled':''}>Notu kaydet</button></div><p class="subtle" id="review-save" role="status">${reviewReady?'Kararlar anında bu Mac’te dosyaya kaydedilir. Kararsız olanlar eğitim listesine alınmaz.':'Kayıt bağlantısı yükleniyor; karar düğmeleri henüz açık değil.'}</p>${decision==='accepted'&&!registrationIsReady(r)?'<p class="error">Kabul ettin; ancak güvenilir kaynak eşleşmesi eksik. Bu çıktı ayrıca kontrol edilene kadar eğitim adayı listesinde bekletiliyor.</p>':''}</section>`;
}
function updateReviewSummary() {
  const text=reviewSummaryText();
  for(const selector of ['#review-summary','#fast-selection-count'])if($(selector))$(selector).textContent=text;
  if($('#review-rejected-start'))$('#review-rejected-start').textContent='Reddettiklerimi yeniden incele ('+data.rows.filter(r=>reviewDecision(r)==='rejected').length+')';
  if($('#review-recovered-start'))$('#review-recovered-start').textContent='Yeni kaynak alanlarını kontrol et ('+data.rows.filter(r=>r.registration_recovery&&!registrationIsReady(r)).length+')';
  if($('#reliable'))$('#reliable').textContent=data.rows.filter(registrationIsReady).length;
  if($('#unavailable'))$('#unavailable').textContent=data.rows.filter(r=>!registrationIsReady(r)&&r.audit_status!=='pending').length;
}
function reviewSummaryText(){
  const counts={accepted:0,rejected:0,pending:0,unreviewed:0};
  for(const r of data.rows){
    const saved=trainingReviews[r.id];
    counts[saved?.raw_sha256===r.raw_sha256?saved.decision:'unreviewed']++;
  }
  return `✓ Seçtin: ${counts.accepted} · Kötü: ${counts.rejected} · Kararsız: ${counts.pending} · İncelenmedi: ${counts.unreviewed}`;
}
function bindReviewControls(r) {
  document.querySelectorAll('[data-decision]').forEach(button=>button.onclick=()=>saveReview(r,button.dataset.decision));
  $('#save-review-note').onclick=()=>saveReview(r,reviewDecision(r));
  $('#review-reason').oninput=()=>{
    stashReviewNote(r,$('#review-reason').value);
    queueNoteAutosave(r,$('#review-reason').value);
  };
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
  let connected=false;
  try {
    await fetchReviews();
    connected=true;
    await replayReviewOutbox();
    $('#review-connection').textContent='✓ Kararlar bu Mac’te dosyaya kaydediliyor';
    autosaveStatus('✓ Otomatik kayıt açık · kararlar dosyaya yazılır');
    render();if(current&&$('#dialog').open)openDetail(current);
  } catch {
    reviewReady=connected;
    $('#review-connection').textContent=connected?'Bekleyen karar var. Dosyadaki kararlar yüklendi; bekleyen kayıt korunuyor.':'Kayıt sunucusuna bağlanılamadı. Karar vermeden önce Yenile’ye bas.';
    autosaveStatus(connected?'Bekleyen kayıt henüz onaylanmadı. Gerekirse kendi kararını yeniden seç.':'Kayıt bağlantısı yok; Yenile ile tekrar dene.',true);
    render();
    if(current&&$('#dialog').open){
      openDetail(current);
      $('#review-save').textContent='Kayıt bağlantısı yok. Yenile’ye basarak tekrar bağlan.';
    }
  }
}
async function saveReview(r, decision) {
  if(reviewBusy||!reviewReady)return;
  const reason=$('#review-reason').value;
  cancelNoteAutosave();
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
async function sendReviewOperation(operation) {
  if(!operation.attempted){
    const latest=trainingReviews[operation.id];
    const same=latest?.raw_sha256===operation.raw_sha256;
    operation.revision=same?latest.revision:0;
    if(operation.kind==='note')operation.decision=same?latest.decision:'pending';
    if(operation.kind==='geometry')operation.decision=same?latest.decision:'pending';
    if(operation.kind==='note'&&operation.crop_extensions&&same&&latest.crop_extensions)operation.crop_extensions=latest.crop_extensions;
    if(operation.kind==='note'&&operation.target_trim&&same&&latest.target_trim)operation.target_trim=latest.target_trim;
    if(operation.kind==='note'&&same&&'square_fit' in operation)operation.square_fit=latest.square_fit===true;
    if(operation.kind==='note'&&same&&'center_square' in operation)operation.center_square=latest.center_square===true;
    if(operation.kind==='note'&&same&&'registration_confirmation' in operation)operation.registration_confirmation=latest.registration_confirmation||null;
    operation.attempted=true;
    try{stashReviewOutbox()}catch{}
  }
  const response=await fetch('review-api',{method:'POST',keepalive:true,signal:typeof AbortSignal!=='undefined'?AbortSignal.timeout(12000):undefined,headers:{'Content-Type':'application/json'},body:JSON.stringify(operation)});
  const result=await response.json();
  if(!response.ok) {
    if(response.status===409){
      operation.conflict=true;try{stashReviewOutbox()}catch{}
      await fetchReviews();
    }
    throw Error(response.status===409?'Karar başka sekmede değişmiş; dosyanın üzerine yazılmadı. Kendi son tercihini uygulamak için karar düğmesine tekrar bas.':'Sunucuya kaydedilemedi. Karar bu cihazda bekliyor; bağlantı gelince tekrar gönderilecek.');
  }
  const acknowledged=result.decisions?.[operation.id];
  if(!acknowledged||acknowledged.request_id!==operation.request_id||acknowledged.raw_sha256!==operation.raw_sha256||acknowledged.decision!==operation.decision||acknowledged.reason!==operation.reason){
    throw Error('Dosya kaydı onayı doğrulanamadı; bekleyen karar korunuyor.');
  }
  if(operation.crop_extensions&&['top','bottom'].some(side=>acknowledged.crop_extensions?.[side]!==operation.crop_extensions[side]))throw Error('Eğitim alanı kaydı onaylanmadı; seçim cihazda korunuyor.');
  if(operation.target_trim&&['top','bottom'].some(side=>acknowledged.target_trim?.[side]!==operation.target_trim[side]))throw Error('Hedef kırpma kaydı onaylanmadı; seçim cihazda korunuyor.');
  if('square_fit' in operation&&acknowledged.square_fit!==operation.square_fit)throw Error('Kare referans kaydı onaylanmadı; seçim cihazda korunuyor.');
  if('center_square' in operation&&acknowledged.center_square!==operation.center_square)throw Error('Manuel merkez kare kaydı onaylanmadı; seçim cihazda korunuyor.');
  if('registration_confirmation' in operation&&acknowledged.registration_confirmation!==operation.registration_confirmation)throw Error('Kaynak alanı onayı kaydedilmedi; bekleyen kayıt korunuyor.');
  trainingReviews=result.decisions;
  // Only delete after the server acknowledges the exact operation.
  delete reviewOutbox[operation.request_id];
  try{stashReviewOutbox();localStorage.setItem('artworker-training-review-backup',JSON.stringify(trainingReviews))}catch{}
  autosaveStatus('✓ Dosyaya kaydedildi · '+new Date().toLocaleTimeString('tr-TR'));
}
function serialReviewJob(job){
  const pending=mutationChain.then(job);mutationChain=pending.catch(()=>{});return pending;
}
function replayReviewOutbox(){return serialReviewJob(async()=>{
  for(const operation of Object.values(reviewOutbox))await sendReviewOperation(operation);
})}
function persistReview(r,decision,reason,cropOverride=null,trimOverride=null,registrationOverride=undefined,squareOverride=undefined,centerOverride=undefined) {
  // Write intent synchronously, before the first network wait or page navigation.
  // A single id per operation makes retry after a lost response safe.
  const saved=trainingReviews[r.id];
  if(typeof decision!=='function'&&Object.values(reviewOutbox).some(op=>op.id===r.id&&op.raw_sha256===r.raw_sha256&&op.conflict)){
    // A fresh human click supersedes this tab's conflicted intentions for that
    // image. Background note saves may never overwrite a conflicting decision.
    for(const [id,op] of Object.entries(reviewOutbox))if(op.id===r.id&&op.raw_sha256===r.raw_sha256)delete reviewOutbox[id];
  }
  const revision=saved?.raw_sha256===r.raw_sha256?saved.revision:0;
  const operation={id:r.id,raw_sha256:r.raw_sha256,decision:typeof decision==='function'?decision():decision,reason,revision,kind:typeof decision==='function'?(cropOverride||trimOverride||registrationOverride!==undefined||squareOverride!==undefined||centerOverride!==undefined?'geometry':'note'):'decision',crop_extensions:cropOverride||reviewCropExtensions(r),target_trim:trimOverride||reviewTargetTrim(r),registration_confirmation:registrationOverride===undefined?reviewRegistrationConfirmation(r):registrationOverride,square_fit:squareOverride===undefined?reviewSquareFit(r):squareOverride,center_square:centerOverride===undefined?reviewCenterSquare(r):centerOverride,request_id:crypto.randomUUID()};
  reviewOutbox[operation.request_id]=operation;
  try{stashReviewOutbox()}catch{autosaveStatus('Tarayıcı yedeği yazılamadı; dosya kaydı bekleniyor.',true)}
  autosaveStatus('Kaydediliyor…');
  return serialReviewJob(async()=>{
    // Flush earlier intentions before applying a newer note or decision.
    for(const earlier of Object.values(reviewOutbox)){
      if(earlier.request_id===operation.request_id)break;
      await sendReviewOperation(earlier);
    }
    await sendReviewOperation(operation);
  }).catch(error=>{
    autosaveStatus('Cihazda bekleyen kayıt var; dosyaya henüz onaylanmadı. Yeniden deneniyor…',true);
    throw error;
  });
}
function setupReviewAutosave(){
  const retry=async()=>{
    if(autosaveReplay||reviewBusy||!Object.keys(reviewOutbox).length||typeof fastReview!=='undefined'&&fastReview.busy)return;
    autosaveReplay=true;
    try{
      await fetchReviews();await replayReviewOutbox();render();
      $('#review-connection').textContent='✓ Kararlar bu Mac’te dosyaya kaydediliyor';
      if(typeof fastButtons==='function')fastButtons();
    }catch{autosaveStatus('Bekleyen kayıt korunuyor; bağlantı gelince tekrar denenecek.',true)}
    finally{autosaveReplay=false}
  };
  window.addEventListener('online',retry);
  setInterval(retry,4000);
}
function exportReviews() {
  const selections=data.rows.map(r=>({id:r.id,cover_id:r.cover_id,model:r.model,raw_sha256:r.raw_sha256,title:r.title,decision:reviewDecision(r),reason:trainingReviews[r.id]?.raw_sha256===r.raw_sha256?trainingReviews[r.id].reason:'',crop_extensions:reviewCropExtensions(r),target_trim:reviewTargetTrim(r),square_fit:reviewSquareFit(r),center_square:reviewCenterSquare(r),registration_confirmation:reviewRegistrationConfirmation(r),conditioning:trainingCrop(r)}));
  const blob=new Blob([JSON.stringify({schema_version:4,policy:'explicit_accept_and_reliable_registration',selections},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='artworker-egitim-secimleri.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
