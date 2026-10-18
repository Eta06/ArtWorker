// Offline controller tests. No browser access and no real archive decisions.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

function fixture(storage=new Map()){
  const nodes=new Map(), requests=[];
  function node(selector){
    if(!nodes.has(selector))nodes.set(selector,{value:'',textContent:'',innerHTML:'',disabled:false,
      open:false,complete:true,naturalWidth:720,style:{},attributes:{},classList:{toggle(){}},setAttribute(k,v){this.attributes[k]=v},focus(){},showModal(){this.open=true},close(){this.open=false}});
    return nodes.get(selector);
  }
  const rows=['one','two','three'].map(id=>({id,raw_sha256:id+'-sha',title:id,artists:['Artist'],
    asset:'assets/'+id+'/',original_file:'original.jpg',model:'nano',audit_status:'reliable'}));
  let decisions={};
  const c=vm.createContext({console,Map,Image:class{},data:{rows},$:node,crypto:require('node:crypto').webcrypto,setTimeout,clearTimeout,
    esc:s=>String(s??'').replaceAll('"','&quot;'),selected:()=>rows,render(){},openDetail(){},
    document:{querySelectorAll:()=>[],addEventListener(){}},localStorage:{setItem:(k,v)=>storage.set(k,v),getItem:k=>storage.get(k)||null},
    fetch:async(url,options)=>{
      const payload=JSON.parse(options.body);requests.push(payload);
      decisions={...decisions,[payload.id]:{...payload,revision:payload.revision+1}};
      return {ok:true,json:async()=>({decisions})};
    }});
  const run=code=>vm.runInContext(code,c);
  for(const file of ['training-review.js','training-crop.js','fast-review.js'])run(fs.readFileSync(path.join(__dirname,file),'utf8'));
  run('reviewReady=true; fastReview.rows=data.rows.slice(); fastReview.ready=true;');
  return {c,run,node,requests,rows,storage};
}

test('successful skip saves optional note and pending decision before advancing',async()=>{
  const h=fixture();h.node('#fast-note').value='Renkte emin değilim';
  await h.run("chooseFastReview('pending')");
  assert.equal(h.requests.length,1);
  assert.equal(h.requests[0].decision,'pending');
  assert.equal(h.requests[0].reason,'Renkte emin değilim');
  assert.equal(h.run('fastReview.index'),1);
  assert.equal(h.run("reviewDecision(data.rows[0])"),'pending');
  assert.equal(h.run('fastReview.drafts.has(data.rows[1].id+"-"+data.rows[1].raw_sha256)'),false);
});

test('failure keeps the same cover and note; rapid double presses do not submit twice',async()=>{
  const h=fixture();h.node('#fast-note').value='Important note';
  let reject;
  h.c.fetch=()=>new Promise((resolve,fail)=>{reject=fail});
  const saving=h.run("chooseFastReview('accepted')");
  await h.run("chooseFastReview('rejected')");
  assert.equal(h.run('fastReview.index'),0);
  reject(Error('offline'));
  await saving;
  assert.equal(h.run('fastReview.index'),0);
  assert.equal(h.node('#fast-note').value,'Important note');
  assert.match(h.node('#fast-status').textContent,/offline/);
});

test('default queue resumes pending covers; explicit decision filters allow re-review',()=>{
  const h=fixture();
  h.run("trainingReviews.one={raw_sha256:data.rows[0].raw_sha256,decision:'accepted',reason:'',revision:1}");
  h.run('startFastReview()');
  assert.equal(h.run('fastReview.rows.length'),2);
  assert.equal(h.run('fastReview.rows[0].id'),'two');
  h.node('#review-filter').value='accepted';h.run('startFastReview()');
  assert.equal(h.run('fastReview.rows.length'),3); // selected() is the filter's supplied list.
});

test('shortcuts ignore text editing, IME, modifiers and held keys',async()=>{
  const h=fixture();h.node('#fast-dialog').open=true;
  const calls=[];h.c.calls=calls;
  h.run('chooseFastReview=decision=>calls.push(decision)');
  const press=(key,extra={})=>h.c.fastKeydown({key,preventDefault(){},target:{closest:()=>null},...extra});
  press('a',{target:{closest:()=>({tagName:'INPUT'})}});
  press('d',{repeat:true});press(' ',{isComposing:true});press('ArrowLeft',{metaKey:true});
  assert.equal(calls.length,0);
  press('ArrowLeft');press('ArrowRight');press('s');press(' ');
  assert.deepEqual(calls,['rejected','accepted','pending']);
  h.node('#fast-dialog').open=false;press('s');assert.equal(calls.length,3);
});

test('Space only toggles reliable geometry and never sends or advances a decision',()=>{
  const h=fixture();
  h.rows[0].dimensions=[1000,2000];h.rows[0].region={polygon:[[0,500],[1000,500],[1000,1500],[0,1500]]};
  h.node('#fast-dialog').open=true;
  h.run('renderFastReview()');
  assert.match(h.node('#fast-content').innerHTML,/🍌 NANO BANANA/);
  assert.match(h.node('#fast-content').innerHTML,/preserveAspectRatio="xMidYMid meet"/);
  assert.match(h.node('#fast-content').innerHTML,/style="display:none"/);
  const key={key:' ',preventDefault(){},target:{closest:()=>null}};
  h.c.fastKeydown(key);assert.equal(h.node('#fast-boundary').style.display,'block');
  h.c.fastKeydown(key);assert.equal(h.node('#fast-boundary').style.display,'none');
  assert.equal(h.requests.length,0);assert.equal(h.run('fastReview.index'),0);
  h.rows[0].audit_status='unavailable';h.run('renderFastReview()');
  assert.doesNotMatch(h.node('#fast-content').innerHTML,/id="fast-boundary"/);
});

test('intent is durable before the network request and is replayed after refresh',async()=>{
  const h=fixture();
  h.c.fetch=async()=>{assert.equal(JSON.parse(h.storage.get('artworker-review-outbox-v1'))[h.run('Object.keys(reviewOutbox)[0]')].reason,'saved before request');throw Error('offline')};
  await assert.rejects(h.run("persistReview(data.rows[0],'accepted','saved before request')"));
  const reloaded=fixture(h.storage);
  await reloaded.run('replayReviewOutbox()');
  assert.equal(reloaded.requests.length,1);
  assert.equal(reloaded.requests[0].decision,'accepted');
  assert.equal(reloaded.run('Object.keys(reviewOutbox).length'),0);
});

test('resumes cursor and draft without transferring a note to another image',()=>{
  const h=fixture();h.node('#fast-note').value='durable note';h.run('rememberFastNote(); fastReview.index=1; persistFastSession()');
  const next=fixture(h.storage);assert.equal(next.run('restoreFastSession()'),true);
  assert.equal(next.run('fastReview.index'),1);
  assert.equal(next.run('fastReview.drafts.get("one-one-sha")'),'durable note');
  assert.equal(next.run('fastReview.drafts.has("two-two-sha")'),false);
});

test('queued notes inherit the accepted decision instead of resetting it to pending',async()=>{
  const h=fixture();
  const accepted=h.run("persistReview(data.rows[0],'accepted','initial')");
  const note=h.run("persistReview(data.rows[0],()=>reviewDecision(data.rows[0]),'updated note')");
  await Promise.all([accepted,note]);
  assert.equal(h.requests.length,2);assert.equal(h.requests[1].revision,1);
  assert.equal(h.requests[1].decision,'accepted');assert.equal(h.requests[1].reason,'updated note');
});

test('typing a note autosaves without requiring a decision click',async()=>{
  const h=fixture();let timer;
  h.c.setTimeout=callback=>{timer=callback;return 1};h.c.clearTimeout=()=>{};
  h.run("queueNoteAutosave(data.rows[0],'automatic note')");
  timer();await h.run('mutationChain');
  assert.equal(h.requests.length,1);assert.equal(h.requests[0].reason,'automatic note');
  assert.equal(h.requests[0].decision,'pending');
});

test('including the bottom autosaves the crop without accepting or advancing the cover',async()=>{
  const h=fixture();h.rows[0].dimensions=[1000,2000];
  h.rows[0].region={polygon:[[0,500],[1000,500],[1000,1500],[0,1500]]};
  const original=JSON.stringify(h.rows[0].region);
  await h.run("toggleTrainingExtent(data.rows[0],'bottom','fast')");
  assert.equal(h.requests.length,1);assert.equal(h.requests[0].decision,'pending');
  assert.equal(h.requests[0].crop_extensions.bottom,true);assert.equal(h.run('fastReview.index'),0);
  assert.equal(JSON.stringify(h.rows[0].region),original);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(trainingCrop(data.rows[0]).box_px)')),[0,500,1000,2000]);
  const restored=fixture(h.storage);restored.rows[0].dimensions=h.rows[0].dimensions;restored.rows[0].region=h.rows[0].region;
  assert.equal(restored.run('reviewCropExtensions(data.rows[0]).bottom'),true);
});

test('top/bottom selections carry into later decisions and stay bound to the output hash',async()=>{
  const h=fixture();h.rows[0].dimensions=[1000,2000];h.rows[0].region={polygon:[[0,500],[1000,500],[1000,1500],[0,1500]]};
  await h.run("toggleTrainingExtent(data.rows[0],'top','fast')");
  await h.run("persistReview(data.rows[0],'accepted','ready')");
  assert.equal(h.requests[1].crop_extensions.top,true);
  h.rows[0].raw_sha256='new-output';
  assert.equal(h.run('reviewCropExtensions(data.rows[0]).top'),false);
});

test('hover previews the next crop without persisting and restores the hidden saved selection',()=>{
  const h=fixture();h.rows[0].dimensions=[1000,2000];h.rows[0].region={polygon:[[0,500],[1000,500],[1000,1500],[0,1500]]};
  const saved=JSON.stringify([...h.storage]);
  h.run("bindCropControls(data.rows[0],'fast')");
  h.node('#fast-crop-bottom').onmouseenter();
  assert.equal(h.node('#fast-training-crop').attributes.points,'0,500 1000,500 1000,2000 0,2000');
  assert.equal(h.node('#fast-boundary').style.display,'block');
  assert.match(h.node('#fast-crop-info').textContent,/ÖNİZLEME/);
  assert.equal(h.requests.length,0);assert.equal(JSON.stringify([...h.storage]),saved);
  assert.equal(h.run('reviewCropExtensions(data.rows[0]).bottom'),false);
  h.node('#fast-crop-bottom').onmouseleave();
  assert.equal(h.node('#fast-training-crop').attributes.points,'0,500 1000,500 1000,1500 0,1500');
  assert.equal(h.node('#fast-boundary').style.display,'none');
});

test('keyboard focus previews removal, and Space visibility intent survives mouse leave',()=>{
  const h=fixture();h.rows[0].dimensions=[1000,2000];h.rows[0].region={polygon:[[0,500],[1000,500],[1000,1500],[0,1500]]};
  h.run("cropDrafts.set(data.rows[0].id+'-'+data.rows[0].raw_sha256,{top:true,bottom:false});bindCropControls(data.rows[0],'fast')");
  h.node('#fast-crop-top').onfocus();
  assert.equal(h.node('#fast-training-crop').attributes.points,'0,500 1000,500 1000,1500 0,1500');
  assert.match(h.node('#fast-crop-info').textContent,/çıkarılacak/);
  h.run('toggleFastRegion()');
  h.node('#fast-crop-top').onblur();
  assert.equal(h.node('#fast-boundary').style.display,'block');
  assert.equal(h.node('#fast-training-crop').attributes.points,'0,0 1000,0 1000,1500 0,1500');
  assert.equal(h.requests.length,0);
});

test('detail hover follows the current boundary checkbox without changing saved crop',()=>{
  const h=fixture();h.rows[0].dimensions=[1000,2000];h.rows[0].region={polygon:[[0,500],[1000,500],[1000,1500],[0,1500]]};
  h.node('#boundary-toggle').checked=true;
  h.run("bindCropControls(data.rows[0],'detail')");
  h.node('#detail-crop-top').onmouseenter();
  assert.equal(h.node('#detail-training-crop').attributes.points,'0,0 1000,0 1000,1500 0,1500');
  h.node('#boundary-toggle').checked=false;
  h.node('#detail-crop-top').onmouseleave();
  assert.equal(h.node('#boundary').style.display,'none');
  assert.equal(h.run('reviewCropExtensions(data.rows[0]).top'),false);
});

test('summary separates selected, rejected, pending and unseen outputs by current fingerprint',()=>{
  const h=fixture();
  h.run("trainingReviews={one:{raw_sha256:data.rows[0].raw_sha256,decision:'accepted'},two:{raw_sha256:data.rows[1].raw_sha256,decision:'rejected'},three:{raw_sha256:'old-output',decision:'accepted'}}");
  assert.match(h.run('reviewSummaryText()'),/Seçtin: 1 · Kötü: 1 · Kararsız: 0 · İncelenmedi: 1/);
  h.run("trainingReviews.three={raw_sha256:data.rows[2].raw_sha256,decision:'pending'};updateReviewSummary()");
  assert.match(h.node('#fast-selection-count').textContent,/Kararsız: 1 · İncelenmedi: 0/);
});

test('remove top keeps only source plus bottom target, autosaves, and clears conflicting include-top',async()=>{
  const h=fixture();h.rows[0].dimensions=[1000,2000];h.rows[0].region={polygon:[[0,500],[1000,500],[1000,1500],[0,1500]]};
  h.run("cropDrafts.set(data.rows[0].id+'-'+data.rows[0].raw_sha256,{top:true,bottom:false})");
  await h.run("toggleTargetTrim(data.rows[0],'top','fast')");
  const crop=JSON.parse(h.run('JSON.stringify(trainingCrop(data.rows[0]))'));
  assert.deepEqual(crop.target_box_px,[0,500,1000,2000]);assert.deepEqual(crop.input_box_in_target_px,[0,0,1000,1000]);
  assert.equal(h.requests[0].target_trim.top,true);assert.equal(h.requests[0].crop_extensions.top,false);
  assert.equal(h.requests[0].decision,'pending');assert.equal(h.run('fastReview.index'),0);
  const reloaded=fixture(h.storage);assert.equal(reloaded.run('reviewTargetTrim(data.rows[0]).top'),true);
});

test('trim hover does not save and resets the target overlay on mouse leave',()=>{
  const h=fixture();h.rows[0].dimensions=[1000,2000];h.rows[0].region={polygon:[[0,500],[1000,500],[1000,1500],[0,1500]]};
  h.run("bindCropControls(data.rows[0],'fast')");h.node('#fast-trim-bottom').onmouseenter();
  assert.match(h.node('#fast-trim-zones').innerHTML,/trim-discard/);
  assert.match(h.node('#fast-crop-info').textContent,/eğitim hedefinden çıkarılacak/);
  assert.equal(h.requests.length,0);assert.equal(h.run('reviewTargetTrim(data.rows[0]).bottom'),false);
  h.node('#fast-trim-bottom').onmouseleave();assert.equal(h.node('#fast-trim-zones').innerHTML,'');
});

test('a response without the exact durable operation acknowledgement never advances',async()=>{
  const h=fixture();h.c.fetch=async()=>({ok:true,json:async()=>({decisions:{}})});
  await h.run("chooseFastReview('accepted')");
  assert.equal(h.run('fastReview.index'),0);
  assert.equal(h.run('Object.keys(reviewOutbox).length'),1);
  assert.match(h.node('#fast-status').textContent,/onayı doğrulanamadı/);
});

test('several offline intentions replay in order with fresh revisions after reload',async()=>{
  const h=fixture();h.c.fetch=async()=>{throw Error('offline')};
  await assert.rejects(h.run("persistReview(data.rows[0],'accepted','first')"));
  await assert.rejects(h.run("persistReview(data.rows[0],'rejected','second')"));
  const reloaded=fixture(h.storage);await reloaded.run('replayReviewOutbox()');
  assert.equal(reloaded.requests.length,2);assert.equal(reloaded.requests[0].revision,0);
  assert.equal(reloaded.requests[1].revision,1);assert.equal(reloaded.requests[1].decision,'rejected');
});

test('decisions wait for images; both original and expanded remain in the same view',async()=>{
  const h=fixture();h.run('fastReview.ready=false');
  await h.run("chooseFastReview('accepted')");assert.equal(h.requests.length,0);
  h.run('renderFastReview()');
  const html=h.node('#fast-content').innerHTML;
  assert.match(html,/id="fast-before" src="assets\/one\/original.jpg"/);
  assert.match(html,/id="fast-after" src="assets\/one\/raw.png"/);
  assert.match(html,/maxlength="500"/);
  h.node('#fast-note').value='A note';h.run('rememberFastNote();fastReview.index=1;renderFastReview()');
  assert.doesNotMatch(h.node('#fast-content').innerHTML,/value="A note"/);
  h.run('fastReview.index=0;renderFastReview()');
  assert.match(h.node('#fast-content').innerHTML,/value="A note"/);
});
