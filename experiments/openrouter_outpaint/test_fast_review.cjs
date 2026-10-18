// Offline controller tests. No browser access and no real archive decisions.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

function fixture(){
  const nodes=new Map(), requests=[];
  function node(selector){
    if(!nodes.has(selector))nodes.set(selector,{value:'',textContent:'',innerHTML:'',disabled:false,
      open:false,complete:true,naturalWidth:720,focus(){},showModal(){this.open=true},close(){this.open=false}});
    return nodes.get(selector);
  }
  const rows=['one','two','three'].map(id=>({id,raw_sha256:id+'-sha',title:id,artists:['Artist'],
    asset:'assets/'+id+'/',original_file:'original.jpg',model:'nano',audit_status:'reliable'}));
  let decisions={};
  const c=vm.createContext({console,Map,Image:class{},data:{rows},$:node,
    esc:s=>String(s??'').replaceAll('"','&quot;'),selected:()=>rows,render(){},openDetail(){},
    document:{querySelectorAll:()=>[],addEventListener(){}},localStorage:{setItem(){}},
    fetch:async(url,options)=>{
      const payload=JSON.parse(options.body);requests.push(payload);
      decisions={...decisions,[payload.id]:{...payload,revision:payload.revision+1}};
      return {ok:true,json:async()=>({decisions})};
    }});
  const run=code=>vm.runInContext(code,c);
  for(const file of ['training-review.js','fast-review.js'])run(fs.readFileSync(path.join(__dirname,file),'utf8'));
  run('reviewReady=true; fastReview.rows=data.rows.slice(); fastReview.ready=true;');
  return {c,run,node,requests,rows};
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
  press('ArrowLeft');press('ArrowRight');press(' ');
  assert.deepEqual(calls,['rejected','accepted','pending']);
  h.node('#fast-dialog').open=false;press(' ');assert.equal(calls.length,3);
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
