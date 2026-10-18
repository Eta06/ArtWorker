const {test}=require('node:test');
const assert=require('node:assert/strict');
const {approvedRows,modelResults,relativeSource,previewUrl}=require('./approved-training.js');

test('all current acceptances appear, including blocked; stale decisions and rejects do not',()=>{
  const rows=['a','b','c','d'].map(id=>({id,raw_sha256:id,model:'sunburst',audit_status:'reliable'}));
  const decisions={a:{raw_sha256:'a',decision:'accepted'},b:{raw_sha256:'b',decision:'accepted'},c:{raw_sha256:'old',decision:'accepted'},d:{raw_sha256:'d',decision:'rejected'}};
  const result=approvedRows({rows},{decisions,manifest:{candidates:[{id:'a'}],blocked:[{id:'b',reason:'check'}]}});
  assert.deepEqual(result.map(r=>r.id),['a','b']);
  assert.ok(result[0].candidate);assert.equal(result[1].blocked.reason,'check');
});

test('direct denominator includes every output and excludes recovery and user geometry changes',()=>{
  const rows=['a','b','c','d'].map(id=>({id,raw_sha256:id,model:'sunburst',audit_status:'reliable'}));
  rows[1].registration_recovery={id:'proposal'};rows[1].audit_status='unavailable';
  const decisions=Object.fromEntries(rows.map(r=>[r.id,{raw_sha256:r.id,decision:'accepted'}]));
  decisions.c.target_trim={top:true,bottom:false};decisions.d={raw_sha256:'old',decision:'accepted'};
  const result=modelResults({rows},{decisions});
  assert.equal(result[0].total,4);assert.equal(result[0].accepted,3);assert.equal(result[0].pending,1);assert.equal(result[0].direct,1);
  assert.equal(result[0].initialRegistration,3);
  assert.equal(result[1].total,0);
});

test('source overlay is relative to retained target and preview binds exact decision revision',()=>{
  const row={id:'one',raw_sha256:'hash',review:{revision:7},region:{polygon:[[100,300],[900,300],[900,900],[100,900]]},candidate:{conditioning:{target_box_px:[100,300,900,1500]}}};
  assert.deepEqual(relativeSource(row),[[0,0],[800,0],[800,600],[0,600]]);
  assert.match(previewUrl(row,'input'),/sha=hash&revision=7$/);
});
