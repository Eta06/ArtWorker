// Read-only view: source pixels and targets follow the saved admission manifest.
function approvedRows(data,snapshot){
  const candidates=new Map(snapshot.manifest.candidates.map(r=>[r.id,r]));
  const blocked=new Map(snapshot.manifest.blocked.map(r=>[r.id,r]));
  return data.rows.flatMap(row=>{
    const review=snapshot.decisions[row.id];
    if(review?.raw_sha256!==row.raw_sha256||review.decision!=='accepted')return [];
    return [{...row,review,candidate:candidates.get(row.id),blocked:blocked.get(row.id)}];
  });
}
function modelResults(data,snapshot){
  return ['sunburst','nano'].map(model=>{
    const rows=data.rows.filter(r=>r.model===model),count={model,total:rows.length,accepted:0,rejected:0,pending:0,direct:0,initialRegistration:0};
    for(const row of rows){
      const review=snapshot.decisions[row.id],same=review?.raw_sha256===row.raw_sha256;
      const decision=same?review.decision:'pending';count[decision]++;
      count.initialRegistration+=row.audit_status==='reliable'?1:0;
      const changed=same&&(review.square_fit===true||[review.crop_extensions,review.target_trim].some(value=>Object.values(value||{}).some(Boolean)));
      if(decision==='accepted'&&!changed&&!row.registration_recovery&&row.audit_status==='reliable')count.direct++;
    }
    return count;
  });
}
function previewUrl(row,kind,route=typeof location!=='undefined'?'/'+(location.pathname.split('/')[1]||'sumbul'):'/sumbul'){
  return route+'/training-preview/'+encodeURIComponent(row.id)+'/'+kind+'.jpg?sha='+encodeURIComponent(row.raw_sha256)+'&revision='+row.review.revision;
}
function relativeSource(row){
  const target=row.candidate?.conditioning?.target_box_px;
  const crop=row.candidate?.conditioning,box=crop?.square_fit&&crop.reference_box_px;
  const polygon=box?[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]:row.region?.polygon;
  return target&&polygon?polygon.map(([x,y])=>[x-target[0],y-target[1]]):null;
}
if(typeof module!=='undefined')module.exports={approvedRows,modelResults,previewUrl,relativeSource};
