const aspectOrder=['3:4','4:3','9:16','16:9','2:3','3:2','4:5','5:4','1:2','2:1','1:3','3:1'];
function aspectFor(row){
  if(row.aspect_ratio)return row.aspect_ratio;
  const [w,h]=row.dimensions||[];if(!(w>0&&h>0))return '';
  const gcd=(a,b)=>b?gcd(b,a%b):a,d=gcd(w,h);return `${w/d}:${h/d}`;
}
function aspectMatches(row){const ratio=$('#aspect')?.value;return !ratio||aspectFor(row)===ratio}
function updateAspectGroups(){
  const select=$('#aspect'),label=$('#aspect-label');if(!select||!label)return;
  label.hidden=!data.aspect_groups;if(!data.aspect_groups){select.value='';return}
  let saved='';try{saved=localStorage.getItem('artworker-aspect-group')||''}catch{}
  const ratio=select.value||new URLSearchParams(location.search).get('ratio')||saved||aspectOrder[0];
  select.innerHTML=aspectOrder.map(value=>`<option value="${value}">${value} · ${data.rows.filter(r=>aspectFor(r)===value).length} görsel</option>`).join('');
  select.value=aspectOrder.includes(ratio)?ratio:aspectOrder[0];
}
function aspectSummary(){
  const ratio=$('#aspect')?.value;if(!ratio)return '';
  const rows=data.rows.filter(r=>aspectFor(r)===ratio),pending=rows.filter(r=>reviewDecision(r)==='pending').length;
  return `${ratio} grubu · ${rows.length-pending} değerlendirildi / ${rows.length} · ${pending} incelenmedi veya kararsız`;
}
function updateAspectSummary(){const el=$('#aspect-summary');if(el)el.textContent=aspectSummary()}
function nextAspect(){
  const ratio=$('#aspect')?.value;if(!ratio)return null;
  return aspectOrder.slice(aspectOrder.indexOf(ratio)+1).find(value=>data.rows.some(r=>aspectFor(r)===value))||null;
}
function advanceAspectGroup(){
  const next=nextAspect();if(!next)return;
  $('#aspect').value=next;try{localStorage.setItem('artworker-aspect-group',next)}catch{}page=0;render();
  fastQueue(selected().filter(r=>reviewDecision(r)==='pending'));
}
if(typeof module!=='undefined')module.exports={aspectOrder,aspectFor};
