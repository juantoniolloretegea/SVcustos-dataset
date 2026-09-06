'use strict';
const el=id=>document.getElementById(id);
let data=null;
function draw(){
  const mode=el('filter').value;
  el('cases').replaceChildren();
  for(const c of data.cases){
    if(mode==='negative'&&!c.negative_control||mode==='test'&&c.negative_control)continue;
    const row=document.createElement('tr');
    const status=c.status==='PASS'?(c.negative_control?'Defecto detectado':'Testigo superado'):c.status;
    for(const [i,text] of [c.phase,c.id,c.description,status].entries()){
      const td=document.createElement('td');td.textContent=text;if(i===3)td.className=c.status==='PASS'?'ok':'bad';row.append(td);
    }el('cases').append(row);
  }
}
fetch('estado.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('estado');return r.json();}).then(v=>{
  data=v;el('campaign-state').textContent=v.state;
  const counts={};for(const c of v.cases)counts[c.status]=(counts[c.status]||0)+1;
  el('counts').textContent=v.cases.length+' registros históricos y actuales: '+Object.entries(counts).map(([s,n])=>n+' '+s).join(' · ')+'. Los controles negativos están incluidos; BLOCKED no es una prueba superada.';
  el('hardware').textContent=v.hardware;el('performance').textContent=v.performance;
  el('commit').textContent=v.commit||'Pendiente';el('run').textContent=v.run||'Pendiente';
  el('pending').replaceChildren();for(const p of v.pending){const li=document.createElement('li');li.textContent=p;el('pending').append(li);}draw();
}).catch(()=>{el('campaign-state').textContent='Resultados no disponibles: no se declara éxito.';});
el('filter').addEventListener('change',()=>{if(data)draw();});
