// Verificador público de consistencia documental. No ejecuta la campaña privada.
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {dirname,join} from 'node:path';
const root=dirname(fileURLToPath(import.meta.url));
const read=name=>readFileSync(join(root,name));
const parse=name=>JSON.parse(read(name));
const check=(v,why)=>{if(!v)throw Error(why);};
const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const median=xs=>{const a=[...xs].sort((a,b)=>a-b),n=a.length;return n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2;};
function metrics(a){const sorted=[...a].sort((a,b)=>a-b),m=median(a);return {n:a.length,median_ns:m,mad_ns:median(a.map(x=>Math.abs(x-m))),p95_ns:sorted[Math.ceil(a.length*.95)-1],max_ns:Math.max(...a),min_ns:Math.min(...a)};}
function verify(spec,state){
  check(state.id===spec.id,'identidad');
  check(state.source_commit===spec.source_commit,'fuente');
  check(state.run_id===spec.run_id,'ejecución');
  check(equal(state.pending,spec.pending),'pendientes alterados');
  for(const k of ['sv_core_executed','ai_model_executed','r1_integrated','production_authorized'])check(state[k]===false,'alcance ampliado: '+k);
  check(state.cases.length===spec.cases.length,'testigos ausentes');
  const seen=new Set();let passed=0;
  for(const expected of spec.cases){
    const rows=state.cases.filter(x=>x.id===expected.id);
    check(rows.length===1&&!seen.has(expected.id),'identidad duplicada');seen.add(expected.id);
    const r=rows[0];check(r.observed===expected.expected,'observación '+r.id);
    check(r.recovery===(expected.recovery_required?true:null),'recuperación '+r.id);
    check(r.effect===['good_es','good_en'].includes(r.legacy_id),'efecto '+r.id);
    check(r.sentinel_unchanged===(expected.sentinel_checked?r.legacy_id!=='control_write':null),'centinela '+r.id);
    check(Number.isFinite(r.elapsed_us)&&r.elapsed_us>=0,'duración');
    passed++;
  }
  check(state.passed===passed&&passed===33,'recuento');
  check(equal(state.companion,{entered:true,live_before:true,live_after:true,accepted:true}),'concurrencia');
  check(equal(state.journal,spec.expected_journal),'registro de efectos');
  for(const route of ['direct','ipc','cold']){
    const a=state.samples_ns[route];check(a.length===(route==='cold'?25:1500),'muestras');
    check(a.every(x=>Number.isSafeInteger(x)&&x>0),'unidad de medida');
    check(equal(metrics(a),state.metrics[route]),'aritmética '+route);
  }
  check(state.first_run.companion_status==='OBSERVADOR_LIMITADO_SUCEDIDO','antecedente borrado');
  check(state.first_run.superseded_by===spec.cases.find(x=>x.legacy_id==='companion').id,'sucesión');
  return passed;
}
try {
  const manifest=parse('manifiesto.json'),spec=parse('testigos.json'),state=parse('estado.json');
  check(manifest.id==='LAB-2026-033','manifiesto');
  check(equal(manifest.public_files.map(x=>x.path).sort(),['estado.json','testigos.json','verificar.mjs']),'objetos obligatorios');
  for(const f of manifest.public_files){
    const b=read(f.path);check(b.length===f.bytes&&createHash('sha256').update(b).digest('hex')===f.sha256,'huella '+f.path);
  }
  check(manifest.source_commit===spec.source_commit&&manifest.run_id===spec.run_id,'ligadura de manifiesto');
  console.log('CONFORME: '+verify(spec,state)+' testigos documentados; 3025 muestras recalculadas.');
  if(process.argv.includes('--autoprueba')){
    const mutations=[
      s=>{s.cases.find(x=>x.legacy_id==='omission').observed='ACCEPT';},
      s=>s.cases.pop(),
      s=>{s.cases[1]=structuredClone(s.cases[0]);},
      s=>s.pending.pop(),
      s=>{s.passed=34;},
      s=>{s.source_commit='otro';},
      s=>{s.cases.find(x=>x.legacy_id==='omission').effect=true;}
    ];
    let rejected=0;
    for(const mutate of mutations){const s=structuredClone(state);mutate(s);try{verify(spec,s);}catch{rejected++;}}
    check(rejected===mutations.length,'control negativo no detectado');
    console.log('CONFORME: 7/7 alteraciones documentales rechazadas por predicados, sin depender de sus huellas.');
  }
  console.log('Alcance: identidad y consistencia públicas; la ejecución material está identificada bajo custodia. No acredita seguridad universal ni verdad profesional.');
} catch(e){console.error('NO CONFORME: '+e.message);process.exit(1);}
