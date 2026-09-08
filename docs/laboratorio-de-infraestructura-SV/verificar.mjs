#!/usr/bin/env node
/** Verificación pública SV. Sólo usa Node.js y los cuatro archivos publicados.
 * No ejecuta los prototipos ni consulta credenciales o repositorios privados.
 * Los predicados proceden de testigos.json; estado.json sólo aporta observaciones.
 */
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {resolve, dirname} from 'node:path';
import {isDeepStrictEqual as equal} from 'node:util';

const STATES=['PASS','FAIL','BLOCKED','ERROR','NOT_EXECUTED'];
const requireThat=(ok,why)=>{if(!ok)throw Error(why);};
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const unique=(xs,label)=>requireThat(new Set(xs).size===xs.length,'Identidad repetida: '+label);
const count=rows=>Object.fromEntries(STATES.map(s=>[s,rows.filter(r=>r.status===s).length]));
const sameSet=(a,b,label)=>{unique(a,label);unique(b,label);requireThat(equal([...a].sort(),[...b].sort()),'Inventario divergente: '+label);};
function predicate(rule,observation){
  requireThat(Object.hasOwn(observation,rule.field),'Observable ausente: '+rule.field);
  const actual=observation[rule.field];
  switch(rule.op){
    case 'eq': return equal(actual,rule.value);
    case 'ne': return !equal(actual,rule.value);
    case 'lte': case 'gt':
      requireThat(typeof actual==='number'&&Number.isFinite(actual),'Observable numérico inválido: '+rule.field);
      return rule.op==='lte'?actual<=rule.value:actual>rule.value;
    default:throw Error('Operador no declarado: '+rule.op);
  }
}
function contract(t){
  requireThat(t.schema==='sv-infrastructure-public-witnesses/1','Esquema de especificación');
  unique(t.witnesses.map(w=>w.id),'especificación');
  unique(t.campaigns.map(c=>c.id),'tandas');
  unique(t.negative_mutations.map(m=>m.id),'mutaciones');
  requireThat(t.budgets.samples===t.budgets.warmup_samples+t.budgets.useful_samples,'Aritmética de muestras');
  requireThat(t.budgets.production_latency_threshold===null,'No existe umbral productivo constituido');
  const budgetFields={elapsed_ns:'deadline_ns',control_elapsed_ns:'deadline_ns',reader_max_seconds:'reader_limit_seconds',reader_count:'data_readers',samples:'samples',useful:'useful_samples',cancellations:'cancellations'};
  for(const w of t.witnesses){
    const run=t.campaigns.find(c=>c.id===w.campaign_id);
    requireThat(run&&w.id===w.legacy_id+'@'+run.id,'Identidad sin sufijo de tanda: '+w.id);
    for(const f of ['obligation','condition','specification_origin'])requireThat(typeof w[f]==='string'&&w[f].length>0,'Especificación vacía: '+f);
    requireThat(w.negative_control?.expected==='REJECT','Control negativo sin obligación de rechazo');
    requireThat(['all_public_observations','custodied_result_only'].includes(w.oracle.kind),'Oráculo desconocido');
    requireThat((w.oracle.rules.length>0)===(w.oracle.kind==='all_public_observations'),'Tipo de oráculo inconsistente');
    if(w.oracle.kind==='custodied_result_only')requireThat(w.group==='Controlador / #9','No se permite degradar una comprobación pública a declaración de custodia');
    unique(w.oracle.rules.map(r=>r.field),'campos del oráculo '+w.id);
    for(const rule of w.oracle.rules){
      requireThat(['eq','ne','lte','gt'].includes(rule.op),'Operador desconocido');
      if(budgetFields[rule.field]&&['lte','eq'].includes(rule.op))requireThat(equal(rule.value,t.budgets[budgetFields[rule.field]]),'Presupuesto incoherente: '+rule.field);
    }
  }
  requireThat(t.claim_limits.runtime_reexecuted===false&&t.claim_limits.three_guarantees_closed===false&&t.claim_limits.public_hashes_are_execution_proof===false,'Afirmación fuera del alcance de regularización');
}
export function verify(m,t,s){
  contract(t);
  requireThat(m.schema==='sv-infrastructure-public-manifest/1'&&s.schema==='sv-infrastructure-public-state/2','Esquema de paquete');
  requireThat(m.package_id===t.package_id&&s.package_id===t.package_id,'Paquete incoherente');
  requireThat(equal(m.campaigns,t.campaigns),'Identidades de campaña divergentes');
  unique(m.artifacts.map(a=>a.execution_id+'/'+a.name),'artefactos');
  for(const a of m.artifacts){
    const c=t.campaigns.find(c=>c.id===a.execution_id);
    requireThat(c&&a.source_commit===c.source_commit&&/^[0-9a-f]{40}$/.test(a.source_commit),'Fuente de artefacto incompatible');
    requireThat(Number.isSafeInteger(a.bytes)&&a.bytes>=0&&/^[0-9a-f]{64}$/.test(a.sha256),'Tamaño o huella inválidos');
    requireThat(a.access==='private'&&a.name&&!a.name.startsWith('/')&&!a.name.split('/').includes('..'),'Ruta de custodia inválida');
  }
  for(const c of t.campaigns){
    const a=m.artifacts.filter(a=>a.execution_id===c.id);
    requireThat(a.length===c.inventory_count,'Número de artefactos incoherente');
    const zip=a.find(a=>a.name===c.archive);
    requireThat(zip&&zip.bytes===c.archive_bytes&&zip.sha256===c.archive_sha256,'Archivo de tanda incompatible');
  }
  const rows=[...s.cases];const main=new Map(s.cases.map(c=>[c.id,c]));
  for(const r of s.reconciliations){
    const resolved=r.cases.map(c=>c.case_ref?main.get(c.case_ref):c);
    requireThat(resolved.every(Boolean),'Referencia de conciliación ausente');
    sameSet(resolved.map(c=>c.id),t.reconciliation_ids[r.execution_id],'conciliación '+r.execution_id);
    requireThat(resolved.every(c=>c.campaign_id===r.execution_id),'Conciliación entre tandas diferentes');
    requireThat(equal(count(resolved),r.counts),'Recuento de conciliación');
    requireThat(r.included_in_panel_totals===false&&r.phase02==='NOT_EXECUTED','Alcance de #8');
    rows.push(...r.cases.filter(c=>!c.case_ref));
  }
  sameSet(s.cases.map(c=>c.id),t.panel_ids,'panel');
  sameSet(rows.map(c=>c.id),t.witnesses.map(w=>w.id),'todos los testigos');
  sameSet(s.reconciliations.map(r=>r.execution_id),Object.keys(t.reconciliation_ids),'conciliaciones');
  const all=new Map(rows.map(c=>[c.id,c]));
  unique(s.pending_case_ids,'pendientes por testigo');
  for(const id of s.pending_case_ids)requireThat(all.has(id)&&all.get(id).status!=='PASS','Pendiente inexistente o superado');
  requireThat(equal(s.pending_obligations,t.pending_obligations),'Obligación pendiente retirada o alterada');
  requireThat(equal(s.pending,s.pending_obligations.map(p=>p.description)),'Prosa de pendientes divergente');
  for(const w of t.witnesses){
    const c=all.get(w.id);
    requireThat(c.campaign_id===w.campaign_id&&c.legacy_id===w.legacy_id&&c.phase===w.group,'Identidad del testigo alterada');
    requireThat(STATES.includes(c.status)&&c.negative_control===w.negative_campaign_control,'Clase o estado alterado');
    requireThat(m.artifacts.some(a=>a.execution_id===c.campaign_id&&a.name===w.evidence.artifact),'Evidencia sin manifiesto');
    requireThat(c.evidence_ref===w.evidence.artifact,'Referencia de evidencia divergente');
    if(['PASS','FAIL'].includes(c.status)&&w.oracle.kind==='all_public_observations'){
      const results=w.oracle.rules.map(r=>predicate(r,c.observations));
      requireThat((c.status==='PASS')===results.every(Boolean),'Resultado contrario al oráculo: '+c.id);
    }
    if(c.status!=='PASS'){
      requireThat(s.pending_case_ids.includes(c.id)||c.superseded_by?.length>0,'Fallo retirado sin pendiente ni sucesión: '+c.id);
      if(c.superseded_by){
        requireThat(equal(c.superseded_by,t.successions[c.id]?.superseded_by)&&c.basis===t.successions[c.id]?.basis,'Sucesión sin fundamento declarado');
        unique(c.superseded_by,'sucesores');
        for(const id of c.superseded_by)requireThat(id!==c.id&&all.get(id)?.status==='PASS','Sucesor ausente, propio o no superado');
      }
    }
  }
  const totals=count(s.cases);requireThat(equal(s.totals,totals),'Total del panel');
  const recent=s.cases.filter(c=>c.campaign_id==='34022055251'&&c.status==='PASS').length;
  const historical=s.cases.filter(c=>c.campaign_id==='34010741314'&&c.status==='PASS').length;
  const label=`${totals.PASS} PASS · ${totals.FAIL} FAIL · ${totals.BLOCKED} BLOCKED · ${totals.ERROR} ERROR. Panel: ${recent} PASS de #9 y ${historical} PASS de continuación histórica. La conciliación de #8 se contabiliza aparte.`;
  requireThat(s.state===label,'Resumen state divergente');
  for(const [nested,select] of [[s.continuation.results,c=>c.campaign_id==='34010741314'],[s.phase02.results,c=>c.phase==='0.2 / repetición #9']]){
    sameSet(nested.map(c=>c.id),s.cases.filter(select).map(c=>c.id),'vista derivada');
    for(const c of nested)requireThat(c.status===all.get(c.id).status,'Estado de vista derivada');
  }
  requireThat(s.three_guarantees_closed===false&&s.regularization.new_campaign_executions===0,'Cierre o ejecución no acreditados');
  return {panel:totals,panel_records:s.cases.length,specified_witnesses:t.witnesses.length,private_artifacts_identified:m.artifacts.length,
    private_artifacts_reverified_by_public_reader:0,public_observation_predicates:rows.filter(c=>['PASS','FAIL'].includes(c.status)&&t.witnesses.find(w=>w.id===c.id).oracle.kind==='all_public_observations').length,
    limitation:'Coherencia del paquete y de observaciones publicadas; no reproducción pública de los artefactos privados ni cierre nuclear.'};
}
function mutate(id,m,t,s){
 const bad=()=>s.cases.find(c=>c.status==='FAIL');
 const deadline=()=>s.cases.find(c=>c.legacy_id==='A05-positive');
 switch(id){
  case 'M01-total':s.totals.PASS--;break;
  case 'M02-state':s.state='48 PASS';break;
  case 'M03-duplicate-id':s.cases[1].id=s.cases[0].id;break;
  case 'M04-unknown-case':s.cases[0].id='INVENTADO';break;
  case 'M05-missing-case':s.cases.pop();break;
  case 'M06-retired-without-successor':delete bad().superseded_by;break;
  case 'M07-unknown-successor':bad().superseded_by=['INEXISTENTE'];break;
  case 'M08-self-successor':bad().superseded_by=[bad().id];break;
  case 'M09-run8-count':s.reconciliations[0].counts.PASS=17;break;
  case 'M10-run8-missing-case':s.reconciliations[0].cases.pop();break;
  case 'M11-source-commit':m.artifacts[0].source_commit='0'.repeat(40);break;
  case 'M12-archive-hash':m.artifacts.find(a=>a.name===t.campaigns[0].archive).sha256='0'.repeat(64);break;
  case 'M13-negative-control':s.cases.find(c=>c.negative_control).negative_control=false;break;
  case 'M14-oracle-swap':s.cases.find(c=>c.legacy_id==='A07-before').observations=s.cases.find(c=>c.legacy_id==='A07-after').observations;break;
  case 'M15-false-pass':bad().status='PASS';break;
  case 'M16-drop-pending':s.pending_obligations.pop();break;
  case 'M17-change-budget':deadline().observations.elapsed_ns=t.budgets.deadline_ns+1;break;
  case 'M18-guarantee-closure':s.three_guarantees_closed=true;break;
  case 'M19-downgrade-public-check':t.witnesses[0].oracle={kind:'custodied_result_only',rules:[]};break;
  case 'M20-missing-artifact':m.artifacts.pop();break;
  default:throw Error('Mutación sin implementación: '+id);
 }
}
export function selftest(m,t,s){
 verify(m,t,s);const results=[];
 for(const mutation of t.negative_mutations){
   const {id,expected_rejection}=mutation;
   requireThat(typeof expected_rejection==='string'&&expected_rejection.length>10,'Causa de rechazo no especificada');
   const [a,b,c]=structuredClone([m,t,s]);mutate(id,a,b,c);
   let cause=null;try{verify(a,b,c);}catch(e){cause=e.message;}
   requireThat(cause===expected_rejection,'Mutación superviviente o rechazada por otra causa: '+id+' ('+cause+')');
   results.push({id,status:'REJECTED',cause});
 }
 const observableControls=[];
 for(const w of t.witnesses.filter(w=>w.oracle.kind==='all_public_observations')){
   const c=s.cases.find(c=>c.id===w.id)||s.reconciliations.flatMap(r=>r.cases).find(c=>c.id===w.id);
   if(!['PASS','FAIL'].includes(c.status))continue;
   const o=structuredClone(c.observations);delete o[w.negative_control.field];
   let rejected=false;try{w.oracle.rules.map(r=>predicate(r,o));}catch(e){rejected=e.message==='Observable ausente: '+w.negative_control.field;}
   requireThat(rejected,'Control de observable insensible: '+w.id);observableControls.push(w.id);
 }
 return {directed_mutations:results,observable_absence_controls:observableControls.length,survivors:0,scope:'Pruebas del verificador sobre copias en memoria; archivos originales conservados.'};
}
async function main(){
 const args=process.argv.slice(2);const where=args.find(a=>!a.startsWith('--'));
 const base=where&&/^https:\/\//.test(where)?new URL(where.endsWith('/')?where:where+'/'):pathToFileURL(resolve(where||dirname(fileURLToPath(import.meta.url)))+'/');
 const bytes={};
 for(const f of ['manifiesto.json','testigos.json','estado.json','verificar.mjs']){
   const url=new URL(f,base);
   if(url.protocol==='file:')bytes[f]=await readFile(url);
   else{const r=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(30000)});requireThat(r.ok,'Archivo público no disponible: '+f);bytes[f]=Buffer.from(await r.arrayBuffer());}
 }
 const m=JSON.parse(bytes['manifiesto.json']),t=JSON.parse(bytes['testigos.json']),s=JSON.parse(bytes['estado.json']);
 sameSet(m.public_files.map(f=>f.name),['testigos.json','estado.json','verificar.mjs'],'archivos públicos sujetos a huella');
 for(const f of m.public_files)requireThat(f.bytes===bytes[f.name].length&&f.sha256===hash(bytes[f.name]),'Huella pública discordante: '+f.name);
 const result=verify(m,t,s);
 if(args.includes('--autoprueba'))result.sensitivity=selftest(m,t,s);
 result.verification='CONFORME_EN_ALCANCE_PUBLICO';
 result.manifest_sha256=hash(bytes['manifiesto.json']);
 console.log(JSON.stringify(result,null,2));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{console.error('RECHAZO: '+e.message);process.exitCode=1;});
