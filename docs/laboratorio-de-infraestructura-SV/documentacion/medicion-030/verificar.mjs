// LAB-2026-030: comprobación documental; no ejecuta ni cronometra SV.
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const root=new URL('./',import.meta.url);
const read=n=>readFileSync(new URL(n,root));
const json=n=>JSON.parse(read(n));
const require=(ok,why)=>{if(!ok)throw Error(why)};
function keys(x,expected){require(x&&typeof x==='object'&&!Array.isArray(x),'Objeto requerido');require(JSON.stringify(Object.keys(x).sort())===JSON.stringify([...expected].sort()),'Campos no declarados o ausentes')}
const spec=json('testigos.json');
function slope(x,y){const a=x.map(Math.log),b=y.map(Math.log),m=a.reduce((s,v)=>s+v)/a.length,n=b.reduce((s,v)=>s+v)/b.length;return a.reduce((s,v,i)=>s+(v-m)*(b[i]-n),0)/a.reduce((s,v)=>s+(v-m)**2,0)}
function validate(s){
 keys(s,['schema','id','fecha','tablas','datos_temporales','muestras_brutas_disponibles','ejecuciones_sv_nuevas','necesidad_refactorizacion_acreditada','suficiencia_universal_acreditada','pendientes','exponentes_bytes']);
 require(s.schema==='sv-lab-030-estado-v1'&&s.id==='LAB-2026-030'&&s.fecha==='2026-09-09','Identidad de registro');
 require(s.datos_temporales==='DECLARADOS_POR_INFORME_RECIBIDO'&&s.muestras_brutas_disponibles===false&&s.ejecuciones_sv_nuevas===0,'Atribución de evidencia');
 require(s.necesidad_refactorizacion_acreditada===false&&s.suficiencia_universal_acreditada===false,'Alcance no acreditado');
 require(JSON.stringify(s.tablas)===JSON.stringify(spec.tablas_transcritas),'Transcripción distinta de la especificación');
 require(JSON.stringify(s.pendientes)===JSON.stringify(['Muestras brutas y orden','Script adicional y tratamiento de ceros','Verificación del binario y entorno','CPU y RSS de la ejecución recibida']),'Pendientes alterados');
 require(Array.isArray(s.exponentes_bytes)&&s.exponentes_bytes.length===4,'Pendientes numéricas ausentes');
 return Object.entries(s.tablas).map(([family,rows],i)=>{
  const good=rows.filter(r=>r[2]>spec.criterio_exploratorio_ms);
  require(good.length===spec.puntos_superiores[i],'Clasificación del criterio');
  const exponent=good.length>=3?slope(good.map(r=>r[1]),good.map(r=>r[2])):null;
  require(exponent===null?s.exponentes_bytes[i]===null:typeof s.exponentes_bytes[i]==='number'&&Math.abs(s.exponentes_bytes[i]-exponent)<1e-12,'Exponente incorrecto');
  return {familia:family,puntos:good.length,exponente_bytes:exponent};
 });
}
try{
 const manifest=json('manifiesto.json');
 keys(manifest,['schema','id','alcance','fuente','artefactos']);
 require(manifest.schema==='sv-lab-030-manifiesto-v1'&&manifest.id==='LAB-2026-030','Manifiesto ajeno');
 require(JSON.stringify(manifest.artefactos.map(x=>x.ruta).sort())===JSON.stringify(['estado.json','punto-control.md','testigos.json','verificar.mjs']),'Conjunto de artefactos');
 for(const a of manifest.artefactos){keys(a,['ruta','bytes','sha256']);const b=read(a.ruta);require(b.length===a.bytes&&createHash('sha256').update(b).digest('hex')===a.sha256,'Huella o tamaño: '+a.ruta)}
 const state=json('estado.json');const result=validate(state);
 let rejected=0;
 if(process.argv.includes('--autoprueba')){
  for(const mutate of [s=>s.tablas.N02_objetos[2][2]=1.2,s=>s.exponentes_bytes[0]=1,s=>s.ejecuciones_sv_nuevas=40,s=>s.campo_ajeno=true]){
   const copy=structuredClone(state);mutate(copy);let failed=false;try{validate(copy)}catch{failed=true}require(failed,'Control negativo superviviente');rejected++;
  }
 }
 console.log(JSON.stringify({registro:'LAB-2026-030',consistencia_documental:'CONFORME',recalculo:result,controles_negativos_rechazados:rejected,tiempos_reproducidos:false,ejecuciones_sv:0,seguridad_material_acreditada:false}));
}catch(error){console.error('Comprobación documental rechazada: '+error.message);process.exitCode=1}
