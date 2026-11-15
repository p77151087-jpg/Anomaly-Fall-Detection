// Offline VM tests with synthetic DOM and points. Never attest/export in a real browser.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const template = fs.readFileSync(path.join(root, 'src/fall_detect/visualization/joint_localization_review.html'), 'utf8');
const fixture = {bundle_sha256:'synthetic-only',diagnostic_id:'synthetic-ui',sequence_id:'synthetic',split:'validation',pose_array_sha256:'synthetic',pose_metadata_sha256:'synthetic',source_manifest_sha256:'synthetic',names:Array.from({length:13},(_,i)=>'joint'+i),zh_names:Array.from({length:13},(_,i)=>'關節'+i),edges:[],required_joint_indices:[7,8,9,10,11,12],frames:[44,47,49].map(source_frame_id=>({source_frame_id,track_id:1,timestamp_ms:source_frame_id*33,image:'synthetic:no-network',image_sha256:'synthetic',frame_size_wh:[640,480],keypoints:Array.from({length:13},(_,i)=>[300+i,300+i,0.8])}))};
const script = template.replace('__DATA__',JSON.stringify(fixture)).match(/<script>([\s\S]*)<\/script>/)[1];
const storage = new Map();
function create(){
  const elements = new Map();
  const ctx = new Proxy({}, {get:(target,key)=>key in target ? target[key] : ()=>{}});
  function el(key){if(!elements.has(key))elements.set(key,{value:'',textContent:'',disabled:false,checked:false,width:key==='#zoom'?510:640,height:key==='#zoom'?720:480,append(){},getContext(){return ctx;},getBoundingClientRect(){return {left:0,top:0,width:255,height:360};}});return elements.get(key);}
  const document={querySelector:el,createElement(){return {value:'',textContent:''};}};
  class Image {set src(value){this.source=value;if(this.onload)this.onload();}}
  const context=vm.createContext({document,Image,console,Date,JSON,Math,Array,Error,localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)}});
  vm.runInContext(script,context);
  return {context,el,run:source=>vm.runInContext(source,context)};
}
let checks=0;
function check(value){assert.equal(value,true);checks++;}
let ui=create();
check(ui.run("count()===0 && draft.flat().every(p=>p.status==='unreviewed'&&p.xy===null)"));
check(ui.el('#export').disabled && !ui.el('#attest').checked);
ui.el('#relocate').onclick();
check(ui.run("count()===0 && current().status==='unreviewed'"));
check(ui.el('#error').textContent.includes('請先'));
ui.el('#zoom').onpointerdown({clientX:127.5,clientY:180});
check(ui.run("picked[0]===345 && picked[1]===360 && count()===0"));
ui.el('#relocate').onclick();
check(ui.run("current().status==='mislocated_visible' && current().xy[0]===345 && current().xy[1]===360 && count()===1"));
check(!ui.el('#attest').checked && ui.el('#export').disabled);
ui.el('#undo').onclick();
check(ui.run("current().xy===null && count()===0"));
ui.el('#uncertain').onclick();
check(ui.run("current().status==='not_visible_or_uncertain' && current().xy===null && count()===1"));
ui=create(); // New fake DOM uses saved synthetic draft; no actual browser storage.
check(ui.run("current().status==='not_visible_or_uncertain' && count()===1"));
check(!ui.el('#attest').checked);
ui.el('#nextFrame').onclick();ui.el('#nextFrame').onclick();
check(ui.el('#nextFrame').disabled && ui.el('#frame').textContent.includes('49'));
ui.el('#prevFrame').onclick();ui.el('#prevFrame').onclick();
check(ui.el('#prevFrame').disabled && ui.el('#frame').textContent.includes('44'));
for(let i=0;i<3;i++){
  for(let j=7;j<13;j++){ui.el('#joint').value=String(j);ui.el('#joint').onchange();ui.el('#uncertain').onclick();}
  if(i<2)ui.el('#nextFrame').onclick();
}
check(ui.run("count()===18 && draft.flat().filter(p=>p.status==='unreviewed').length===21"));
check(ui.el('#export').disabled);
ui.el('#reviewer').value='SYNTHETIC TEST ONLY';ui.el('#reviewer').oninput();
check(ui.el('#export').disabled);
ui.el('#attest').checked=true;ui.el('#attest').onchange();
check(!ui.el('#export').disabled);
check(ui.run("exportPayload().frames.every(f=>f.points.slice(7).every(p=>p.xy===null&&p.status==='not_visible_or_uncertain'))"));
ui.el('#note').value='synthetic edit';ui.el('#note').oninput();
check(ui.el('#export').disabled && !ui.el('#attest').checked);
ui.el('#joint').value='0';ui.el('#joint').onchange();ui.el('#matches').onclick();
check(ui.run("count()===18 && current().status==='matches_original' && current().xy[0]===300"));
console.log(JSON.stringify({synthetic_ui:true,checks,passed:true,real_browser_attestation:false,real_file_export:false}));
