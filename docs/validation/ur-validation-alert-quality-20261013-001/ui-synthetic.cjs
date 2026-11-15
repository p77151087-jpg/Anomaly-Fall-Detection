// Offline fake DOM tests: never modifies a browser or creates real human annotations.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const template = fs.readFileSync(path.join(process.cwd(),'scripts/templates/validation-alert-review.html'),'utf8');
const script = template.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const clips = ['fall-18','fall-09','fall-30'].map(sequence_id=>({sequence_id,frames:[1,2,3,4].map(source_frame_id=>({source_frame_id,timestamp_ms:(source_frame_id-1)*33,width:640,height:480,poses:[],image:'fake'}))}));
const elements=new Map();let asserts=0,downloads=[];
const check=(condition)=>{assert.ok(condition);asserts++;};
function element(id=''){return{id,value:'',checked:false,disabled:false,textContent:'',children:[],dataset:{},events:{},className:'',append(c){this.children.push(c);},addEventListener(n,f){this.events[n]=f;},click(){if(id==='link')downloads.push({name:this.download,blob:blobStore.get(this.href)});else this.onclick?.();},getContext(){return{drawImage(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},fill(){}};}};}
function get(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id);}
get('bundle').textContent=JSON.stringify({bundle_sha256:'synthetic-only',clips});
const marks=['onsetLo','onsetHi','impactLo','impactHi'].map(id=>{const e=element();e.dataset.mark=id;return e;});
const document={getElementById:get,createElement(t){return element(t==='a'?'link':'');},querySelector(s){return marks.find(e=>s.includes(e.dataset.mark));},querySelectorAll(){return marks;}};
const storage=new Map(),blobStore=new Map();let urlIndex=0;
class Image{set src(value){this.onload?.();}}
const context=vm.createContext({document,Image,Blob,URL:{createObjectURL(b){const url='blob:'+urlIndex++;blobStore.set(url,b);return url;},revokeObjectURL(){}},localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},setTimeout:()=>1,clearTimeout(){},console});
vm.runInContext(script,context);
const run=code=>vm.runInContext(code,context);
const pull=()=>run('pull()');
const clickTab=i=>get('tabs').children[i].onclick();
check(get('export').disabled && get('progress').textContent.includes('0/3'));
check(get('onsetLo').disabled && get('impactHi').disabled);
get('export').onclick();check(downloads.length===0);
get('onsetStatus').value='bounded';pull();check(!get('onsetLo').disabled);
get('onsetLo').value='2';get('onsetHi').value='3';pull();
get('onsetStatus').value='uncertain';pull();check(get('onsetLo').value==='' && get('onsetLo').disabled);
check(run('state.clips[0].onset.bounds_frame_ids===null'));
get('onsetStatus').value='bounded';get('onsetLo').value='4';get('onsetHi').value='2';get('impactStatus').value='uncertain';get('quality').value='uncertain';get('reviewed').checked=true;pull();
check(run('checkRow(state.clips[0],bundle.clips[0])!==null'));
get('onsetLo').value='2';get('onsetHi').value='3';pull();
check(run('checkRow(state.clips[0],bundle.clips[0])===null'));
check(get('progress').textContent.includes('1/3'));
clickTab(1);check(get('clipTitle').textContent.includes('fall-09') && get('onsetStatus').value==='');
get('slider').value='3';get('slider').oninput();check(get('next').disabled && get('frameLabel').textContent.includes('第 4 幀'));
get('prev').onclick();check(!get('next').disabled && get('frameLabel').textContent.includes('第 3 幀'));
get('onsetStatus').value='uncertain';get('impactStatus').value='outside_clip';get('quality').value='occluded_or_cropped';get('reviewed').checked=true;pull();
check(get('progress').textContent.includes('2/3') && get('export').disabled);
clickTab(2);get('onsetStatus').value='bounded';get('onsetLo').value='1';get('onsetHi').value='2';get('impactStatus').value='bounded';get('impactLo').value='3';get('impactHi').value='4';get('quality').value='mostly_matches';get('reviewed').checked=true;pull();
check(get('progress').textContent.includes('3/3') && get('export').disabled);
get('reviewer').value='SYNTHETIC TEST';pull();check(get('export').disabled);
get('attest').checked=true;pull();check(!get('export').disabled);
get('export').onclick();check(downloads.length===1 && downloads[0].name==='validation-three-clips-human-review.json');
get('draft').onclick();check(downloads.length===2 && downloads[1].name==='validation-three-clips-draft.json');
clickTab(0);check(get('onsetLo').value===2 || get('onsetLo').value==='2');
check(storage.size===1 && [...storage.keys()][0].endsWith('synthetic-only'));
(async()=>{const exported=JSON.parse(await downloads[0].blob.text());const draft=JSON.parse(await downloads[1].blob.text());check(exported.human_attested===true && exported.clips.length===3);check(draft.human_attested===false && draft.draft===true);
const filename=path.join(__dirname,'ui-synthetic.json');fs.writeFileSync(filename,JSON.stringify({passed:true,assertions:asserts,evidence_tier:'synthetic_fake_DOM_not_browser',actual_human_annotations_created:0},null,2),{flag:'wx'});console.log(JSON.stringify({passed:true,assertions:asserts}));})().catch(e=>{console.error(e);process.exitCode=1;});
