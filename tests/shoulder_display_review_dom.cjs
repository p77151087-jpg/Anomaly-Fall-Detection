// Synthetic DOM plumbing only. This never supplies human annotations for real footage.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('scripts/templates/shoulder-display-review-v2.html','utf8');
const code=html.split('<script>')[1].split('</script>')[0];
const clips=['fall-19','fall-09','fall-18','fall-30'].map((id,i)=>({sequence_id:id,video_sha256:'video'+i,reference_sha256:'ref'+i,video:id+'.mp4',fps:30,focus_seconds:[.1,.8],first_alarm_ms:500,current_alert_frames:4,timeline:Array.from({length:30},(_,i)=>({sample_index:i,timestamp_ms:i*1000/30,source_frame_id:i+1,state:'normal'})),inherited_annotation:{kind:'human_instantaneous_uncertainty_bounds',onset_bounds_ms:[100,200],impact_bounds_ms:[500,700],skeleton_quality:'synthetic',notes:'synthetic',note:'synthetic only'}}));
const bundle={schema_version:1,bundle_sha256:'synthetic-only',clips};
class Element {constructor(){this.children=[];this.listeners={};this.value='';this.checked=false;this.disabled=false;this.textContent='';this.currentTime=0;this.duration=1;this.paused=true;this.files=[];}addEventListener(e,f){this.listeners[e]=f;}setAttribute(){}append(x){this.children.push(x);}pause(){this.paused=true;}play(){this.paused=false;return Promise.resolve();}click(){this.clicked=true;}}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};get('bundle').textContent=JSON.stringify(bundle);get('speed').value='1';
const cache=new Map(),downloads=[];let lastLink;
const sandbox={console,setTimeout:()=>1,clearTimeout(){},Blob:class {constructor(parts){this.parts=parts;}},URL:{createObjectURL(b){downloads.push(JSON.parse(b.parts[0]));return 'blob:synthetic';},revokeObjectURL(){}},localStorage:{getItem:k=>cache.get(k)??null,setItem:(k,v)=>cache.set(k,v)},document:{getElementById:get,createElement(){return lastLink=new Element();}}};
vm.createContext(sandbox);vm.runInContext(code,sandbox);
const run=s=>vm.runInContext(s,sandbox);
assert.equal(get('export').disabled,true);assert.match(get('progress').textContent,/0\/4/);
assert.equal(run('state.clips.length'),4);assert.equal(run('state.clips.every(r=>!r.reviewed&&!r.timing_relation)'),true);
get('export').onclick();assert.equal(downloads.length,0);
for(let i=0;i<4;i++){get('tabs').children[i].onclick();get('timing').value='uncertain';get('pose').value='same_person_misaligned';get('clarity').value='slow_only';get('notes').value='synthetic fixture';get('reviewed').checked=true;get('reviewed').listeners.input();}
assert.match(get('progress').textContent,/4\/4/);assert.equal(get('export').disabled,true);
get('reviewer').value='SYNTHETIC AUTOMATION ONLY';get('attest').checked=true;get('attest').listeners.input();assert.equal(get('export').disabled,false);
get('export').onclick();assert.equal(downloads.length,1);assert.equal(lastLink.download,'shoulder-four-clips-human-review.json');
assert.equal(downloads[0].clips.length,4);assert.equal(downloads[0].inherited_annotations.length,4);assert.equal(downloads[0].new_temporal_ground_truth,false);
get('draft').onclick();assert.equal(downloads.length,2);assert.equal(downloads[1].draft,true);assert.equal(downloads[1].human_attested,false);
get('tabs').children[0].onclick();assert.equal(get('pose').value,'same_person_misaligned');
get('alarm').onclick();assert.equal(get('video').currentTime,.5);get('next').onclick();assert.equal(Number(get('scrub').value),16);
const altered=JSON.parse(JSON.stringify(downloads[0]));altered.clips[0].video_sha256='bad';sandbox.altered=altered;assert.equal(run('compatible(altered)'),false);
// WebKit may expose a seek at sample 137 as 4.566666 instead of 4.5666666667.
get('video').duration=10;get('video').currentTime=4.566666;run("bundle.clips[0].timeline=Array.from({length:200},(_,i)=>({sample_index:i,timestamp_ms:i*1000/30,source_frame_id:i+1,state:'normal'}))");
run('updateTime()');assert.equal(Number(get('scrub').value),137);get('scrub').max=199;get('next').onclick();assert.equal(Number(get('scrub').value),138);get('prev').onclick();assert.equal(Number(get('scrub').value),137);
console.log(JSON.stringify({evidence:'synthetic_dom',assertions:(fs.readFileSync(__filename,'utf8').match(/assert\.(?:equal|match)\(/g)||[]).length,real_human_annotations_created:0,passed:true}));
