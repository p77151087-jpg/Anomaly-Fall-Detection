"""Append a local-only player for the already encoded comparison videos."""

import json

import validation_display_replay as replay


def main():
    path = replay.OUT / "index.html"
    if path.exists():
        raise replay.ContractError("Preserve existing video gallery")
    data = replay.read(replay.OUT / "summary.json")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Validation 實際警示畫面對照</title><style>
*{box-sizing:border-box}body{margin:0;background:#f0f4f8;color:#1e3043;font:16px/1.6 system-ui,"Microsoft JhengHei",sans-serif}main{max-width:1360px;margin:auto;padding:24px}h1{font-size:26px;margin:0}p{margin:8px 0}button,select{font:inherit;padding:7px 12px;border:1px solid #aebfcc;border-radius:7px;background:#fff;cursor:pointer}.active{background:#145375;color:#fff}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.card{background:white;border:1px solid #d7e1e9;border-radius:12px;padding:16px;margin-top:16px}video{display:block;width:100%;background:#111;margin:12px 0}.muted{color:#607486}.note{background:#fff5da;padding:12px;border-radius:7px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dde6ed;padding:8px;white-space:nowrap}.scroll{overflow:auto}a{color:#145375}#clock{font-weight:600}button:disabled{opacity:.4;cursor:default}button:focus-visible,a:focus-visible{outline:3px solid #dc9b25}
</style><main>
<h1>三段 Validation：demo 預設與 wide 畫面對照</h1>
<p>左側是 demo 預設模型，右側是 wide 參照。既有 RGB／骨架／分數／本人時間範圍重播；未重訓、未推論、未跑 Test，原顯示與品質規則不變。</p>
<p class="note"><strong>影片中的 ACTION</strong> 是獨立七類分類對照；<strong>AE EVENT HISTORY / ALERT RECORDED</strong> 是已發生事件；<strong>Current</strong> 才是目前 AE 可用狀態。三者分開看。本人開始／落地標的是不確定範圍，不能當精確時刻。</p>
<section class="card"><div class="row" id="tabs"></div><p class="muted" id="span"></p>
<div class="row"><label>觀看版本 <select id="view"><option value="compare">左右並排</option><option value="demo">demo 預設單片</option><option value="wide">wide 單片</option></select></label><label>速度 <select id="speed"><option value="1">正常</option><option value="0.5">半速</option><option value="0.25">四分之一</option></select></label></div>
<video id="player" controls playsinline preload="metadata" aria-label="模型警示對照影片"></video>
<div class="row"><button id="prev">上一幀</button><button id="next">下一幀</button><button id="event">跳到 wide 首次事件</button></div>
<p id="clock"></p><p class="muted">播放器時間是片段時間；上方格點時間及影片註記對應原影片。30Hz重採樣會重複或略過部分原幀，沒有補造未來影格。</p>
</section>
<section class="card"><h2>影片判定、當前警示與歷史紀錄</h2><div class="scroll"><table><thead><tr><th>影片</th><th>模型</th><th>影片判定</th><th>CSV首次事件秒</th><th>當前AE警示幀</th><th>歷史紅字幀</th><th>七類FALLING幀</th></tr></thead><tbody id="results"></tbody></table></div>
<p>wide 的 fall-18／30 雖有事件，觸發當下當前 AE 因品質不足而不可判定；歷史紅字仍保留。這不改寫影片TP，也不代表持續有良好骨架。fall-09兩模型都是5.900秒首次事件；是否落地前仍受本人界限不確定性限制。</p></section>
<section class="card"><h2>全部九支短片與紀錄</h2><div id="downloads"></div><p><a href="summary.csv">摘要CSV</a> · <a href="display-frames.csv">逐格點／顯示狀態CSV</a> · <a href="summary.json">來源與媒體時間JSON</a></p>
<p class="muted">此為內部診斷短片；正式 Notebook／五分鐘影片及公開上傳仍暫緩。35項synthetic與獨立解碼／像素驗證各自保存，不將重播當成新模型辨識改善。</p></section>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),ids=['fall-18','fall-09','fall-30'];let selected=ids[0];const player=$('player');
function info(){return data.videos.find(v=>v.sequence_id===selected&&v.model===$('view').value);}
function updateClock(){const v=info(),frame=Math.max(0,Math.min(v.frames-1,Math.floor(player.currentTime*30+1e-5)));$('clock').textContent=`媒體第 ${frame+1}/${v.frames} 幀｜原格點 ${((v.grid_start_ms+frame*1000/30)/1000).toFixed(6)} 秒`;$('prev').disabled=frame===0;$('next').disabled=frame===v.frames-1;}
function choose(){player.pause();const v=info();player.src=v.file;player.load();player.playbackRate=Number($('speed').value);$('span').textContent=`${selected}｜格點 ${(v.grid_start_ms/1000).toFixed(6)}–${(v.grid_end_ms/1000).toFixed(6)} 秒｜${v.frames} 幀／30fps`;Array.from($('tabs').children).forEach((b,i)=>b.className=ids[i]===selected?'active':'');updateClock();}
function seek(frame){const v=info();player.pause();frame=Math.max(0,Math.min(v.frames-1,frame));player.currentTime=(frame+.1)/30;updateClock();}
ids.forEach(sid=>{const b=document.createElement('button');b.textContent=sid;b.onclick=()=>{selected=sid;choose();};$('tabs').append(b);});
$('view').onchange=choose;$('speed').onchange=()=>player.playbackRate=Number($('speed').value);player.ontimeupdate=updateClock;player.onloadedmetadata=()=>{player.playbackRate=Number($('speed').value);updateClock();};
$('prev').onclick=()=>seek(Math.floor(player.currentTime*30+1e-5)-1);$('next').onclick=()=>seek(Math.floor(player.currentTime*30+1e-5)+1);$('event').onclick=()=>{const r=data.rows.find(r=>r.sequence_id===selected&&r.model==='wide');seek(Math.round((r.first_event_ms-info().grid_start_ms)*30/1000));};
for(const r of data.rows){const tr=document.createElement('tr');for(const value of [r.sequence_id,r.model==='demo'?'demo 預設':'wide 參照',r.video_outcome,r.first_event_ms===null?'無':(r.first_event_ms/1000).toFixed(6),`${r.current_ae_alert_frames}/${r.rendered_frames}`,`${r.history_recorded_frames}/${r.rendered_frames}`,`${r.action_FALLING_frames}/${r.rendered_frames}`]){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('results').append(tr);}
for(const sid of ids){const p=document.createElement('p');p.textContent=sid+'：';for(const [m,label] of [['compare','並排片'],['demo','demo 預設'],['wide','wide 參照']]){const a=document.createElement('a');a.href=data.videos.find(v=>v.sequence_id===sid&&v.model===m).file;a.textContent=label;p.append(a);p.append(document.createTextNode('　'));}$('downloads').append(p);}
choose();
</script></html>"""
    with path.open("x", encoding="utf8") as stream:
        stream.write(page.replace("__DATA__", payload))
    print(path)


if __name__ == "__main__":
    main()
