// 一次性探针：用假麦克风在真页面里点「直接录」，看 10 秒自动停之后
// 会不会出现「要这段声音吗」那块（#recModal）。
// 用法：node tests/rec_probe.mjs [页面地址]
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const URL0 = process.argv[2] || 'http://127.0.0.1:7864/';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9333;
const profile = mkdtempSync(path.join(tmpdir(), 'vc-probe-'));

const chrome = spawn(CHROME, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter', '--disable-breakpad',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--disable-extensions',
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  '--autoplay-policy=no-user-gesture-required', '--disable-gpu',
  URL0,
], { stdio: 'ignore' });

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function pageTarget() {
  for (let i = 0; i < 80; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const t = list.find(t => t.type === 'page' && t.url.startsWith('http'));
      if (t) return t;
    } catch {}
    await sleep(250);
  }
  throw new Error('没等到页面 target');
}

const tgt = await pageTarget();
const ws = new WebSocket(tgt.webSocketDebuggerUrl);
let id = 0;
const waiting = new Map();
const events = [];
ws.addEventListener('message', ev => {
  const m = JSON.parse(typeof ev.data === 'string' ? ev.data : Buffer.from(ev.data).toString());
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); return; }
  if (m.method === 'Runtime.exceptionThrown') events.push('异常: ' + (m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text));
  if (m.method === 'Runtime.consoleAPICalled') events.push('控制台 ' + m.params.type + ': ' + m.params.args.map(a => a.value ?? a.description).join(' '));
  if (m.method === 'Log.entryAdded') events.push('日志 ' + m.params.entry.level + ': ' + m.params.entry.text);
});
await new Promise((res, rej) => { ws.addEventListener('open', res); ws.addEventListener('error', rej); });
await sleep(1200);

function send(method, params = {}) {
  return new Promise(res => { const i = ++id; waiting.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
}
async function ev(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) return { error: r.result.exceptionDetails.exception?.description };
  return { value: r.result?.result?.value };
}

await send('Page.enable');
await send('Runtime.enable');
await send('Log.enable');
await sleep(1500);

console.log('页面标题:', (await ev('document.title')).value);
console.log('切到声音克隆页:', JSON.stringify(await ev(`showTab('clone'); 'ok'`)));
await sleep(400);
console.log('麦克风权限探一下:', JSON.stringify(await ev(`navigator.mediaDevices.getUserMedia({audio:true}).then(s=>{s.getTracks().forEach(t=>t.stop());return 'ok'}).catch(e=>'失败:'+e.name)`)));

console.log('点录制:', JSON.stringify(await ev(`document.getElementById('recBtn').click(); 'clicked'`)));

const t0 = Date.now();
let last = null;
for (let i = 0; i < 70; i++) {
  await sleep(500);
  const s = await ev(`(()=>({msg: document.getElementById('cloneMsg').textContent,
            btn: document.getElementById('recBtn').textContent,
            reviewHidden: document.getElementById('recModal').classList.contains('hidden'),
            reviewText: (document.getElementById('rmTitle')||{}).textContent +
                        ' / ' + ((document.getElementById('rmMsg')||{}).textContent||'').slice(0,46),
            playing: (()=>{const a=document.getElementById('rmAudio'); return !!(a && !a.paused);})(),
            src: ((document.getElementById('rmAudio')||{}).src||'').split('/').pop()}))()`);
  if(s.error){ console.log('取值出错:', s.error); break; }
  const v = s.value || {};
  const line = `${((Date.now() - t0) / 1000).toFixed(1)}s | 按钮=${v.btn} | 提示=${v.msg} | 弹窗隐藏=${v.reviewHidden} | 在放=${v.playing} | ${v.src} | ${v.reviewText || ''}`;
  if (line !== last) { console.log(line); last = line; }
  if (v.reviewHidden === false) {
    console.log('>>> 弹窗出现了');
    await sleep(600);
    // 走「就用这段」：真的存进库，验完删掉
    await ev("document.getElementById('rmAccept').click(); 'accept'");
    await sleep(3500);
    const res = await ev(`(async()=>{
        const r = await fetch('/api/characters').then(r=>r.json());
        const list = Array.isArray(r) ? r : (r.characters || r.cards || []);
        return {msg: document.getElementById('cloneMsg').textContent,
                modalHidden: document.getElementById('recModal').classList.contains('hidden'),
                ids: list.map(c=>c.id), names: list.map(c=>c.name)};})()`);
    console.log('存库结果:', JSON.stringify(res.value || res.error));
    const ids = (res.value && res.value.ids) || [];
    for (const id of ids) {
      await fetch(`http://127.0.0.1:7864/api/characters/${id}/delete`, {method:'POST'});
      console.log('已删掉测试条目', id);
    }
    const after = await ev(`fetch('/api/characters').then(r=>r.json()).then(r=>JSON.stringify(Array.isArray(r)?r.length:(r.characters||[]).length))`);
    console.log('清库后角色数:', after.value);
    break;
  }
}

console.log('--- 控制台/异常 ---');
console.log(events.length ? events.join('\n') : '（无）');
ws.close();
chrome.kill('SIGKILL');
