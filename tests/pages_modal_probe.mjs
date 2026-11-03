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
console.log('切到 PPT 页:', JSON.stringify(await ev(`showTab('ppt'); 'ok'`)));
await sleep(400);
const set = await ev(`(async()=>{
  const d = await (await fetch('/api/ppt/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:'_两页测试'})})).json();
  PPT.name = d.name; PPT.pages = d.pages; PPT.existing = d.existing || {};
  renderPpt();
  pptDoneOpen('好了：2 页，4 句');
  return 'ok';})()`);
console.log('注入课件状态:', JSON.stringify(set.value || set.error));
await sleep(800);
const st = await ev(`(()=>({shown: !document.getElementById('saveModal').classList.contains('hidden'),
        msg: document.getElementById('saveMsg').innerText,
        rows: [...document.querySelectorAll('#pagesList .prow')].map(r=>r.innerText.replace(/\n/g,' | '))}))()`);
console.log('弹窗:', JSON.stringify(st.value || st.error, null, 1));
// 点第 1 行的「听」，看真的挂上音频没有
const play = await ev(`(async()=>{
  const btn=[...document.querySelectorAll('#pagesList .prow button')][0];
  btn.click();
  await new Promise(r=>setTimeout(r, 600));
  return {btn: btn.textContent, hasAudio: !!(ONE && ONE.el), src: (ONE && ONE.el && ONE.el.src || '').split('/').slice(-1)[0]};})()`);
console.log('点「听」之后:', JSON.stringify(play.value || play.error));
const shot = await send('Page.captureScreenshot', {format:'png'});
const { writeFileSync } = await import('node:fs');
writeFileSync('/tmp/vc-pages-modal.png', Buffer.from(shot.result.data,'base64'));
console.log('截图 /tmp/vc-pages-modal.png');

// 点第 2 页的首句：应该弹出整页讲稿
const ed = await ev(`(async()=>{
  const rows=[...document.querySelectorAll('#pagesList .prow')];
  rows[1].querySelector('.ptext').click();
  await new Promise(r=>setTimeout(r,400));
  return {shown: !document.getElementById('editPageModal').classList.contains('hidden'),
          title: document.getElementById('peTitle').textContent,
          text: document.getElementById('peText').value,
          btns: [...document.querySelectorAll('#editPageModal button')].map(b=>b.textContent)};})()`);
console.log('编辑弹窗:', JSON.stringify(ed.value || ed.error));
await sleep(300);
const shot2 = await send('Page.captureScreenshot', {format:'png'});
writeFileSync('/tmp/vc-page-editor.png', Buffer.from(shot2.result.data,'base64'));
console.log('截图 /tmp/vc-page-editor.png');

// 改一句、点确定，看它是按新的句子重做（会真的生成，约半分钟）
const go = await ev(`(async()=>{
  document.getElementById('peText').value = '这是第二页，我把这句改了。\n念到这儿，这组测试就结束了。';
  document.getElementById('editPageModal').querySelector('.go').click();
  await new Promise(r=>setTimeout(r, 3000));
  return {editorHidden: document.getElementById('editPageModal').classList.contains('hidden'),
          prog: document.getElementById('pptPmsg').textContent};})()`);
console.log('点确定之后:', JSON.stringify(go.value || go.error));
for(let k=0;k<20;k++){
  await sleep(3000);
  const t = await ev(`document.getElementById('pptPmsg').textContent`);
  console.log('  进度:', t.value);
  if(/好了|出错/.test(t.value||'')) break;
}
const m = await ev(`fetch('/api/ppt/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({name:'_两页测试'})}).then(r=>r.json()).then(d=>({
    sec1:(d.existing['1']||{}).seconds, sec2:(d.existing['2']||{}).seconds,
    text2:(d.pages.find(p=>p.index===2)||{}).sentences}))`);
console.log('重做后:', JSON.stringify(m.value || m.error));

console.log('--- 控制台/异常 ---');
console.log(events.length ? events.join('\n') : '（无）');
ws.close();
chrome.kill('SIGKILL');
