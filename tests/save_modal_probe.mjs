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
console.log('调到 PPT 页:', JSON.stringify(await ev(`showTab('ppt'); 'ok'`)));
await sleep(500);
console.log('模拟任务跑完，弹保存框:', JSON.stringify(await ev(`pptDoneOpen('好了：61 页，229 句'); 'called'`)));
await sleep(700);
const st = await ev(`(()=>({shown: !document.getElementById('saveModal').classList.contains('hidden'),
        text: document.getElementById('saveMsg').innerText,
        buttons: [...document.querySelectorAll('#saveModal button')].map(b=>b.textContent)}))()`);
console.log('弹窗状态:', JSON.stringify(st.value || st.error));
await sleep(300);
const shot = await send('Page.captureScreenshot', {format:'png'});
const { writeFileSync } = await import('node:fs');
writeFileSync('/tmp/vc-savemodal.png', Buffer.from(shot.result.data,'base64'));
console.log('截图 /tmp/vc-savemodal.png');
const sup = await ev(`({picker: !!window.showSaveFilePicker})`);
console.log('浏览器支持直接写盘:', JSON.stringify(sup.value));

console.log('--- 控制台/异常 ---');
console.log(events.length ? events.join('\n') : '（无）');
ws.close();
chrome.kill('SIGKILL');
