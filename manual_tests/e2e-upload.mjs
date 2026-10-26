import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const IMG = 'D:\\Stage\\Tunisian-Sign-Language-Translator\\test_images\\labes.jpg';
const VID = 'C:\\Users\\maher\\AppData\\Local\\Temp\\opencode\\e2e\\labes_video.mp4';

const results = [];
function check(name, pass, extra = '') {
  results.push({ name, pass: !!pass, extra });
  console.log(`${pass ? 'PASS' : 'FAIL'} | ${name}${extra ? ' | ' + extra : ''}`);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu', '--window-size=1280,900'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });

  // Delay every translation POST so the in-flight spinner state is observable
  let postCount = 0;
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    if (req.method() === 'POST' && req.url().includes('/translate/')) {
      postCount++;
      setTimeout(() => req.continue().catch(() => {}), 1200);
    } else {
      req.continue().catch(() => {});
    }
  });
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(String(e)));

  await page.goto('http://localhost:3000', {
    waitUntil: 'networkidle0',
    timeout: 30000,
  });

  // ── T1: spinner must be hidden on page load (CSS fix) ──
  const d1 = await page.$eval('#upload-progress', (el) => getComputedStyle(el).display);
  check('T1 spinner masque au chargement', d1 === 'none', `display=${d1}`);

  // ── T1b: user path — open the Upload tab right away, spinner must be absent ──
  await page.click('button[data-tab="upload"]').catch(async () => {
    await page.evaluate(() => {
      document.querySelector('#tab-upload').classList.add('active');
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === 'upload'));
    });
  });
  await sleep(300);
  const dUpload = await page.$eval('#upload-progress', (el) => getComputedStyle(el).display);
  const uploadVisible = await page.$eval('#tab-upload', (el) => getComputedStyle(el).display !== 'none');
  check('T1b onglet Upload ouvert: aucun Processing visible', uploadVisible && dUpload === 'none', `tab=${uploadVisible} display=${dUpload}`);

  await page.click('.tab[data-tab="upload"]');

  // ── T2-T4: translate via file picker ──
  const input = await page.$('#file-input');
  await input.uploadFile(IMG);
  await page.click('#btn-translate-upload');
  await sleep(500);
  const mid = await page.$eval('#upload-progress', (el) => getComputedStyle(el).display);
  check('T2 spinner visible pendant le traitement', mid === 'flex', `display=${mid}`);
  const inFlight = await page.evaluate(() => ({
    disabled: document.querySelector('#btn-translate-upload').disabled,
    label: document.querySelector('#upload-progress span').textContent,
  }));
  check(
    'T13 bouton desactive + compteur affiche pendant le traitement',
    inFlight.disabled && inFlight.label.includes('Processing'),
    JSON.stringify(inFlight)
  );
  try {
    await page.waitForFunction(
      () => document.querySelector('#output-raw').textContent.includes('labes'),
      { timeout: 20000 }
    );
    check('T3 image (selecteur) -> resultat "labes"', true);
  } catch {
    const out = await page.$eval('#output-raw', (el) => el.textContent);
    check('T3 image (selecteur) -> resultat "labes"', false, `out=${out}`);
  }
  const d4 = await page.$eval('#upload-progress', (el) => getComputedStyle(el).display);
  check('T4 spinner masque apres traitement', d4 === 'none', `display=${d4}`);
  const after = await page.evaluate(() => ({
    disabled: document.querySelector('#btn-translate-upload').disabled,
    label: document.querySelector('#upload-progress span').textContent,
  }));
  check(
    'T14 bouton reactive + compteur reinitialise apres traitement',
    !after.disabled && after.label === 'Processing...',
    JSON.stringify(after)
  );

  // ── T5-T6: drag & drop path (the reported bug) ──
  await page.evaluate(() => {
    document.querySelector('#output-raw').textContent = '—';
  });
  const b64 = fs.readFileSync(IMG).toString('base64');
  await page.evaluate((bytes) => {
    const bin = Uint8Array.from(atob(bytes), (c) => c.charCodeAt(0));
    const file = new File([bin], 'labes.jpg', { type: 'image/jpeg' });
    const dt = new DataTransfer();
    dt.items.add(file);
    document
      .querySelector('#drop-zone')
      .dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true }));
  }, b64);
  const hasSelected = await page.evaluate(
    () => typeof selectedFile !== 'undefined' && selectedFile !== null
  );
  check('T5 drag&drop enregistre le fichier (selectedFile)', hasSelected);
  await page.click('#btn-translate-upload');
  try {
    await page.waitForFunction(
      () => document.querySelector('#output-raw').textContent.includes('labes'),
      { timeout: 20000 }
    );
    check('T6 drag&drop -> traduction "labes"', true);
  } catch {
    const out = await page.$eval('#output-raw', (el) => el.textContent);
    check('T6 drag&drop -> traduction "labes"', false, `out=${out}`);
  }

  // ── T7-T8: clicking Translate with no file must explain, not send ──
  await page.evaluate(() => {
    selectedFile = null;
    document.querySelector('#file-input').value = '';
    document.querySelector('#output-raw').textContent = '—';
  });
  const before = postCount;
  await page.click('#btn-translate-upload');
  await sleep(400);
  const guide = await page.$eval('#output-raw', (el) => el.textContent);
  check('T7 clic sans fichier -> message daide', guide.startsWith('Aucun fichier'), `out=${guide}`);
  check('T8 aucune requete envoyee sans fichier', postCount === before, `posts=${postCount}`);

  // ── T9-T10: video upload path ──
  const input2 = await page.$('#file-input');
  await input2.uploadFile(VID);
  await page.click('#btn-translate-upload');
  await sleep(500);
  const midV = await page.$eval('#upload-progress', (el) => getComputedStyle(el).display);
  check('T9 spinner visible pendant la video', midV === 'flex', `display=${midV}`);
  try {
    await page.waitForFunction(
      () => getComputedStyle(document.querySelector('#upload-progress')).display === 'none',
      { timeout: 90000 }
    );
    const outV = await page.$eval('#output-raw', (el) => el.textContent);
    const btnBack = await page.$eval('#btn-translate-upload', (el) => !el.disabled);
    const hasLlmRow = await page.evaluate(() =>
      !!document.querySelector('#upload-result .badge-llm')
    );
    check(
      'T10 video: sortie affichee, bouton reactive, ligne LLM presente',
      outV !== '—' && !outV.startsWith('Erreur') && !outV.startsWith('Échec') && btnBack && hasLlmRow,
      `out=${outV.slice(0, 80)} llmRow=${hasLlmRow}`
    );
  } catch {
    check('T10 video: sortie affichee, bouton reactive, ligne LLM presente', false, 'timeout');
  }

  // ── T11-T12: error path (bad file) must surface an error, not stick ──
  await page.evaluate(() => {
    document.querySelector('#output-raw').textContent = '—';
  });
  const badInput = await page.$('#file-input');
  await badInput.uploadFile('D:\\Stage\\Tunisian-Sign-Language-Translator\\README.md');
  await page.click('#btn-translate-upload');
  try {
    await page.waitForFunction(() => {
      const raw = document.querySelector('#output-raw').textContent;
      return raw.startsWith('Erreur') || raw.startsWith('Échec');
    }, { timeout: 20000 });
    const outB = await page.$eval('#output-raw', (el) => el.textContent);
    const dB = await page.$eval('#upload-progress', (el) => getComputedStyle(el).display);
    check('T12 fichier invalide -> erreur affichee + spinner masque', dB === 'none', `out=${outB}`);
  } catch {
    const outB = await page.$eval('#output-raw', (el) => el.textContent);
    check('T12 fichier invalide -> erreur affichee + spinner masque', false, `out=${outB}`);
  }

  // ── T11: no uncaught JS errors ──
  check('T11 aucune erreur JS non interceptee', pageErrors.length === 0, pageErrors.join(' | '));
} finally {
  await browser.close();
}

const fails = results.filter((r) => !r.pass);
console.log(`\nRESULTAT: ${results.length - fails.length}/${results.length} tests OK`);
process.exit(fails.length ? 1 : 0);
