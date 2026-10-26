// ── TSL Translator Frontend ──

const API_BASE = `${window.location.protocol}//${window.location.hostname || 'localhost'}:8000`;

const WS_URL = API_BASE.replace('http', 'ws') + '/ws/camera';

// ── State ──
let ws = null;
let stream = null;
let cameraActive = false;
let sendInterval = null;
let currentMode = 'sign';
let selectedFile = null; // set by file picker OR drag & drop
let lastCorrectedText = '';
let lastCorrectAt = 0;
const CORRECT_MIN_INTERVAL_MS = 3000; // throttle LLM calls (cost + latency)

// ── DOM ──
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Tabs ──
$$('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        $$('.tab').forEach(t => t.classList.remove('active'));
        $$('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        $(`#tab-${tab.dataset.tab}`).classList.add('active');
    });
});

// ── WebSocket ──
function connectWS() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        $('#ws-status').textContent = 'Connected';
        $('#ws-status').className = 'status connected';
        $('#btn-toggle-mode').disabled = false;
    };

    ws.onclose = () => {
        $('#ws-status').textContent = 'Disconnected';
        $('#ws-status').className = 'status disconnected';
        $('#btn-toggle-mode').disabled = true;
        setTimeout(connectWS, 3000);
    };

    ws.onerror = () => {
        ws.close();
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'translation':
            updateTranslation(msg.data);
            break;
        case 'mode_changed':
            currentMode = msg.data.mode;
            $('#mode-status').textContent = `Mode: ${msg.data.mode.toUpperCase()}`;
            break;
        case 'spell_flushed':
            if (msg.data.sentence) {
                addToHistory(msg.data.sentence);
            }
            break;
        case 'spell':
            updateSpell(msg.data);
            break;
        case 'corrected':
            $('#output-corrected').textContent = msg.data.corrected || '—';
            break;
        case 'history_cleared':
            $('#history-list').innerHTML = '<p class="empty">No translations yet</p>';
            break;
        case 'error':
            console.error('Server error:', msg.message);
            break;
    }
}

function updateTranslation(data) {
    const raw = data.tsl || data.tunisian || data.french || data.english || data.arabic || '';
    const conf = (data.confidence || 0).toFixed(2);

    if (data.rejected || !raw) {
        $('#output-raw').textContent = '—';
        $('#output-confidence').textContent = '—';
        return;
    }

    $('#output-raw').textContent = raw;
    $('#output-confidence').textContent = conf;

    // Auto-correct via LLM if configured (throttled: one call per few
    // seconds and only when the text actually changed)
    const now = Date.now();
    if (
        ws && ws.readyState === WebSocket.OPEN
        && raw !== lastCorrectedText
        && now - lastCorrectAt >= CORRECT_MIN_INTERVAL_MS
    ) {
        lastCorrectedText = raw;
        lastCorrectAt = now;
        ws.send(JSON.stringify({ action: 'correct', text: raw }));
    }

    // Add to history
    addToHistory(raw, conf);
}

function updateSpell(data) {
    currentMode = 'spell';
    $('#mode-status').textContent = 'Mode: SPELL';
    const text = data.sentence || data.current_word || '';
    const letter = data.arabic || data.letter || '';
    if (text) {
        $('#output-raw').textContent = text;
    } else if (letter && !data.rejected) {
        $('#output-raw').textContent = letter;
    }
    if (data.confidence != null && !data.rejected) {
        $('#output-confidence').textContent = (data.confidence || 0).toFixed(2);
    }
    if (data.event === 'word_done' && data.current_word) {
        addToHistory(data.current_word);
    }
}

function addToHistory(text, confidence = '') {
    if (!text || text === '—') return;
    const list = $('#history-list');
    const empty = list.querySelector('.empty');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'history-item';
    const time = new Date().toLocaleTimeString();
    item.innerHTML = `
        <span>${text}</span>
        <span class="time">${confidence ? confidence + ' ' : ''}${time}</span>
    `;
    list.insertBefore(item, list.firstChild);

    // Keep max 50 items
    while (list.children.length > 50) {
        list.removeChild(list.lastChild);
    }
}

// ── Camera ──
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        const video = $('#video');
        video.srcObject = stream;
        video.hidden = false;
        $('#camera-placeholder').hidden = true;

        connectWS();
        cameraActive = true;

        $('#btn-start-camera').disabled = true;
        $('#btn-stop-camera').disabled = false;
        $('#btn-flush').disabled = false;

        // Send frames at ~10 FPS
        sendInterval = setInterval(sendFrame, 100);
    } catch (err) {
        console.error('Camera error:', err);
        alert('Could not access camera: ' + err.message);
    }
}

function stopCamera() {
    if (sendInterval) {
        clearInterval(sendInterval);
        sendInterval = null;
    }
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    $('#video').srcObject = null;
    $('#video').hidden = true;
    $('#camera-placeholder').hidden = false;
    cameraActive = false;

    $('#btn-start-camera').disabled = false;
    $('#btn-stop-camera').disabled = true;
    $('#btn-flush').disabled = true;
}

function sendFrame() {
    if (!cameraActive || !ws || ws.readyState !== WebSocket.OPEN) return;

    const video = $('#video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
    const base64 = dataUrl.split(',')[1];

    ws.send(JSON.stringify({ frame: base64 }));
}

// ── File Upload ──
$('#btn-select-file').addEventListener('click', () => $('#file-input').click());
$('#drop-zone').addEventListener('click', (e) => {
    if (e.target === e.currentTarget || e.target.tagName === 'P') {
        $('#file-input').click();
    }
});

$('#drop-zone').addEventListener('dragover', (e) => {
    e.preventDefault();
    e.currentTarget.style.borderColor = 'var(--primary)';
});
$('#drop-zone').addEventListener('dragleave', (e) => {
    e.currentTarget.style.borderColor = '';
});
$('#drop-zone').addEventListener('drop', (e) => {
    e.preventDefault();
    e.currentTarget.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

$('#file-input').addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

function isVideoFile(file) {
    if (!file) return false;
    if (file.type && file.type.startsWith('video/')) return true;
    const ext = (file.name || '').split('.').pop().toLowerCase();
    return ['mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv', 'flv', 'm4v'].includes(ext);
}

function isImageFile(file) {
    if (!file) return false;
    if (file.type && file.type.startsWith('image/')) return true;
    const ext = (file.name || '').split('.').pop().toLowerCase();
    return ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif', 'svg'].includes(ext);
}

function handleFile(file) {
    const preview = $('#upload-preview');
    const img = $('#preview-image');
    const vid = $('#preview-video');
    const resultBox = $('#upload-result');

    selectedFile = file;

    // Clear previous results and progress
    if (resultBox) {
        resultBox.hidden = true;
        resultBox.style.display = 'none';
        resultBox.innerHTML = '';
    }
    const progress = $('#upload-progress');
    if (progress) {
        progress.hidden = true;
        progress.style.display = 'none';
    }

    const isVid = isVideoFile(file);
    const isImg = isImageFile(file);

    if (isImg) {
        img.src = URL.createObjectURL(file);
        img.hidden = false;
        img.style.display = 'block';
        vid.hidden = true;
        vid.style.display = 'none';
        try { vid.pause(); } catch (_) {}
    } else if (isVid) {
        vid.src = URL.createObjectURL(file);
        vid.hidden = false;
        vid.style.display = 'block';
        img.hidden = true;
        img.style.display = 'none';
    } else {
        // Fallback: try image preview
        img.src = URL.createObjectURL(file);
        img.hidden = false;
        img.style.display = 'block';
        vid.hidden = true;
        vid.style.display = 'none';
    }

    preview.hidden = false;
    preview.style.display = 'block';
    preview.dataset.file = file.name;
    preview.dataset.type = file.type;

    const btn = $('#btn-translate-upload');
    if (btn) btn.disabled = false;
}

$('#btn-translate-upload').addEventListener('click', async () => {
    const fileInput = $('#file-input');
    const file = selectedFile || (fileInput.files && fileInput.files[0]);
    if (!file) {
        $('#output-raw').textContent = 'Aucun fichier — glisse-dépose une image/vidéo ou clique sur « Select File »';
        $('#output-corrected').textContent = '—';
        $('#output-confidence').textContent = '—';
        return;
    }

    const progress = $('#upload-progress');
    const label = $('#upload-progress span');
    const btn = $('#btn-translate-upload');
    const resultBox = $('#upload-result');
    if (resultBox) {
        resultBox.hidden = true;
        resultBox.style.display = 'none';
    }

    progress.hidden = false;
    progress.style.display = 'flex';
    btn.disabled = true;

    // Elapsed counter: long videos keep visible feedback (never look frozen)
    const startedAt = Date.now();
    const tick = setInterval(() => {
        const s = Math.round((Date.now() - startedAt) / 1000);
        label.textContent = s > 0 ? `Processing... ${s}s` : 'Processing...';
    }, 250);

    try {
        const isVideo = isVideoFile(file);
        const sampleRate = ($('#sample-rate') && $('#sample-rate').value) || '4';
        const targetLang = ($('#target-lang') && $('#target-lang').value) || 'en';
        const url = `${API_BASE}${isVideo
            ? `/translate/video?sample_rate=${encodeURIComponent(sampleRate)}&llm=1&target_lang=${encodeURIComponent(targetLang)}`
            : '/translate/frame'}`;
        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch(url, { method: 'POST', body: formData });
        let result = {};
        try { result = await resp.json(); } catch (_) { /* non-JSON body */ }

        if (!resp.ok) {
            let errorMsg = result.detail || `HTTP ${resp.status}`;
            if (resp.status === 422 && String(result.detail).includes('No hands detected')) {
                errorMsg = 'Aucune main détectée. Assurez-vous que les mains et les gestes sont bien visibles.';
            } else if (resp.status === 413) {
                errorMsg = 'Fichier trop volumineux (max 50 Mo pour vidéo, 10 Mo pour image).';
            }
            $('#output-raw').textContent = 'Erreur : ' + errorMsg;
            $('#output-corrected').textContent = '—';
            $('#output-confidence').textContent = '—';
            if (resultBox) {
                resultBox.hidden = false;
                resultBox.style.display = 'block';
                resultBox.innerHTML = `<div class="alert-error">⚠️ ${errorMsg}</div>`;
            }
            return;
        }

        if (result.sentences) {
            // Video result
            if (result.sentences.length > 0) {
                const rawTexts = result.sentences.map(s => s.sentence).filter(Boolean).join(' ');
                const coordTexts = result.sentences.map(s => s.coordinated).filter(Boolean).join(' ');
                const langLabel = LANG_LABELS[result.target_lang] || result.target_lang || 'English';
                $('#output-raw').textContent = rawTexts || '—';
                $('#output-corrected').textContent = coordTexts || '—';
                $('#output-confidence').textContent = '1.00';
                if (resultBox) {
                    resultBox.hidden = false;
                    resultBox.style.display = 'block';
                    let html = `<div class="upload-result-card">
                        <h3>Phrases reconnues (${result.sentences.length})</h3>`;
                    result.sentences.forEach((s, idx) => {
                        const signsStr = (s.signs || []).join(', ') || '—';
                        const fr = s.translations && s.translations.fr ? s.translations.fr : '';
                        const ar = s.translations && s.translations.ar ? s.translations.ar : '';
                        const coordRow = s.coordinated
                            ? `<div class="result-row"><span class="badge badge-llm">LLM</span> <strong>${s.coordinated}</strong></div>`
                            : `<div class="result-row"><span class="badge badge-en">${langLabel}</span> ${s.sentence}</div>`;
                        html += `
                            <div class="sentence-item">
                                <div class="sentence-num">#${idx + 1}</div>
                                <div class="sentence-details">
                                    <div class="result-row"><span class="badge">Signes</span> <strong>${signsStr}</strong></div>
                                    ${coordRow}
                                    ${fr ? `<div class="result-row"><span class="badge badge-fr">Français</span> ${fr}</div>` : ''}
                                    ${ar ? `<div class="result-row"><span class="badge badge-ar">العربية</span> <span dir="rtl">${ar}</span></div>` : ''}
                                </div>
                            </div>
                        `;
                    });
                    html += `</div>`;
                    resultBox.innerHTML = html;
                }
                addToHistory(coordTexts || rawTexts);
            } else {
                $('#output-raw').textContent = 'Aucun signe détecté dans la vidéo';
                $('#output-corrected').textContent = '—';
                $('#output-confidence').textContent = '—';
                if (resultBox) {
                    resultBox.hidden = false;
                    resultBox.style.display = 'block';
                    resultBox.innerHTML = `<div class="alert-warning">⚠️ Aucun geste de langue des signes n'a pu être identifié dans la vidéo. Essayez de faire des signes plus nets et visibles face à la caméra.</div>`;
                }
            }
        } else {
            // Single image frame result
            const raw = result.tsl || result.tunisian || result.french || result.english || result.arabic || '';
            const conf = (result.confidence || 0).toFixed(2);
            $('#output-raw').textContent = raw || '—';
            $('#output-corrected').textContent = '—';
            $('#output-confidence').textContent = conf;

            if (result.rejected || !raw) {
                if (resultBox) {
                    resultBox.hidden = false;
                    resultBox.style.display = 'block';
                    resultBox.innerHTML = `<div class="alert-warning">⚠️ Geste non reconnu avec une confiance suffisante (${conf}). Essayez avec un angle plus net.</div>`;
                }
            } else {
                if (resultBox) {
                    resultBox.hidden = false;
                    resultBox.style.display = 'block';
                    resultBox.innerHTML = `
                        <div class="upload-result-card">
                            <h3>Résultat de la traduction</h3>
                            <div class="result-grid">
                                <div class="result-item"><span class="label">Signe TSL</span><span class="val highlight">${result.tsl || '—'}</span></div>
                                <div class="result-item"><span class="label">Dialecte tunisien</span><span class="val">${result.tunisian || '—'}</span></div>
                                <div class="result-item"><span class="label">Français</span><span class="val">${result.french || '—'}</span></div>
                                <div class="result-item"><span class="label">Anglais</span><span class="val">${result.english || '—'}</span></div>
                                <div class="result-item"><span class="label">Arabe</span><span class="val" dir="rtl">${result.arabic || result.arabic_msa || '—'}</span></div>
                                <div class="result-item"><span class="label">Confiance</span><span class="val badge-conf">${Math.round((result.confidence || 0) * 100)}%</span></div>
                            </div>
                        </div>
                    `;
                }
                addToHistory(raw, conf);
            }
        }
    } catch (err) {
        const netErr = err instanceof TypeError || /fetch|network/i.test(err.message || '');
        const msg = netErr
            ? 'Serveur API injoignable (localhost:8000) — vérifie que « python start.py » tourne, puis réessaie.'
            : 'Échec de la traduction : ' + err.message;
        $('#output-raw').textContent = msg;
        $('#output-corrected').textContent = '—';
        $('#output-confidence').textContent = '—';
        if (resultBox) {
            resultBox.hidden = false;
            resultBox.style.display = 'block';
            resultBox.innerHTML = `<div class="alert-error">⚠ ${msg}</div>`;
        }
    } finally {
        clearInterval(tick);
        label.textContent = 'Processing...';
        btn.disabled = false;
        progress.hidden = true;
        progress.style.display = 'none';
    }
});

// ── Settings ──
// Built-in defaults per provider (mirrors LLMConfig.PROVIDER_DEFAULTS).
// A model that matches another provider's default is a leftover from the
// previous selection and must not be sent to the newly selected provider.
const LLM_PROVIDER_DEFAULTS = {
    openrouter: 'openrouter/free',
    openai: 'gpt-4o-mini',
    anthropic: 'claude-3-haiku-20240307',
    ollama: 'llama3.2:3b',
    custom: '',
};

const LANG_LABELS = { en: 'English', fr: 'Français', ar: 'العربية', tunisian: 'Tunisien' };

function setLlmStatus(text, isError = false) {
    const el = $('#llm-status');
    el.textContent = text;
    el.classList.toggle('error', !!isError);
}

function renderActiveLlm(config) {
    const el = $('#llm-active');
    if (!el || !config || !config.provider) return;
    const local = config.is_local ? ' · local' : '';
    const state = config.enabled
        ? 'actif'
        : 'désactivé (clé API manquante)';
    el.innerHTML = `Actif : <b>${config.provider} / ${config.model}</b> `
        + `(${state}${local})`;
}

function syncModelPlaceholder(provider) {
    $('#llm-model').placeholder = LLM_PROVIDER_DEFAULTS[provider]
        || 'nom-du-modèle';
}

$('#llm-provider').addEventListener('change', (e) => {
    const provider = e.target.value;
    $('#custom-url-group').hidden = provider !== 'custom';
    syncModelPlaceholder(provider);
    // Drop a default inherited from the previously selected provider so it
    // can never be sent to the new one (the server re-applies its default).
    const modelInput = $('#llm-model');
    const defaults = Object.values(LLM_PROVIDER_DEFAULTS).filter(Boolean);
    if (defaults.includes(modelInput.value.trim())) {
        modelInput.value = '';
    }
    // The masked key belongs to the previous provider (keys are per provider)
    const keyInput = $('#llm-api-key');
    if (keyInput.value === '••••••••') keyInput.value = '';
    keyInput.placeholder = provider === 'ollama'
        ? 'non requis (modèle local)'
        : 'sk-...';
});

$('#confidence-threshold').addEventListener('input', (e) => {
    $('#confidence-value').textContent = e.target.value;
});

$('#btn-save-llm').addEventListener('click', async () => {
    const keyField = $('#llm-api-key').value;
    const masked = keyField === '••••••••';
    const config = {
        provider: $('#llm-provider').value,
        // null = "keep the stored key"; only a freshly typed value replaces it.
        api_key: masked ? null : keyField,
        model: $('#llm-model').value.trim(),
        base_url: $('#llm-base-url').value.trim(),
    };
    try {
        const resp = await fetch(`${API_BASE}/llm/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const result = await resp.json();
        if (!resp.ok) {
            setLlmStatus('Error: ' + (result.detail || resp.status), true);
            return;
        }
        setLlmStatus('Saved!');
        // Reflect exactly what the server stored — no optimistic UI.
        if (result.config) {
            renderActiveLlm(result.config);
            $('#llm-provider').value = result.config.provider;
            $('#llm-model').value = result.config.model;
            syncModelPlaceholder(result.config.provider);
            if (result.config.has_api_key) $('#llm-api-key').value = '••••••••';
            $('#custom-url-group').hidden = result.config.provider !== 'custom';
            if (result.config.provider === 'custom') {
                $('#llm-base-url').value = result.config.base_url;
            }
        }
        setTimeout(() => { setLlmStatus(''); }, 2000);
    } catch (err) {
        setLlmStatus('Error: ' + err.message, true);
    }
});

$('#btn-save-settings').addEventListener('click', async () => {
    const confidence = parseFloat($('#confidence-threshold').value);
    const smoothing_window = parseInt($('#smoothing-window').value, 10);
    localStorage.setItem('tsl_confidence', confidence);
    localStorage.setItem('tsl_smoothing', smoothing_window);
    try {
        const resp = await fetch(`${API_BASE}/settings/inference`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confidence_threshold: confidence, smoothing_window }),
        });
        const result = await resp.json();
        if (!resp.ok) {
            $('#settings-status').textContent = 'Error: ' + (result.detail || resp.status);
            return;
        }
        $('#settings-status').textContent = 'Saved!';
    } catch (err) {
        $('#settings-status').textContent = 'Error: ' + err.message;
    }
    setTimeout(() => { $('#settings-status').textContent = ''; }, 2000);
});

// ── Mode Toggle ──
$('#btn-toggle-mode').addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'toggle_mode' }));
    }
});

$('#btn-flush').addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'flush_spell' }));
    }
});

$('#btn-clear-history').addEventListener('click', () => {
    fetch(`${API_BASE}/history`, { method: 'DELETE' })
        .then(() => {
            $('#history-list').innerHTML = '<p class="empty">No translations yet</p>';
        });
});

// ── Camera Buttons ──
$('#btn-start-camera').addEventListener('click', startCamera);
$('#btn-stop-camera').addEventListener('click', stopCamera);

// ── Load saved settings ──
function loadSettings() {
    const conf = localStorage.getItem('tsl_confidence');
    const smooth = localStorage.getItem('tsl_smoothing');
    if (conf) {
        $('#confidence-threshold').value = conf;
        $('#confidence-value').textContent = conf;
    }
    if (smooth) {
        $('#smoothing-window').value = smooth;
    }

    // Load server-side inference settings (fall back to localStorage)
    fetch(`${API_BASE}/settings/inference`)
        .then(r => r.json())
        .then(s => {
            if (typeof s.confidence_threshold === 'number') {
                $('#confidence-threshold').value = s.confidence_threshold;
                $('#confidence-value').textContent = s.confidence_threshold;
            }
            if (typeof s.smoothing_window === 'number') {
                $('#smoothing-window').value = s.smoothing_window;
            }
        })
        .catch(() => {});

    // Load LLM config
    fetch(`${API_BASE}/llm/config`)
        .then(r => r.json())
        .then(config => {
            if (config.provider) {
                $('#llm-provider').value = config.provider;
                syncModelPlaceholder(config.provider);
            }
            if (config.model) $('#llm-model').value = config.model;
            if (config.base_url) $('#llm-base-url').value = config.base_url;
            $('#custom-url-group').hidden = config.provider !== 'custom';
            if (config.has_api_key) $('#llm-api-key').value = '••••••••';
            renderActiveLlm(config);
        })
        .catch(() => {});
}

// ── Init ──
loadSettings();
