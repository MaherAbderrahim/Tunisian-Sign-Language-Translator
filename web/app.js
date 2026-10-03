// ── TSL Translator Frontend ──

// API is always on port 8000; frontend may be served from a different port
const API_BASE = 'http://localhost:8000';

const WS_URL = API_BASE.replace('http', 'ws') + '/ws/camera';

// ── State ──
let ws = null;
let stream = null;
let cameraActive = false;
let sendInterval = null;
let currentMode = 'sign';
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

function handleFile(file) {
    const preview = $('#upload-preview');
    const img = $('#preview-image');
    const vid = $('#preview-video');

    if (file.type.startsWith('image/')) {
        img.src = URL.createObjectURL(file);
        img.hidden = false;
        vid.hidden = true;
    } else if (file.type.startsWith('video/')) {
        vid.src = URL.createObjectURL(file);
        vid.hidden = false;
        img.hidden = true;
    }
    preview.hidden = false;
    preview.dataset.file = file.name;
    preview.dataset.type = file.type;
}

$('#btn-translate-upload').addEventListener('click', async () => {
    const preview = $('#upload-preview');
    const fileInput = $('#file-input');
    if (!fileInput.files[0]) return;

    const progress = $('#upload-progress');
    progress.hidden = false;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        let result;
        if (fileInput.files[0].type.startsWith('video/')) {
            const resp = await fetch(`${API_BASE}/translate/video`, {
                method: 'POST',
                body: formData,
            });
            result = await resp.json();
        } else {
            const resp = await fetch(`${API_BASE}/translate/frame`, {
                method: 'POST',
                body: formData,
            });
            result = await resp.json();
        }

        if (result.sentences) {
            const texts = result.sentences.map(s => s.sentence).join(' ');
            $('#output-raw').textContent = texts || '—';
        } else {
            const raw = result.tsl || result.tunisian || result.french || result.english || '';
            $('#output-raw').textContent = raw || '—';
            $('#output-confidence').textContent = (result.confidence || 0).toFixed(2);
        }
    } catch (err) {
        alert('Translation failed: ' + err.message);
    } finally {
        progress.hidden = true;
    }
});

// ── Settings ──
$('#llm-provider').addEventListener('change', (e) => {
    $('#custom-url-group').hidden = e.target.value !== 'custom';
});

$('#confidence-threshold').addEventListener('input', (e) => {
    $('#confidence-value').textContent = e.target.value;
});

$('#btn-save-llm').addEventListener('click', async () => {
    const config = {
        provider: $('#llm-provider').value,
        api_key: $('#llm-api-key').value === '••••••••' ? '' : $('#llm-api-key').value,
        model: $('#llm-model').value,
        base_url: $('#llm-base-url').value,
    };
    try {
        const resp = await fetch(`${API_BASE}/llm/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const result = await resp.json();
        if (!resp.ok) {
            $('#llm-status').textContent = 'Error: ' + (result.detail || resp.status);
        } else {
            $('#llm-status').textContent = 'Saved!';
            setTimeout(() => { $('#llm-status').textContent = ''; }, 2000);
        }
    } catch (err) {
        $('#llm-status').textContent = 'Error: ' + err.message;
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
            if (config.provider) $('#llm-provider').value = config.provider;
            if (config.model) $('#llm-model').value = config.model;
            if (config.has_api_key) $('#llm-api-key').value = '••••••••';
        })
        .catch(() => {});
}

// ── Init ──
loadSettings();
