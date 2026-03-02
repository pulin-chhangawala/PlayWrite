/* playwrite: neural network playground */

// -------------------------------------------------------------------------
//  STATE
// -------------------------------------------------------------------------
let state = {
    dataset: 'mnist',
    layers: [128, 64],
    datasets: {},
    charts: { accuracy: null, loss: null },
    eventSource: null,
    totalEpochs: 20,
    startTime: null
};

// -------------------------------------------------------------------------
//  INIT
// -------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    loadDatasets();
    renderLayers();
    bindRangeInputs();
});

function bindRangeInputs() {
    document.querySelectorAll('input[type="range"]').forEach(el => {
        const display = document.getElementById(el.id + '_val');
        if (display) {
            el.addEventListener('input', () => {
                display.textContent = el.value;
            });
        }
    });
}

// -------------------------------------------------------------------------
//  DATASETS
// -------------------------------------------------------------------------
async function loadDatasets() {
    try {
        const res = await fetch('/api/datasets');
        state.datasets = await res.json();
        renderDatasetOptions();
    } catch (e) {
        console.error('Failed to load datasets:', e);
    }
}

function renderDatasetOptions() {
    const grid = document.getElementById('dataset-grid');
    if (!grid) return;

    grid.innerHTML = '';
    const builtIn = ['mnist', 'fashion_mnist', 'cifar10'];

    builtIn.forEach(key => {
        const ds = state.datasets[key];
        if (!ds) return;
        const div = document.createElement('div');
        div.className = `dataset-option ${key === state.dataset ? 'active' : ''}`;
        div.onclick = () => selectDataset(key);
        div.innerHTML = `
            <div class="ds-name">${ds.name}</div>
            <div class="ds-info">${ds.train_size.toLocaleString()} train / ${ds.num_classes} classes</div>
        `;
        grid.appendChild(div);
    });

    // custom upload option
    if (state.datasets['custom']) {
        const ds = state.datasets['custom'];
        const div = document.createElement('div');
        div.className = `dataset-option ${state.dataset === 'custom' ? 'active' : ''}`;
        div.onclick = () => selectDataset('custom');
        div.innerHTML = `
            <div class="ds-name">${ds.name}</div>
            <div class="ds-info">${ds.train_size.toLocaleString()} samples</div>
        `;
        grid.appendChild(div);
    }
}

function selectDataset(key) {
    state.dataset = key;
    document.querySelectorAll('.dataset-option').forEach(el => el.classList.remove('active'));
    event.currentTarget.classList.add('active');
    loadSamples();
}

async function loadSamples() {
    try {
        const res = await fetch('/api/dataset/samples', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset: state.dataset })
        });
        const data = await res.json();
        renderSamplePreview(data);
    } catch (e) {
        console.error('Failed to load samples:', e);
    }
}

function renderSamplePreview(data) {
    const container = document.getElementById('sample-preview');
    if (!container) return;
    container.innerHTML = '';

    const previewGrid = document.createElement('div');
    previewGrid.style.cssText = 'display:grid;grid-template-columns:repeat(8,1fr);gap:4px;';

    data.samples.slice(0, 8).forEach(s => {
        const canvas = document.createElement('canvas');
        const size = data.shape[0];
        canvas.width = size;
        canvas.height = data.shape.length > 1 ? data.shape[1] : 1;
        canvas.style.cssText = 'width:100%;image-rendering:pixelated;border-radius:4px;';
        drawImage(canvas, s.pixels, data.shape, data.channels);
        previewGrid.appendChild(canvas);
    });

    container.appendChild(previewGrid);
}

// -------------------------------------------------------------------------
//  FILE UPLOAD
// -------------------------------------------------------------------------
function handleFileUpload(input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const label = document.getElementById('upload-label');
    if (label) label.textContent = 'Uploading...';

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            if (label) label.textContent = 'Error: ' + data.error;
            return;
        }
        state.dataset = 'custom';
        state.datasets['custom'] = {
            name: data.name,
            train_size: data.samples,
            num_classes: data.classes,
            labels: data.labels
        };
        renderDatasetOptions();
        if (label) label.textContent = `Uploaded: ${data.name} (${data.samples} samples, ${data.features} features, ${data.classes} classes)`;
    })
    .catch(e => {
        if (label) label.textContent = 'Upload failed';
    });
}

// -------------------------------------------------------------------------
//  LAYER BUILDER
// -------------------------------------------------------------------------
function renderLayers() {
    const list = document.getElementById('layer-list');
    if (!list) return;
    list.innerHTML = '';

    state.layers.forEach((neurons, i) => {
        const item = document.createElement('div');
        item.className = 'layer-item';
        item.innerHTML = `
            <span class="layer-label">Dense ${i + 1}</span>
            <input type="number" value="${neurons}" min="8" max="1024" step="8"
                   onchange="updateLayer(${i}, this.value)">
            <button class="remove-btn" onclick="removeLayer(${i})" title="Remove layer">&times;</button>
        `;
        list.appendChild(item);
    });
}

function addLayer() {
    state.layers.push(64);
    renderLayers();
}

function removeLayer(index) {
    if (state.layers.length <= 1) return;
    state.layers.splice(index, 1);
    renderLayers();
}

function updateLayer(index, value) {
    state.layers[index] = parseInt(value) || 64;
}

// -------------------------------------------------------------------------
//  TRAINING
// -------------------------------------------------------------------------
function startTraining() {
    const config = {
        dataset: state.dataset,
        learning_rate: parseFloat(document.getElementById('learning_rate').value),
        activation: document.getElementById('activation').value,
        optimizer: document.getElementById('optimizer').value,
        regularization: document.getElementById('regularization').value,
        reg_rate: parseFloat(document.getElementById('reg_rate').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
        epochs: parseInt(document.getElementById('epochs').value),
        train_split: parseFloat(document.getElementById('train_split').value),
        dropout: parseFloat(document.getElementById('dropout').value),
        layers: state.layers,
        use_cnn: document.getElementById('use_cnn')?.checked || false
    };

    state.totalEpochs = config.epochs;
    state.startTime = Date.now();

    // reset UI
    resetCharts();
    setTrainingUI(true);

    // close older connection
    if (state.eventSource) {
        state.eventSource.close();
    }

    const eventSource = new EventSource('/api/train?' + new URLSearchParams({
        _body: 'sse'
    }));

    // SSE won't send POST body, so we use a workaround: POST first, then SSE
    // Actually, let's use fetch with streaming
    fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    setTrainingUI(false);
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            handleTrainingEvent(data);
                        } catch (e) {
                            // ignore parse errors
                        }
                    }
                });

                read();
            });
        }

        read();
    }).catch(err => {
        console.error('Training error:', err);
        setTrainingUI(false);
    });
}

function handleTrainingEvent(data) {
    switch (data.type) {
        case 'model_info':
            showModelInfo(data);
            break;
        case 'epoch':
            updateEpochProgress(data);
            updateCharts(data);
            break;
        case 'complete':
            showResults(data);
            setTrainingUI(false);
            break;
        case 'stopped':
            setTrainingUI(false);
            break;
        case 'error':
            showError(data.message);
            setTrainingUI(false);
            break;
    }
}

function stopTraining() {
    fetch('/api/stop', { method: 'POST' });
}

// -------------------------------------------------------------------------
//  UI UPDATES
// -------------------------------------------------------------------------
function setTrainingUI(isTraining) {
    const trainBtn = document.getElementById('train-btn');
    const stopBtn = document.getElementById('stop-btn');
    const progress = document.getElementById('progress-container');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    if (trainBtn) trainBtn.disabled = isTraining;
    if (stopBtn) stopBtn.style.display = isTraining ? 'flex' : 'none';
    if (progress) progress.classList.toggle('visible', isTraining);
    if (statusDot) statusDot.classList.toggle('active', isTraining);
    if (statusText) statusText.textContent = isTraining ? 'Training...' : 'Ready';
}

function showModelInfo(data) {
    const el = document.getElementById('model-summary');
    if (el) {
        el.textContent = data.summary.join('\n');
        el.parentElement.style.display = 'block';
    }
    const params = document.getElementById('param-count');
    if (params) params.textContent = data.total_params.toLocaleString();
}

function updateEpochProgress(data) {
    const fill = document.getElementById('progress-fill');
    const label = document.getElementById('epoch-label');
    const timeLabel = document.getElementById('time-label');
    const accVal = document.getElementById('stat-accuracy');
    const valAccVal = document.getElementById('stat-val-accuracy');
    const lossVal = document.getElementById('stat-loss');
    const bestVal = document.getElementById('stat-best');

    const pct = (data.epoch / state.totalEpochs) * 100;
    if (fill) fill.style.width = pct + '%';
    if (label) label.textContent = `Epoch ${data.epoch} / ${state.totalEpochs}`;

    const elapsed = ((Date.now() - state.startTime) / 1000).toFixed(0);
    if (timeLabel) timeLabel.textContent = `${elapsed}s elapsed`;

    if (accVal) accVal.textContent = (data.accuracy * 100).toFixed(2) + '%';
    if (valAccVal) valAccVal.textContent = (data.val_accuracy * 100).toFixed(2) + '%';
    if (lossVal) lossVal.textContent = data.loss.toFixed(4);
    if (bestVal) bestVal.textContent = (data.best_val_accuracy * 100).toFixed(2) + '%';
}

function showError(msg) {
    const progress = document.getElementById('progress-container');
    if (progress) {
        progress.innerHTML = `<div style="color:var(--danger);padding:16px;">Error: ${msg}</div>`;
        progress.classList.add('visible');
    }
}

// -------------------------------------------------------------------------
//  CHARTS
// -------------------------------------------------------------------------
function resetCharts() {
    if (state.charts.accuracy) { state.charts.accuracy.destroy(); state.charts.accuracy = null; }
    if (state.charts.loss) { state.charts.loss.destroy(); state.charts.loss = null; }

    // clear results panels
    ['confusion-container', 'samples-container', 'class-acc-container', 'model-summary-container'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
}

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    plugins: {
        legend: {
            labels: {
                color: '#94a3b8',
                font: { family: 'Inter', size: 11 },
                boxWidth: 12,
                padding: 12
            }
        }
    },
    scales: {
        x: {
            title: { display: true, text: 'Epoch', color: '#64748b', font: { size: 11 } },
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.05)' }
        }
    }
};

function updateCharts(data) {
    // Accuracy chart
    const accCtx = document.getElementById('accuracy-chart');
    if (!accCtx) return;

    if (!state.charts.accuracy) {
        state.charts.accuracy = new Chart(accCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Train Accuracy',
                        data: [],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99,102,241,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Val Accuracy',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16,185,129,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: { ...chartDefaults.scales.y, title: { display: true, text: 'Accuracy', color: '#64748b', font: { size: 11 } } }
                }
            }
        });
    }

    state.charts.accuracy.data.labels.push(data.epoch);
    state.charts.accuracy.data.datasets[0].data.push(data.accuracy);
    state.charts.accuracy.data.datasets[1].data.push(data.val_accuracy);
    state.charts.accuracy.update();

    // Loss chart
    const lossCtx = document.getElementById('loss-chart');
    if (!lossCtx) return;

    if (!state.charts.loss) {
        state.charts.loss = new Chart(lossCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Train Loss',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245,158,11,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Val Loss',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239,68,68,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: { ...chartDefaults.scales.y, title: { display: true, text: 'Loss', color: '#64748b', font: { size: 11 } } }
                }
            }
        });
    }

    state.charts.loss.data.labels.push(data.epoch);
    state.charts.loss.data.datasets[0].data.push(data.loss);
    state.charts.loss.data.datasets[1].data.push(data.val_loss);
    state.charts.loss.update();
}

// -------------------------------------------------------------------------
//  RESULTS
// -------------------------------------------------------------------------
function showResults(data) {
    renderConfusionMatrix(data);
    renderSamplePredictions(data);
    renderClassAccuracy(data);
}

function renderConfusionMatrix(data) {
    const container = document.getElementById('confusion-container');
    if (!container) return;
    container.style.display = 'block';

    const content = container.querySelector('.confusion-content');
    if (!content) return;
    content.innerHTML = '';

    const cm = data.confusion_matrix;
    const n = cm.length;
    const maxVal = Math.max(...cm.flat());

    // header with test accuracy
    const header = document.createElement('div');
    header.style.cssText = 'margin-bottom:8px;font-size:0.8rem;color:var(--text-secondary);';
    header.innerHTML = `Test Accuracy: <strong style="color:var(--accent)">${(data.test_accuracy * 100).toFixed(2)}%</strong> | Test Loss: ${data.test_loss.toFixed(4)}`;
    content.appendChild(header);

    const grid = document.createElement('div');
    grid.className = 'confusion-grid';
    grid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const cell = document.createElement('div');
            cell.className = 'confusion-cell';
            const val = cm[i][j];
            const intensity = maxVal > 0 ? val / maxVal : 0;

            if (i === j) {
                cell.style.background = `rgba(99,102,241,${0.2 + intensity * 0.8})`;
            } else {
                cell.style.background = `rgba(239,68,68,${intensity * 0.6})`;
            }

            cell.textContent = val > 0 ? val : '';
            cell.title = `True: ${data.labels[i]}, Pred: ${data.labels[j]}, Count: ${val}`;
            grid.appendChild(cell);
        }
    }

    content.appendChild(grid);

    // labels
    const labels = document.createElement('div');
    labels.className = 'confusion-labels';
    data.labels.forEach(l => {
        const span = document.createElement('span');
        span.textContent = l;
        labels.appendChild(span);
    });
    content.appendChild(labels);
}

function renderSamplePredictions(data) {
    const container = document.getElementById('samples-container');
    if (!container) return;
    container.style.display = 'block';

    const content = container.querySelector('.samples-content');
    if (!content) return;
    content.innerHTML = '';

    const grid = document.createElement('div');
    grid.className = 'samples-grid';

    data.samples.forEach(s => {
        const card = document.createElement('div');
        card.className = `sample-card ${s.correct ? 'correct' : 'incorrect'}`;

        const canvas = document.createElement('canvas');
        const size = data.shape[0];
        canvas.width = size;
        canvas.height = data.shape.length > 1 ? data.shape[1] : size;
        drawImage(canvas, s.pixels, data.shape, data.channels);

        const predDiv = document.createElement('div');
        predDiv.className = 'pred-label';
        predDiv.style.color = s.correct ? 'var(--success)' : 'var(--danger)';
        predDiv.textContent = s.pred_name;

        const confDiv = document.createElement('div');
        confDiv.className = 'confidence';
        confDiv.textContent = `${s.confidence}% ${s.correct ? '✓' : '✗ (was ' + s.true_name + ')'}`;

        card.appendChild(canvas);
        card.appendChild(predDiv);
        card.appendChild(confDiv);
        grid.appendChild(card);
    });

    content.appendChild(grid);
}

function renderClassAccuracy(data) {
    const container = document.getElementById('class-acc-container');
    if (!container) return;
    container.style.display = 'block';

    const content = container.querySelector('.class-acc-content');
    if (!content) return;
    content.innerHTML = '';

    const bars = document.createElement('div');
    bars.className = 'class-bars';

    data.per_class.forEach((c, i) => {
        const hue = (i / data.per_class.length) * 280 + 180;
        const bar = document.createElement('div');
        bar.className = 'class-bar';
        bar.innerHTML = `
            <span class="class-name">${c.label}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:${c.accuracy}%;background:hsl(${hue},70%,55%)"></div>
            </div>
            <span class="bar-value">${c.accuracy}%</span>
        `;
        bars.appendChild(bar);
    });

    content.appendChild(bars);
}

// -------------------------------------------------------------------------
//  IMAGE DRAWING
// -------------------------------------------------------------------------
function drawImage(canvas, pixels, shape, channels) {
    const ctx = canvas.getContext('2d');
    const w = shape[0];
    const h = shape.length > 1 ? shape[1] : 1;

    if (channels === 0 || shape.length === 1) {
        // tabular data, just draw a placeholder
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, w, h);
        return;
    }

    const imgData = ctx.createImageData(w, h);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const pixIdx = (y * w + x) * 4;

            if (channels === 1) {
                // grayscale
                const val = Math.round(pixels[y][x] * 255);
                imgData.data[pixIdx]     = val;
                imgData.data[pixIdx + 1] = val;
                imgData.data[pixIdx + 2] = val;
                imgData.data[pixIdx + 3] = 255;
            } else if (channels === 3) {
                // RGB
                imgData.data[pixIdx]     = Math.round(pixels[y][x][0] * 255);
                imgData.data[pixIdx + 1] = Math.round(pixels[y][x][1] * 255);
                imgData.data[pixIdx + 2] = Math.round(pixels[y][x][2] * 255);
                imgData.data[pixIdx + 3] = 255;
            }
        }
    }

    ctx.putImageData(imgData, 0, 0);
}
