/* playwrite: neural network playground */

// -------------------------------------------------------------------------
//  STATE
// -------------------------------------------------------------------------
let state = {
    dataset: 'mnist',
    layers: [128, 64],
    datasets: {},
    charts: { accuracy: null, loss: null, classDist: null, pca: null },
    totalEpochs: 20,
    startTime: null
};

const CLASS_COLORS = [
    '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#a855f7'
];

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
        if (display) el.addEventListener('input', () => { display.textContent = el.value; });
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
        // auto-load visualization for default
        visualizeDataset();
    } catch (e) {
        console.error('Failed to load datasets:', e);
    }
}

function renderDatasetOptions() {
    const grid = document.getElementById('dataset-grid');
    if (!grid) return;
    grid.innerHTML = '';

    const order = ['mnist', 'fashion_mnist', 'cifar10', 'digits', 'iris', 'wine', 'breast_cancer', 'custom'];

    order.forEach(key => {
        const ds = state.datasets[key];
        if (!ds) return;
        const div = document.createElement('div');
        div.className = `dataset-option ${key === state.dataset ? 'active' : ''}`;
        div.onclick = function() { selectDataset(key, this); };
        div.innerHTML = `
            <div class="ds-name">${ds.name}</div>
            <div class="ds-info">${ds.train_size.toLocaleString()} samples</div>
            <div class="ds-type">${ds.type}</div>
        `;
        grid.appendChild(div);
    });
}

function selectDataset(key, el) {
    state.dataset = key;
    document.querySelectorAll('.dataset-option').forEach(e => e.classList.remove('active'));
    if (el) el.classList.add('active');
    visualizeDataset();
}

// -------------------------------------------------------------------------
//  DATASET VISUALIZATION
// -------------------------------------------------------------------------
async function visualizeDataset() {
    try {
        const res = await fetch('/api/dataset/visualize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset: state.dataset })
        });
        const data = await res.json();
        renderVisualization(data);
    } catch (e) {
        console.error('Failed to visualize:', e);
    }
}

function renderVisualization(data) {
    const panel = document.getElementById('viz-panel');
    if (!panel) return;
    panel.classList.add('visible');

    // hide empty state
    const empty = document.getElementById('empty-state');
    if (empty) empty.style.display = 'none';

    // description
    const descEl = document.getElementById('viz-description');
    if (descEl) descEl.textContent = data.description || '';

    // badges
    const badgesEl = document.getElementById('viz-badges');
    if (badgesEl) {
        badgesEl.innerHTML = `
            <span class="ds-badge samples">${data.total_samples.toLocaleString()} samples</span>
            <span class="ds-badge features">${data.input_shape.join('x')} input</span>
            <span class="ds-badge classes">${data.num_classes} classes</span>
            <span class="ds-badge type-badge">${data.type}</span>
        `;
    }

    // class distribution chart
    renderClassDistribution(data);

    // right panel: type-specific
    const rightPanel = document.getElementById('viz-right');
    if (!rightPanel) return;
    rightPanel.innerHTML = '';

    if (data.type === 'image' && data.class_means) {
        renderClassMeans(rightPanel, data);
    } else if (data.type === 'tabular') {
        if (data.pca_scatter) renderPCAScatter(rightPanel, data);
        if (data.feature_stats) renderFeatureStats(rightPanel, data);
    }
}

function renderClassDistribution(data) {
    if (state.charts.classDist) { state.charts.classDist.destroy(); state.charts.classDist = null; }

    // replace the canvas entirely to reset Chart.js sizing
    const container = document.getElementById('class-dist-chart').parentElement;
    const oldCanvas = document.getElementById('class-dist-chart');
    const newCanvas = document.createElement('canvas');
    newCanvas.id = 'class-dist-chart';
    newCanvas.style.height = '180px';
    oldCanvas.replaceWith(newCanvas);

    const labels = data.class_distribution.map(c => c.label);
    const counts = data.class_distribution.map(c => c.count);
    const colors = labels.map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length]);

    state.charts.classDist = new Chart(newCanvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: colors.map(c => c + '66'),
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#64748b', font: { size: 9 }, maxRotation: 45 },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: '#64748b', font: { size: 9 } },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

function renderClassMeans(container, data) {
    const section = document.createElement('div');
    section.innerHTML = '<div class="viz-title">Average Pixel Intensity per Class</div>';

    const grid = document.createElement('div');
    grid.className = 'mean-images-grid';

    data.class_means.forEach(cm => {
        const card = document.createElement('div');
        card.className = 'mean-image-card';

        const canvas = document.createElement('canvas');
        const shape = data.input_shape;
        canvas.width = shape[0];
        canvas.height = shape.length > 1 ? shape[1] : shape[0];
        drawImage(canvas, cm.pixels, shape, shape.length === 3 ? 3 : 1);

        const lbl = document.createElement('div');
        lbl.className = 'mean-label';
        lbl.textContent = cm.label;

        card.appendChild(canvas);
        card.appendChild(lbl);
        grid.appendChild(card);
    });

    section.appendChild(grid);
    container.appendChild(section);
}

function renderPCAScatter(container, data) {
    const section = document.createElement('div');
    section.innerHTML = `<div class="viz-title">PCA 2D Projection (${(data.pca_variance[0]*100).toFixed(1)}% + ${(data.pca_variance[1]*100).toFixed(1)}% variance)</div>`;

    const canvas = document.createElement('canvas');
    canvas.style.height = '200px';
    section.appendChild(canvas);

    // group points by label
    const groups = {};
    data.pca_scatter.forEach(p => {
        if (!groups[p.label]) groups[p.label] = { points: [], name: p.label_name };
        groups[p.label].points.push({ x: p.x, y: p.y });
    });

    const datasets = Object.entries(groups).map(([label, g], i) => ({
        label: g.name,
        data: g.points,
        backgroundColor: CLASS_COLORS[i % CLASS_COLORS.length] + '99',
        borderColor: CLASS_COLORS[i % CLASS_COLORS.length],
        borderWidth: 1,
        pointRadius: 3,
        pointHoverRadius: 5
    }));

    if (state.charts.pca) state.charts.pca.destroy();
    state.charts.pca = new Chart(canvas.getContext('2d'), {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 8, padding: 8 },
                    position: 'bottom'
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'PC1', color: '#64748b', font: { size: 10 } },
                    ticks: { color: '#64748b', font: { size: 9 } },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    title: { display: true, text: 'PC2', color: '#64748b', font: { size: 10 } },
                    ticks: { color: '#64748b', font: { size: 9 } },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });

    container.appendChild(section);
}

function renderFeatureStats(container, data) {
    const section = document.createElement('div');
    section.style.marginTop = '12px';
    section.innerHTML = '<div class="viz-title">Feature Statistics (normalized)</div>';

    const table = document.createElement('table');
    table.className = 'feat-table';
    table.innerHTML = `
        <thead><tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>
        <tbody>${data.feature_stats.map(f => `
            <tr>
                <td style="color:var(--text-primary);font-family:Inter,sans-serif;">${f.name}</td>
                <td>${f.mean.toFixed(2)}</td>
                <td>${f.std.toFixed(2)}</td>
                <td>${f.min.toFixed(2)}</td>
                <td>${f.max.toFixed(2)}</td>
            </tr>
        `).join('')}</tbody>
    `;

    section.appendChild(table);
    container.appendChild(section);
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

    fetch('/api/upload', { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
        if (data.error) { if (label) label.textContent = 'Error: ' + data.error; return; }
        state.dataset = 'custom';
        state.datasets['custom'] = {
            name: data.name, train_size: data.samples,
            num_classes: data.classes, labels: data.labels, type: 'tabular'
        };
        renderDatasetOptions();
        visualizeDataset();
        if (label) label.textContent = `Uploaded: ${data.name} (${data.samples} samples)`;
    })
    .catch(() => { if (label) label.textContent = 'Upload failed'; });
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
            <button class="remove-btn" onclick="removeLayer(${i})" title="Remove">&times;</button>
        `;
        list.appendChild(item);
    });
}

function addLayer() { state.layers.push(64); renderLayers(); }
function removeLayer(i) { if (state.layers.length > 1) { state.layers.splice(i, 1); renderLayers(); } }
function updateLayer(i, v) { state.layers[i] = parseInt(v) || 64; }

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

    resetTrainingCharts();
    setTrainingUI(true);

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
                if (done) { setTrainingUI(false); return; }
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        try { handleTrainingEvent(JSON.parse(line.slice(6))); } catch (e) {}
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
        case 'model_info': showModelInfo(data); break;
        case 'epoch': updateEpochProgress(data); updateCharts(data); break;
        case 'complete': showResults(data); setTrainingUI(false); break;
        case 'stopped': setTrainingUI(false); break;
        case 'error': showError(data.message); setTrainingUI(false); break;
    }
}

function stopTraining() { fetch('/api/stop', { method: 'POST' }); }

// -------------------------------------------------------------------------
//  UI UPDATES
// -------------------------------------------------------------------------
function setTrainingUI(training) {
    const trainBtn = document.getElementById('train-btn');
    const stopBtn = document.getElementById('stop-btn');
    const progress = document.getElementById('progress-container');
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');

    if (trainBtn) trainBtn.disabled = training;
    if (stopBtn) stopBtn.style.display = training ? 'flex' : 'none';
    if (progress) progress.classList.toggle('visible', training);
    if (dot) dot.classList.toggle('active', training);
    if (text) text.textContent = training ? 'Training...' : 'Ready';
}

function showModelInfo(data) {
    const el = document.getElementById('model-summary');
    if (el) { el.textContent = data.summary.join('\n'); el.parentElement.style.display = 'block'; }
    const p = document.getElementById('param-count');
    if (p) p.textContent = data.total_params.toLocaleString();
}

function updateEpochProgress(data) {
    const fill = document.getElementById('progress-fill');
    const label = document.getElementById('epoch-label');
    const time = document.getElementById('time-label');

    const pct = (data.epoch / state.totalEpochs) * 100;
    if (fill) fill.style.width = pct + '%';
    if (label) label.textContent = `Epoch ${data.epoch} / ${state.totalEpochs}`;

    const elapsed = ((Date.now() - state.startTime) / 1000).toFixed(0);
    if (time) time.textContent = `${elapsed}s elapsed`;

    const set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
    set('stat-accuracy', (data.accuracy * 100).toFixed(2) + '%');
    set('stat-val-accuracy', (data.val_accuracy * 100).toFixed(2) + '%');
    set('stat-loss', data.loss.toFixed(4));
    set('stat-best', (data.best_val_accuracy * 100).toFixed(2) + '%');
}

function showError(msg) {
    const p = document.getElementById('progress-container');
    if (p) { p.innerHTML = `<div style="color:var(--danger);padding:14px;">Error: ${msg}</div>`; p.classList.add('visible'); }
}

// -------------------------------------------------------------------------
//  CHARTS
// -------------------------------------------------------------------------
function resetTrainingCharts() {
    if (state.charts.accuracy) { state.charts.accuracy.destroy(); state.charts.accuracy = null; }
    if (state.charts.loss) { state.charts.loss.destroy(); state.charts.loss = null; }
    ['confusion-container', 'samples-container', 'class-acc-container', 'model-summary-container'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
}

const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    plugins: {
        legend: {
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 10 }, boxWidth: 10, padding: 10 }
        }
    },
    scales: {
        x: { title: { display: true, text: 'Epoch', color: '#64748b', font: { size: 10 } }, ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
    }
};

function makeDataset(label, color) {
    return { label, data: [], borderColor: color, backgroundColor: color + '18', borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0 };
}

function updateCharts(data) {
    // accuracy
    const accCtx = document.getElementById('accuracy-chart');
    if (!accCtx) return;
    if (!state.charts.accuracy) {
        state.charts.accuracy = new Chart(accCtx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [makeDataset('Train', '#6366f1'), makeDataset('Val', '#10b981')] },
            options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, title: { display: true, text: 'Accuracy', color: '#64748b', font: { size: 10 } } } } }
        });
    }
    state.charts.accuracy.data.labels.push(data.epoch);
    state.charts.accuracy.data.datasets[0].data.push(data.accuracy);
    state.charts.accuracy.data.datasets[1].data.push(data.val_accuracy);
    state.charts.accuracy.update();

    // loss
    const lossCtx = document.getElementById('loss-chart');
    if (!lossCtx) return;
    if (!state.charts.loss) {
        state.charts.loss = new Chart(lossCtx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [makeDataset('Train', '#f59e0b'), makeDataset('Val', '#ef4444')] },
            options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, title: { display: true, text: 'Loss', color: '#64748b', font: { size: 10 } } } } }
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

    const header = document.createElement('div');
    header.style.cssText = 'margin-bottom:8px;font-size:0.75rem;color:var(--text-secondary);';
    header.innerHTML = `Test Accuracy: <strong style="color:var(--accent)">${(data.test_accuracy * 100).toFixed(2)}%</strong> | Loss: ${data.test_loss.toFixed(4)}`;
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
            cell.style.background = i === j
                ? `rgba(99,102,241,${0.2 + intensity * 0.8})`
                : `rgba(239,68,68,${intensity * 0.6})`;
            cell.textContent = val > 0 ? val : '';
            cell.title = `True: ${data.labels[i]}, Pred: ${data.labels[j]}, Count: ${val}`;
            grid.appendChild(cell);
        }
    }
    content.appendChild(grid);

    const labels = document.createElement('div');
    labels.className = 'confusion-labels';
    data.labels.forEach(l => { const s = document.createElement('span'); s.textContent = l; labels.appendChild(s); });
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
        canvas.width = data.shape[0];
        canvas.height = data.shape.length > 1 ? data.shape[1] : data.shape[0];
        drawImage(canvas, s.pixels, data.shape, data.channels);

        const pred = document.createElement('div');
        pred.className = 'pred-label';
        pred.style.color = s.correct ? 'var(--success)' : 'var(--danger)';
        pred.textContent = s.pred_name;

        const conf = document.createElement('div');
        conf.className = 'confidence';
        conf.textContent = `${s.confidence}% ${s.correct ? '✓' : '✗ (' + s.true_name + ')'}`;

        card.appendChild(canvas);
        card.appendChild(pred);
        card.appendChild(conf);
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
            <div class="bar-track"><div class="bar-fill" style="width:${c.accuracy}%;background:hsl(${hue},70%,55%)"></div></div>
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
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, w, h);
        return;
    }

    const imgData = ctx.createImageData(w, h);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const pi = (y * w + x) * 4;
            if (channels === 1) {
                const v = Math.round(pixels[y][x] * 255);
                imgData.data[pi] = v; imgData.data[pi+1] = v; imgData.data[pi+2] = v; imgData.data[pi+3] = 255;
            } else if (channels === 3) {
                imgData.data[pi]   = Math.round(pixels[y][x][0] * 255);
                imgData.data[pi+1] = Math.round(pixels[y][x][1] * 255);
                imgData.data[pi+2] = Math.round(pixels[y][x][2] * 255);
                imgData.data[pi+3] = 255;
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);
}
