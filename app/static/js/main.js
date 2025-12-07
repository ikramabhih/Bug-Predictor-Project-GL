let selectedModel = 'rf';
let uploadedFiles = [];

// Sélection du modèle
document.querySelectorAll('.model-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        selectedModel = this.dataset.model;
        console.log('Model selected:', selectedModel);
    });
});

// Gestion des tabs
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        this.classList.add('active');
        const tabId = this.dataset.tab + '-tab';
        document.getElementById(tabId).classList.add('active');
        
        document.getElementById('results').style.display = 'none';
        document.getElementById('batchResults').style.display = 'none';
    });
});

// Prédiction avec métriques manuelles - CORRIGÉ AVEC 21 MÉTRIQUES
document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Collecter TOUTES les 21 métriques requises
    const features = {
        'loc': parseFloat(document.getElementById('loc').value),
        'v(g)': parseFloat(document.getElementById('vg').value),
        'ev(g)': parseFloat(document.getElementById('evg').value),
        'iv(g)': parseFloat(document.getElementById('ivg').value),
        'n': parseFloat(document.getElementById('n').value),
        'v': parseFloat(document.getElementById('v').value),
        'l': parseFloat(document.getElementById('l').value),
        'd': parseFloat(document.getElementById('d').value),
        'i': parseFloat(document.getElementById('i').value),
        'e': parseFloat(document.getElementById('e').value),
        'b': parseFloat(document.getElementById('b').value),
        't': parseFloat(document.getElementById('t').value),
        'lOCode': parseFloat(document.getElementById('lOCode').value),
        'lOComment': parseFloat(document.getElementById('lOComment').value),
        'lOBlank': parseFloat(document.getElementById('lOBlank').value),
        'locCodeAndComment': parseFloat(document.getElementById('locCodeAndComment').value),
        'uniq_Op': parseFloat(document.getElementById('uniq_Op').value),
        'uniq_Opnd': parseFloat(document.getElementById('uniq_Opnd').value),
        'total_Op': parseFloat(document.getElementById('total_Op').value),
        'total_Opnd': parseFloat(document.getElementById('total_Opnd').value),
        'branchCount': parseFloat(document.getElementById('branchCount').value)
    };
    
    console.log('Sending features:', features);
    showResults();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: selectedModel, features: features})
        });
        
        const data = await response.json();
        console.log('Response:', data);
        
        if (data.status === 'success') {
            displayResults(data);
        } else {
            alert('Erreur: ' + data.error);
            console.error('Error response:', data);
        }
    } catch (error) {
        alert('Erreur de connexion: ' + error);
        console.error('Fetch error:', error);
    }
});

// Analyser du code copié
async function analyzeCode() {
    const code = document.getElementById('codeInput').value.trim();
    
    if (!code) {
        alert('Veuillez coller du code à analyser');
        return;
    }
    
    console.log('Analyzing code, length:', code.length);
    showResults();
    
    try {
        const response = await fetch('/api/analyze/code', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: selectedModel, code: code})
        });
        
        const data = await response.json();
        console.log('Code analysis response:', data);
        
        if (data.status === 'success') {
            displayResults(data);
        } else {
            alert('Erreur: ' + data.error);
            console.error('Error response:', data);
        }
    } catch (error) {
        alert('Erreur: ' + error);
        console.error('Fetch error:', error);
    }
}

// Gestion de l'upload
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

function handleFiles(files) {
    uploadedFiles = Array.from(files);
    displayFileList();
    document.getElementById('analyzeFilesBtn').style.display = 'block';
}

function displayFileList() {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span class="file-name">📄 ${file.name}</span>
            <span class="remove-btn" onclick="removeFile(${index})">✕</span>
        `;
        fileList.appendChild(fileItem);
    });
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    displayFileList();
    if (uploadedFiles.length === 0) {
        document.getElementById('analyzeFilesBtn').style.display = 'none';
    }
}

async function analyzeFiles() {
    if (uploadedFiles.length === 0) {
        alert('Aucun fichier sélectionné');
        return;
    }
    
    console.log('Analyzing files:', uploadedFiles.map(f => f.name));
    
    const formData = new FormData();
    uploadedFiles.forEach(file => formData.append('files', file));
    formData.append('model', selectedModel);
    
    document.getElementById('batchResults').style.display = 'block';
    document.getElementById('batchResultsContent').innerHTML = '<p>⏳ Analyse en cours...</p>';
    
    try {
        const response = await fetch('/api/analyze/files', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        console.log('File analysis response:', data);
        
        if (data.status === 'success') {
            displayBatchResults(data.results);
        } else {
            alert('Erreur: ' + data.error);
            console.error('Error response:', data);
        }
    } catch (error) {
        alert('Erreur: ' + error);
        console.error('Fetch error:', error);
    }
}

async function analyzeGit() {
    const gitUrl = document.getElementById('gitUrl').value.trim();
    
    if (!gitUrl) {
        alert('Veuillez entrer une URL Git');
        return;
    }
    
    console.log('Analyzing Git repo:', gitUrl);
    
    const gitStatus = document.getElementById('gitStatus');
    gitStatus.className = 'git-status loading';
    gitStatus.textContent = '⏳ Clonage et analyse du dépôt...';
    
    document.getElementById('batchResults').style.display = 'block';
    document.getElementById('batchResultsContent').innerHTML = '<p>⏳ Clone en cours...</p>';
    
    try {
        const response = await fetch('/api/analyze/git', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: selectedModel, git_url: gitUrl})
        });
        
        const data = await response.json();
        console.log('Git analysis response:', data);
        
        if (data.status === 'success') {
            gitStatus.className = 'git-status success';
            gitStatus.textContent = `✅ Analyse terminée ! ${data.total_files_analyzed} fichiers analysés`;
            displayBatchResults(data.results);
        } else {
            gitStatus.className = 'git-status error';
            gitStatus.textContent = '❌ ' + (data.error || 'Erreur inconnue');
            document.getElementById('batchResultsContent').innerHTML = 
                `<p style="color: #c00;">❌ ${data.error}</p>` +
                (data.details ? `<p style="font-size: 0.9em; color: #666;">${data.details}</p>` : '');
        }
    } catch (error) {
        gitStatus.className = 'git-status error';
        gitStatus.textContent = '❌ Erreur: ' + error;
        console.error('Git analysis error:', error);
    }
}

function showResults() {
    document.getElementById('results').style.display = 'block';
    document.getElementById('batchResults').style.display = 'none';
    document.getElementById('resultCard').style.opacity = '0.5';
    document.getElementById('statusText').textContent = 'Analyse en cours...';
}

function displayResults(data) {
    const resultCard = document.getElementById('resultCard');
    resultCard.style.opacity = '1';
    
    const statusIcon = document.getElementById('statusIcon');
    const statusText = document.getElementById('statusText');
    
    if (data.prediction === 1) {
        statusIcon.textContent = '🐛';
        statusText.textContent = 'Bug Détecté !';
        statusText.style.color = '#ff4444';
    } else {
        statusIcon.textContent = '✅';
        statusText.textContent = 'Aucun Bug Prédit';
        statusText.style.color = '#00C851';
    }
    
    const bugProba = (data.probability.bug * 100).toFixed(1);
    document.getElementById('bugProba').textContent = bugProba + '%';
    
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = data.risk_level.toUpperCase();
    riskBadge.className = 'badge ' + data.risk_level;
    
    document.getElementById('modelUsed').textContent = data.model_used.toUpperCase();
    
    const progressFill = document.getElementById('progressFill');
    progressFill.style.width = bugProba + '%';
    
    // Couleur de la barre selon le risque
    if (data.risk_level === 'high') {
        progressFill.style.background = 'linear-gradient(90deg, #ff4444, #cc0000)';
    } else if (data.risk_level === 'medium') {
        progressFill.style.background = 'linear-gradient(90deg, #ffa500, #ff8c00)';
    } else {
        progressFill.style.background = 'linear-gradient(90deg, #00C851, #007E33)';
    }
}

function displayBatchResults(results) {
    const content = document.getElementById('batchResultsContent');
    
    if (!results || results.length === 0) {
        content.innerHTML = '<p>Aucun fichier analysé.</p>';
        return;
    }
    
    // Trier par probabilité décroissante
    results.sort((a, b) => b.bug_probability - a.bug_probability);
    
    let html = `
        <table class="batch-results-table">
            <thead>
                <tr>
                    <th>Fichier</th>
                    <th>Probabilité de Bug</th>
                    <th>Risque</th>
                    <th>Prédiction</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    results.forEach(result => {
        const proba = (result.bug_probability * 100).toFixed(1);
        const prediction = result.prediction === 1 ? ' Bug' : ' OK';
        
        html += `
            <tr>
                <td><strong>${result.file}</strong></td>
                <td>${proba}%</td>
                <td><span class="risk-badge-small ${result.risk_level}">${result.risk_level.toUpperCase()}</span></td>
                <td>${prediction}</td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
        <p style="margin-top: 20px; text-align: center; color: #666;">
            <strong>${results.length}</strong> fichiers analysés - 
            <strong style="color: #ff4444;">${results.filter(r => r.risk_level === 'high').length}</strong> à haut risque - 
            <strong style="color: #ffa500;">${results.filter(r => r.risk_level === 'medium').length}</strong> risque moyen - 
            <strong style="color: #00C851;">${results.filter(r => r.risk_level === 'low').length}</strong> faible risque
        </p>
    `;
    
    content.innerHTML = html;
}

console.log('Bug Predictor JS loaded successfully!');