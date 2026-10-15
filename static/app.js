/**
 * ScratchNet — 7-Class Flower Classifier
 * Frontend Controller & Interactive Neural Network Client
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const dropzoneEmpty = document.getElementById('dropzoneEmpty');
    const dropzonePreview = document.getElementById('dropzonePreview');
    const fileInput = document.getElementById('fileInput');
    const browseFilesBtn = document.getElementById('browseFilesBtn');
    const previewImage = document.getElementById('previewImage');
    const previewFilename = document.getElementById('previewFilename');
    const previewFilesize = document.getElementById('previewFilesize');
    const clearImageBtn = document.getElementById('clearImageBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const analyzeBtnText = document.getElementById('analyzeBtnText');

    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContent = document.getElementById('resultContent');
    const winnerName = document.getElementById('winnerName');
    const winnerScientific = document.getElementById('winnerScientific');
    const winnerIcon = document.getElementById('winnerIcon');
    const winnerConfidence = document.getElementById('winnerConfidence');
    const winnerDescription = document.getElementById('winnerDescription');
    const winnerFamily = document.getElementById('winnerFamily');
    const winnerSeason = document.getElementById('winnerSeason');
    const probabilitiesList = document.getElementById('probabilitiesList');
    const jsonCodeOutput = document.getElementById('jsonCodeOutput');

    const hardwareStatusBadge = document.getElementById('hardwareStatusBadge');
    const hardwareStatusText = document.getElementById('hardwareStatusText');
    const inferenceDeviceBadge = document.getElementById('inferenceDeviceBadge');
    const samplesContainer = document.getElementById('samplesContainer');

    // State
    let currentFile = null;

    // 1. Initial Health Check
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            if (data.cuda_available) {
                hardwareStatusText.textContent = `⚡ CUDA Acceleration (${data.device.toUpperCase()})`;
                hardwareStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
            } else {
                hardwareStatusText.textContent = `💻 CPU Active (${data.device.toUpperCase()})`;
                hardwareStatusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
            }
            inferenceDeviceBadge.textContent = `${data.num_classes} Classes • ${data.device.toUpperCase()}`;
        })
        .catch(() => {
            hardwareStatusText.textContent = '● Backend Connected';
        });

    // 2. Fetch & Render Quick Test Samples
    fetch('/api/samples')
        .then(res => res.json())
        .then(data => {
            if (!data.samples || data.samples.length === 0) {
                samplesContainer.innerHTML = '<span class="samples-hint">No pre-loaded samples found.</span>';
                return;
            }

            samplesContainer.innerHTML = '';
            data.samples.forEach(sample => {
                const card = document.createElement('div');
                card.className = 'sample-card';
                card.setAttribute('data-category', sample.category);
                card.innerHTML = `
                    <div class="sample-thumb-wrapper">
                        <img src="${sample.url}" alt="${sample.display_name}" class="sample-thumb" loading="lazy">
                    </div>
                    <div class="sample-name-badge">
                        <span>${sample.icon}</span>
                        <span>${sample.display_name}</span>
                    </div>
                `;

                card.addEventListener('click', () => loadSampleImage(sample.url, sample.category));
                samplesContainer.appendChild(card);
            });
        })
        .catch(err => {
            samplesContainer.innerHTML = '<span class="samples-hint">Could not load samples.</span>';
            console.error(err);
        });

    // 3. File Input Trigger
    browseFilesBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener('click', () => {
        if (!currentFile) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // 4. Drag & Drop Event Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleSelectedFile(dt.files[0]);
        }
    });

    // 5. Global Clipboard Paste Handler (Ctrl+V)
    window.addEventListener('paste', (e) => {
        if (e.clipboardData && e.clipboardData.items) {
            const items = e.clipboardData.items;
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    const blob = items[i].getAsFile();
                    handleSelectedFile(blob, 'pasted_image.png');
                    break;
                }
            }
        }
    });

    // 6. Clear Image Button
    clearImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetDropzone();
    });

    function resetDropzone() {
        currentFile = null;
        fileInput.value = '';
        dropzoneEmpty.style.display = 'flex';
        dropzonePreview.style.display = 'none';
        previewImage.src = '';
        analyzeBtn.disabled = true;
        
        // Remove active state on sample cards
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
    }

    function handleSelectedFile(file, overrideName = null) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP, BMP).');
            return;
        }

        currentFile = file;
        const name = overrideName || file.name || 'image.jpg';
        const sizeFormatted = (file.size / 1024).toFixed(1) + ' KB';

        previewFilename.textContent = name;
        previewFilesize.textContent = sizeFormatted;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropzoneEmpty.style.display = 'none';
            dropzonePreview.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // 7. Load & Auto-Analyze a Sample Flower
    function loadSampleImage(url, category) {
        // Highlight active sample card
        document.querySelectorAll('.sample-card').forEach(c => {
            if (c.getAttribute('data-category') === category) {
                c.classList.add('active');
            } else {
                c.classList.remove('active');
            }
        });

        fetch(url)
            .then(res => res.blob())
            .then(blob => {
                const sampleFile = new File([blob], `${category}.jpg`, { type: 'image/jpeg' });
                handleSelectedFile(sampleFile, `${category}_sample.jpg`);
                // Auto trigger detection for quick feedback
                runInference(sampleFile);
            })
            .catch(err => {
                console.error('Failed to load sample image:', err);
            });
    }

    // 8. Analyze Button Trigger
    analyzeBtn.addEventListener('click', () => {
        if (currentFile) {
            runInference(currentFile);
        }
    });

    // 9. Inference Execution Function
    function runInference(file) {
        // UI Loading State
        analyzeBtn.disabled = true;
        btnSpinner.style.display = 'inline-block';
        analyzeBtnText.textContent = 'Classifying Flower...';

        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/predict', {
            method: 'POST',
            body: formData,
        })
            .then(res => {
                if (!res.ok) {
                    throw new Error(`Inference error: HTTP ${res.status}`);
                }
                return res.json();
            })
            .then(data => {
                if (data.success) {
                    renderPredictionResults(data);
                } else {
                    alert('Prediction failed: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error('Error during prediction:', err);
                alert('Could not complete prediction. Please verify the server is running.');
            })
            .finally(() => {
                analyzeBtn.disabled = false;
                btnSpinner.style.display = 'none';
                analyzeBtnText.textContent = 'Analyze Flower';
            });
    }

    // 10. Render Prediction Results & Progress Bars
    function renderPredictionResults(data) {
        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'flex';

        // Winner Card
        const topInfo = data.flower_info || {};
        winnerName.textContent = data.display_name;
        winnerScientific.textContent = topInfo.scientific_name || '';
        winnerIcon.textContent = topInfo.icon || '🌸';
        winnerConfidence.textContent = data.confidence_pct;
        winnerDescription.textContent = topInfo.description || '';
        winnerFamily.textContent = topInfo.family || 'Botanical';
        winnerSeason.textContent = topInfo.season || 'Annual';

        if (topInfo.color) {
            winnerConfidence.style.color = topInfo.color;
            winnerName.style.color = topInfo.color;
        }

        // Probabilities List
        probabilitiesList.innerHTML = '';
        if (data.ranked_list && data.ranked_list.length > 0) {
            data.ranked_list.forEach((item, index) => {
                const probItem = document.createElement('div');
                probItem.className = 'prob-item';

                const isWinner = index === 0;
                probItem.innerHTML = `
                    <div class="prob-info-row">
                        <div class="prob-name-group" style="${isWinner ? 'color: ' + item.color + ';' : ''}">
                            <span>${item.icon}</span>
                            <span>${item.display_name}</span>
                        </div>
                        <span class="prob-pct-val" style="${isWinner ? 'color: ' + item.color + ';' : ''}">
                            ${item.percentage}
                        </span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" id="probBar_${item.name}" style="background: ${item.color}; width: 0%;"></div>
                    </div>
                `;
                probabilitiesList.appendChild(probItem);

                // Animate progress bar fill width after insertion
                requestAnimationFrame(() => {
                    setTimeout(() => {
                        const bar = document.getElementById(`probBar_${item.name}`);
                        if (bar) {
                            bar.style.width = `${item.percent_value}%`;
                        }
                    }, 50 * index);
                });
            });
        }

        // Raw JSON output
        jsonCodeOutput.textContent = JSON.stringify(data, null, 2);
    }
});
