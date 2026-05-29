// Global Variables
let micStream = null;
let scriptProcessor = null;
let leftChannel = [];
let recordingLength = 0;
let sampleRate = 16000;
let recordTimerInterval = null;
let recordStartTime = null;
let audioContext = null;
let analyser = null;
let dataArray = null;
let animationFrameId = null;
let isRecording = false;
let selectedFile = null;

// DOM Elements
const asrModelSelect = document.getElementById('asr-model-select');
const tabRecordBtn = document.getElementById('tab-record-btn');
const tabUploadBtn = document.getElementById('tab-upload-btn');
const recordContent = document.getElementById('record-content');
const uploadContent = document.getElementById('upload-content');
const visualizerCanvas = document.getElementById('visualizer');
const recordTimer = document.getElementById('record-timer');
const recordBtn = document.getElementById('record-btn');
const recordStatus = document.getElementById('record-status');
const dropzone = document.getElementById('dropzone');
const browseBtn = document.getElementById('browse-btn');
const fileInput = document.getElementById('file-input');
const fileInfoText = document.getElementById('file-info-text');
const asrOutput = document.getElementById('asr-output');
const copyAsrBtn = document.getElementById('copy-asr-btn');
const translateBtn = document.getElementById('translate-btn');
const copyTransBtn = document.getElementById('copy-trans-btn');
const translationOutput = document.getElementById('translation-output');
const dictionaryTermsList = document.getElementById('dictionary-terms-list');
const termsCount = document.getElementById('terms-count');
const loadingOverlay = document.getElementById('loading-overlay');
const overlayTitle = document.getElementById('overlay-title');
const overlaySubtitle = document.getElementById('overlay-subtitle');

// Initialize on Load
window.addEventListener('DOMContentLoaded', () => {
    fetchModels();
    setupEventListeners();
    setupCanvas();
});

// Setup Canvas size
function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = visualizerCanvas.getBoundingClientRect();
    visualizerCanvas.width = rect.width * dpr;
    visualizerCanvas.height = rect.height * dpr;
    const ctx = visualizerCanvas.getContext('2d');
    ctx.scale(dpr, dpr);
    drawIdleWave();
}

// Fetch models list from API
async function fetchModels() {
    try {
        const response = await fetch('/api/models');
        if (response.ok) {
            const data = await response.json();
            asrModelSelect.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model === 'whisper-lora-nghe-tinh' ? 'Whisper LoRA (Nghe Tĩnh)' : 
                                    model === 'whisper-dora-nghe-tinh' ? 'Whisper DoRA (Nghe Tĩnh)' : model;
                if (model === data.active) {
                    option.selected = true;
                }
                asrModelSelect.appendChild(option);
            });
        }
    } catch (e) {
        console.error('Error fetching models:', e);
    }
}

// Event Listeners setup
function setupEventListeners() {
    // Tabs switching
    tabRecordBtn.addEventListener('click', () => {
        tabRecordBtn.classList.add('active');
        tabUploadBtn.classList.remove('active');
        recordContent.classList.remove('hidden');
        uploadContent.classList.add('hidden');
        setupCanvas();
    });

    tabUploadBtn.addEventListener('click', () => {
        tabUploadBtn.classList.add('active');
        tabRecordBtn.classList.remove('active');
        uploadContent.classList.remove('hidden');
        recordContent.classList.add('hidden');
        if (isRecording) {
            stopRecording();
        }
    });

    // Recording action
    recordBtn.addEventListener('click', () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    // Upload & Drag-and-drop
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', handleFileSelect);

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });

    // Translate action
    translateBtn.addEventListener('click', handleTranslate);

    // Text area validation for Translate button
    asrOutput.addEventListener('input', () => {
        translateBtn.disabled = asrOutput.value.trim().length === 0;
    });

    // Copy buttons
    copyAsrBtn.addEventListener('click', () => copyToClipboard(asrOutput.value, copyAsrBtn));
    copyTransBtn.addEventListener('click', () => {
        const text = translationOutput.innerText.replace(translationOutput.querySelector('.placeholder-text')?.innerText || '', '');
        copyToClipboard(text, copyTransBtn);
    });

    // Window resize
    window.addEventListener('resize', setupCanvas);
}

// Handle selected file
async function handleFileSelect() {
    const file = fileInput.files[0];
    if (!file) return;

    selectedFile = file;
    fileInfoText.textContent = `Đã chọn: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    fileInfoText.classList.remove('hidden');
    
    showLoader('Đang xử lý file...', 'Giải mã định dạng âm thanh trên trình duyệt...');
    
    try {
        // Read file bytes
        const arrayBuffer = await file.arrayBuffer();
        
        // Create temporary AudioContext to decode
        const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        // Decode audio data
        const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);
        
        // Extract mono channel data (first channel)
        const channelData = audioBuffer.getChannelData(0);
        
        // Whisper expects 16000Hz. Resample if necessary
        let finalSamples = channelData;
        const targetRate = 16000;
        
        if (audioBuffer.sampleRate !== targetRate) {
            console.log(`[Frontend] Resampling uploaded file from ${audioBuffer.sampleRate}Hz to ${targetRate}Hz...`);
            finalSamples = resampleBuffer(channelData, audioBuffer.sampleRate, targetRate);
        }
        
        // Encode to standard WAV Blob
        const wavBlob = createWavBlob(finalSamples, targetRate);
        tempCtx.close();
        
        // Send to backend as a clean, standardized WAV file
        uploadAndTranscribe(wavBlob, 'uploaded_audio.wav');
    } catch (err) {
        console.error('Error decoding audio file:', err);
        hideLoader();
        alert('Lỗi: Không thể giải mã file âm thanh này. Vui lòng chọn file ở định dạng phổ biến như .wav, .mp3, .m4a.');
    }
}

// Resampling helper using nearest-neighbor interpolation
function resampleBuffer(samples, fromRate, toRate) {
    const ratio = fromRate / toRate;
    const newLength = Math.round(samples.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        const nextIndex = Math.floor(i * ratio);
        result[i] = samples[nextIndex];
    }
    return result;
}

// Show/Hide loader overlay
function showLoader(title, subtitle) {
    overlayTitle.textContent = title;
    overlaySubtitle.textContent = subtitle;
    loadingOverlay.classList.remove('hidden');
}

function hideLoader() {
    loadingOverlay.classList.add('hidden');
}

// Audio Recording Logic using Web Audio API for native WAV generation
async function startRecording() {
    leftChannel = [];
    recordingLength = 0;
    
    try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Whisper expects 16000Hz. Request 16000Hz from AudioContext
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
        sampleRate = audioContext.sampleRate;
        
        const source = audioContext.createMediaStreamSource(micStream);
        
        // scriptProcessor to capture raw float32 samples
        scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);
        
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        
        source.connect(analyser);
        analyser.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
        
        scriptProcessor.onaudioprocess = function(e) {
            if (!isRecording) return;
            const left = e.inputBuffer.getChannelData(0);
            leftChannel.push(new Float32Array(left));
            recordingLength += left.length;
        };

        isRecording = true;
        recordBtn.classList.add('recording');
        recordStatus.textContent = 'Đang ghi âm... Nhấn nút đỏ để dừng';
        
        // Start timer
        recordStartTime = Date.now();
        recordTimerInterval = setInterval(updateTimer, 1000);
        
        // Start animation
        drawWave();
    } catch (err) {
        console.error('Error opening microphone:', err);
        alert('Không thể mở microphone. Vui lòng cấp quyền truy cập mic.');
        isRecording = false;
        recordBtn.classList.remove('recording');
        recordStatus.textContent = 'Ghi âm lỗi. Sẵn sàng thu âm';
    }
}

function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    recordBtn.classList.remove('recording');
    recordStatus.textContent = 'Đang xử lý âm thanh...';
    
    clearInterval(recordTimerInterval);
    cancelAnimationFrame(animationFrameId);
    drawIdleWave();
    
    if (scriptProcessor) {
        scriptProcessor.disconnect();
    }
    if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    
    // Convert accumulated buffers to WAV format
    const flatSamples = flattenChannelBuffer(leftChannel, recordingLength);
    const wavBlob = createWavBlob(flatSamples, sampleRate);
    uploadAndTranscribe(wavBlob, 'recorded_audio.wav');
}

// Helpers for WAV generation in the browser
function flattenChannelBuffer(channelBuffer, length) {
    const result = new Float32Array(length);
    let offset = 0;
    for (let i = 0; i < channelBuffer.length; i++) {
        result.set(channelBuffer[i], offset);
        offset += channelBuffer[i].length;
    }
    return result;
}

function createWavBlob(samples, rate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM Format
    view.setUint16(22, 1, true); // Mono Channel
    view.setUint32(24, rate, true); // Sample Rate
    view.setUint32(28, rate * 2, true); // Byte Rate
    view.setUint16(32, 2, true); // Block Align
    view.setUint16(34, 16, true); // Bits Per Sample
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    // Convert Float32 to 16-bit PCM Signed Integer
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    
    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function updateTimer() {
    const elapsed = Date.now() - recordStartTime;
    const totalSeconds = Math.floor(elapsed / 1000);
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
    const seconds = String(totalSeconds % 60).padStart(2, '0');
    recordTimer.textContent = `${minutes}:${seconds}`;
}

// Wave animations
function drawIdleWave() {
    const canvas = visualizerCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(14, 165, 233, 0.25)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    
    // Draw simple flat wave
    for (let i = 0; i < width; i++) {
        const y = height / 2 + Math.sin(i * 0.03) * 2;
        ctx.lineTo(i, y);
    }
    ctx.stroke();
}

function drawWave() {
    if (!isRecording) return;
    
    animationFrameId = requestAnimationFrame(drawWave);
    
    analyser.getByteFrequencyData(dataArray);
    
    const canvas = visualizerCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 2.5;
    
    // Create subtle gradient for wave
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, '#0ea5e9');
    gradient.addColorStop(0.5, '#6366f1');
    gradient.addColorStop(1, '#a855f7');
    ctx.strokeStyle = gradient;
    
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    
    const sliceWidth = width / dataArray.length;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0; // range 0 to 2
        // amplify slightly for visual effect
        const amp = 25;
        const y = height / 2 + (v - 1) * amp * Math.sin(x * 0.05);
        
        ctx.lineTo(x, y);
        x += sliceWidth;
    }
    
    ctx.lineTo(width, height / 2);
    ctx.stroke();
}

// Upload & Transcribe API call
async function uploadAndTranscribe(audioBlob, filename = 'audio.wav') {
    showLoader('Đang chuyển đổi giọng nói...', 'Mô hình Whisper đang nhận dạng tiếng Nghệ Tĩnh...');
    
    const formData = new FormData();
    formData.append('file', audioBlob, filename);
    
    // Append active model selection
    const selectedModel = asrModelSelect.value;
    formData.append('model_name', selectedModel);

    try {
        const response = await fetch('/api/asr', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            asrOutput.value = data.transcript;
            translateBtn.disabled = data.transcript.trim().length === 0;
            recordStatus.textContent = `Nhận diện xong (Model: ${data.model_used})`;
            
            // Auto translate if transcript is found
            if (data.transcript.trim()) {
                handleTranslate();
            }
        } else {
            const errData = await response.json();
            recordStatus.textContent = 'Lỗi nhận diện giọng nói';
            alert(`Lỗi: ${errData.detail || 'Không thể nhận diện giọng nói'}`);
        }
    } catch (e) {
        console.error('Error during ASR request:', e);
        recordStatus.textContent = 'Lỗi kết nối server';
        alert('Lỗi kết nối tới server. Vui lòng kiểm tra lại backend.');
    } finally {
        hideLoader();
    }
}

// Translation API call
async function handleTranslate() {
    const text = asrOutput.value.trim();
    if (!text) return;

    showLoader('Đang biên dịch phương ngữ...', 'Chạy RAG và truy vấn LLM DeepSeek...');

    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (response.ok) {
            const data = await response.json();
            
            // Display translation text
            translationOutput.innerHTML = data.translated_text;
            
            // Display matched terms
            displayDictionaryTerms(data.matched_terms);
        } else {
            const errData = await response.json();
            alert(`Lỗi dịch: ${errData.detail || 'Không thể dịch'}`);
        }
    } catch (e) {
        console.error('Error during translate request:', e);
        alert('Lỗi kết nối tới server dịch.');
    } finally {
        hideLoader();
    }
}

// Display dictionary terms in list
function displayDictionaryTerms(terms) {
    dictionaryTermsList.innerHTML = '';
    termsCount.textContent = `${terms.length} từ`;

    if (terms.length === 0) {
        dictionaryTermsList.innerHTML = `
            <div class="empty-dictionary">
                <i class="fa-solid fa-book-open"></i>
                <p>Không tìm thấy từ phương ngữ Nghệ Tĩnh đặc thù trong cơ sở dữ liệu từ điển.</p>
            </div>
        `;
        return;
    }

    terms.forEach(item => {
        const card = document.createElement('div');
        card.className = `dict-term-card ${item.type}`;
        
        const badgeText = item.type === 'exact' ? 'Chính xác' : 
                          item.type === 'fuzzy' ? 'Gần giống' : 'Semantic';
        
        card.innerHTML = `
            <div class="dict-term-header">
                <span class="dict-term-name">${item.term}</span>
                <span class="dict-term-type">${badgeText}</span>
            </div>
            <div class="dict-term-meaning">${item.info}</div>
        `;
        dictionaryTermsList.appendChild(card);
    });
}

// Copy to Clipboard utility
function copyToClipboard(text, btnElement) {
    if (!text || text.trim() === '') return;
    
    navigator.clipboard.writeText(text).then(() => {
        const icon = btnElement.querySelector('i');
        icon.className = 'fa-solid fa-check';
        btnElement.style.color = '#10b981';
        btnElement.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        
        setTimeout(() => {
            icon.className = 'fa-regular fa-copy';
            btnElement.style.color = '';
            btnElement.style.borderColor = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}
