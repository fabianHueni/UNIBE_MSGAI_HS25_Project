import {JobScheduler} from './scheduler.js';
import {RequestManager} from './requestManager.js';
import {OnDeviceService} from './services/onDeviceService.js';
import {CloudService} from './services/cloudService.js';
import {Evaluator} from './evaluator.js';
import {logTo} from './utils.js';


// get references to html elements
const logEl = document.getElementById('log-table-body');
const statsEl = document.getElementById('stats');
const deviceStatusEl = document.getElementById('deviceStatus');


// instantiate services and components
const onDeviceInferenceService = new OnDeviceService({modelName: document.getElementById('deviceModel').value});
const cloudInferenceService = new CloudService({
    apiKey: document.getElementById('cloudApiKey').value,
    model: document.getElementById('cloudModel').value
});
const evaluator = new Evaluator();


const requestManager = new RequestManager({
    deviceService: onDeviceInferenceService, cloudService: cloudInferenceService, evaluator, logger: evt => {
        logTo(logEl, evt);
        updateStats();
    }
});


// instantiate the job scheduler with some mock prompts TODO: replace with real prompts
const scheduler = new JobScheduler('boolq_validation');


scheduler.onJob(async (job) => {
    await requestManager.pushJob(job);
});


// add event listeners for configuration inputs
document.getElementById('deviceModel').addEventListener('change', (e) => {
        try {
            const {modelName, quantization} = JSON.parse(e.target.value);
            onDeviceInferenceService.updateConfig({modelName, quantization});
        } catch (error) {
            console.error('Invalid JSON in on device model selection:', e.target.value);
        }
    }
);
document.getElementById('cloudModel').addEventListener('change', (e) =>
    cloudInferenceService.updateConfig({model: e.target.value})
);
document.getElementById('cloudApiKey').addEventListener('input', (e) =>
    cloudInferenceService.updateConfig({apiKey: e.target.value})
);

// add event listener for run button
document.getElementById('startBtn').addEventListener('click', async () => {

    // toggle start and stop buttons
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;

    // get configuration values from UI
    const pattern = document.getElementById('patternSelect').value;
    const routeStrategy = document.getElementById('routeStrategy').value;
    const cloudProb = parseFloat(document.getElementById('cloudProb').value);

    // update request manager routing strategy
    requestManager.updateRouting({routeStrategy, cloudProb});


    // TODO Adjust that the model is loaded with a button such that user can see loading status and trigger loading before starting
    // starting is only available when model is loaded
    if (routeStrategy !== 'always_cloud' && !onDeviceInferenceService.isReady()) {
        await loadDeviceModel();
    }

    // start the job scheduler with the selected pattern
    scheduler.startPattern(pattern);
});


document.getElementById('stopBtn').addEventListener('click', () => {
    scheduler.stop();
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
});

document.getElementById('downloadStatsJson').addEventListener('click', () => {
    downloadStatsAsJson();
});
document.getElementById('downloadStatsCsv').addEventListener('click', () => {
    downloadStatsAsCSV();
});
document.getElementById('loadDeviceModelBtn').addEventListener('click', () => {
    loadDeviceModel();
});

document.getElementById('interArrivalTimeLambda').addEventListener('input', (event) => {
    const newValue = parseFloat(event.target.value);
    if (!isNaN(newValue) && newValue > 0) {
        scheduler._interArrivalTimeLambda = newValue;
    }
});

let currentExperiment = null;
let experimentJobCount = 0;
let experimentTargetJobs = 0;
let isExperimentRunning = false;

document.getElementById('start1000Btn').addEventListener('click', async () => {
    const TARGET_JOBS = 1000;

    // Get configuration from UI
    const pattern = document.getElementById('patternSelect').value;
    const routeStrategy = document.getElementById('routeStrategy').value;
    const cloudProb = parseFloat(document.getElementById('cloudProb').value);
    const deviceModel = document.getElementById('deviceModel').value;
    const cloudModel = document.getElementById('cloudModel').value;

    // Validate
    if (routeStrategy !== 'always_cloud' && !onDeviceInferenceService.isReady()) {
        alert('Please load the on-device model first, or select "Always Cloud" strategy.');
        return;
    }

    if (routeStrategy !== 'always_device') {
        const apiKey = document.getElementById('cloudApiKey').value;
        if (!apiKey || apiKey.trim() === '') {
            alert('Please enter a Cloud API Key, or select "Always Device" strategy.');
            return;
        }
    }

    // Store experiment config
    currentExperiment = {
        deviceModel,
        cloudModel,
        routeStrategy,
        pattern,
        startTime: new Date().toISOString()
    };

    experimentJobCount = 0;
    experimentTargetJobs = TARGET_JOBS;
    isExperimentRunning = true;

    // Reset stats
    requestManager.stats.count = 0;
    requestManager.stats.cloud = 0;
    requestManager.stats.device = 0;
    requestManager.stats.totalLatencyMs = 0;
    requestManager.stats.results = [];

    // Update UI
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('start1000Btn').disabled = true;
    document.getElementById('start1000Btn').textContent = `Running`;

    // Update routing
    requestManager.updateRouting({routeStrategy, cloudProb});

    console.log(`🚀 Starting experiment: ${TARGET_JOBS} jobs`);
    console.log(`📊 Config: Strategy=${routeStrategy}, Pattern=${pattern}`);

    try {
        // Reload dataset to ensure we have enough items
        await scheduler.reloadDataset();

        // Start the limited run
        await scheduler.startPattern(pattern, TARGET_JOBS);

    } catch (error) {
        console.error('❌ Experiment error:', error);
        alert(`Experiment failed: ${error.message}`);
    }

    // Finish experiment
    finishExperiment();
});

function finishExperiment() {
    if (!isExperimentRunning) return;

    isExperimentRunning = false;
    console.log('✅ Experiment complete!');

    // Stop the scheduler
    scheduler.stop();

    // Update UI
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('start1000Btn').disabled = false;
    document.getElementById('start1000Btn').textContent = 'Start 1000';

    // Auto-download results
    setTimeout(() => {
        downloadExperimentResults();
    }, 500);
}

function downloadExperimentResults() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

    // Build model name for filename
    let modelName = '';
    if (currentExperiment.routeStrategy === 'always_cloud') {
        modelName = currentExperiment.cloudModel.replace(/[^a-zA-Z0-9]/g, '-');
    } else if (currentExperiment.routeStrategy === 'always_device') {
        modelName = currentExperiment.deviceModel.split('/').pop().replace(/[^a-zA-Z0-9]/g, '-');
    } else {
        const device = currentExperiment.deviceModel.split('/').pop().replace(/[^a-zA-Z0-9]/g, '-');
        const cloud = currentExperiment.cloudModel.replace(/[^a-zA-Z0-9]/g, '-');
        modelName = `${device}_${cloud}`;
    }

    const filename = `experiment_${modelName}_${currentExperiment.routeStrategy}_${currentExperiment.pattern}_${timestamp}`;

    // Build stats object with experiment info
    const stats = {
        experiment: {
            ...currentExperiment,
            endTime: new Date().toISOString(),
            completedJobs: requestManager.stats.count
        },
        stats: requestManager.stats
    };

    // Download CSV
    const csvContent = buildExperimentCSV(stats);
    const csvBlob = new Blob([csvContent], {type: 'text/csv'});
    const csvUrl = URL.createObjectURL(csvBlob);
    const csvLink = document.createElement('a');
    csvLink.href = csvUrl;
    csvLink.download = `${filename}.csv`;
    csvLink.click();
    URL.revokeObjectURL(csvUrl);

    console.log(`📥 Downloaded: ${filename}.csv`);
}

function buildExperimentCSV(stats) {
    const lines = [];

    // Header
    lines.push('job_id,route,latency_ms,total_latency_ms,queueing_time_ms,inference_time_ms,exact_match,f1_score,ground_truth,answer');

    // Data rows
    stats.stats.results.forEach((result, index) => {
        const row = [
            index,
            result.route || '',
            (result.latency || 0).toFixed(2),
            (result.totalLatency || 0).toFixed(2),
            (result.queueingTime || 0).toFixed(2),
            (result.inferenceTime || 0).toFixed(2),
            result.evalRes?.exactMatch || false,
            (result.evalRes?.f1WordLevel || 0).toFixed(4),
            `"${(result.job?.groundTruth || '').replace(/"/g, '""')}"`,
            `"${(result.text?.answer || '').replace(/"/g, '""')}"`
        ];
        lines.push(row.join(','));
    });

    // Calculate averages
    const results = stats.stats.results;
    const count = results.length;

    if (count > 0) {
        const avgLatency = results.reduce((sum, r) => sum + (r.latency || 0), 0) / count;
        const avgTotalLatency = results.reduce((sum, r) => sum + (r.totalLatency || 0), 0) / count;
        const avgQueueingTime = results.reduce((sum, r) => sum + (r.queueingTime || 0), 0) / count;
        const avgInferenceTime = results.reduce((sum, r) => sum + (r.inferenceTime || 0), 0) / count;
        const accuracy = results.filter(r => r.evalRes?.exactMatch).length / count * 100;

        // Add empty line and summary
        lines.push('');
        lines.push('# Summary');
        lines.push(`total_requests,${count}`);
        lines.push(`accuracy_percent,${accuracy.toFixed(2)}`);
        lines.push(`avg_latency_ms,${avgLatency.toFixed(2)}`);
        lines.push(`avg_total_latency_ms,${avgTotalLatency.toFixed(2)}`);
        lines.push(`avg_queueing_time_ms,${avgQueueingTime.toFixed(2)}`);
        lines.push(`avg_inference_time_ms,${avgInferenceTime.toFixed(2)}`);
    }

    return lines.join('\n');
}


async function loadDeviceModel() {
    deviceStatusEl.textContent = 'Loading...';
    document.getElementById('loadDeviceModelBtn').disabled = true;
    document.getElementById('loadDeviceModelBtn').textContent = 'Loading Model...';
    const loadingBar = document.getElementById('deviceLoadingBar');
    const loadingText = document.getElementById('deviceLoadingText');
    loadingBar.style.width = '0%';
    loadingText.textContent = '';

    function updateModelLoadingUI(progress) {
        console.log('Model loading progress:', progress);
        if (progress && progress.loaded && progress.total) {
            const percent = ((progress.loaded / progress.total) * 100).toFixed(1);
            loadingBar.style.width = percent + '%';
            loadingText.textContent = `Loading: ${percent}% (${(progress.loaded / (1024 ** 3)).toFixed(2)} GB / ${(progress.total / (1024 ** 3)).toFixed(2)} GB)`;
        } else if (progress && progress.status) {
            loadingText.textContent = progress.status;
        } else if (typeof progress === 'string') {
            loadingText.textContent = progress;
        }
    }

    try {
        await onDeviceInferenceService.load(updateModelLoadingUI);
        deviceStatusEl.textContent = 'Model Ready';
        loadingBar.style.width = '100%';
        loadingText.textContent = 'Model loaded.';
    } catch (e) {
        console.error('❌ Error loading on-device model:', e);
        deviceStatusEl.textContent = `Error: ${e.message}`;
        loadingText.textContent = 'Error loading model.';
        document.getElementById('loadDeviceModelBtn').disabled = false;
        document.getElementById('loadDeviceModelBtn').textContent = 'Load Model';
    }
}

function downloadStatsAsJson() {
    const s = requestManager.stats;
    // add average latency to stats for device and cloud
    s.avgLatencyMs = s.count ? (s.totalLatencyMs / s.count) : 0;
    s.avgDeviceLatencyMs = s.device ? (s.results.filter(e => e.route === 'device').reduce((a, b) => a + b.latency, 0) / s.device) : 0;
    s.avgCloudLatencyMs = s.cloud ? (s.results.filter(e => e.route === 'cloud').reduce((a, b) => a + b.latency, 0) / s.cloud) : 0;

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(s, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", "stats.json");
    dlAnchorElem.click();
}

function downloadStatsAsCSV() {
    const s = requestManager.stats;

    const flattened_evals = s.results.map(evaluation => ({
            route: evaluation.route,
            latency: evaluation.latency,
            totalLatency: evaluation.totalLatency || 0,
            queueingTime: evaluation.queueingTime || 0,
            inferenceTime: evaluation.inferenceTime || 0,
            prompt: evaluation.job.prompt,

            // job details
            groundTruth: evaluation.job.groundTruth,
            answer: evaluation.text.answer,

            // evaluation results
            exactMatch: evaluation.evalRes.exactMatch,
            f1: evaluation.evalRes.f1WordLevel,
            tokensPerSecond: evaluation.evalRes.tokensPerSecond,
            totalTokens: evaluation.evalRes.totalTokens,

            // further stats
            input_tokens: evaluation.text.stats.input_tokens,
            output_tokens: evaluation.text.stats.output_tokens,
        })
    );

    // Convert stats to CSV format
    const headers = Object.keys(flattened_evals[0] || {}).join(',');
    const rows = flattened_evals.map(evaluation =>
        Object.values(evaluation).map(value => `"${value}"`).join(',')
    );
    const csvContent = [headers, ...rows].join('\n');

    const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csvContent);
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", "stats.csv");
    dlAnchorElem.click();
}

/**
 * Update the statistics display in the UI based on the request manager's stats
 */
function updateStats() {
    const s = requestManager.stats;

    // Calculate average timing metrics
    const avgTotalLatency = s.count ? (s.results.reduce((a, b) => a + (b.totalLatency || 0), 0) / s.count) : 0;
    const avgQueueingTime = s.count ? (s.results.reduce((a, b) => a + (b.queueingTime || 0), 0) / s.count) : 0;
    const avgInferenceTime = s.count ? (s.results.reduce((a, b) => a + (b.inferenceTime || 0), 0) / s.count) : 0;

    const cloudResults = s.results.filter(e => e.route === 'cloud');
    const deviceResults = s.results.filter(e => e.route === 'device');

    const avgCloudTotal = s.cloud ? (cloudResults.reduce((a, b) => a + (b.totalLatency || 0), 0) / s.cloud) : 0;
    const avgCloudQueue = s.cloud ? (cloudResults.reduce((a, b) => a + (b.queueingTime || 0), 0) / s.cloud) : 0;
    const avgCloudInference = s.cloud ? (cloudResults.reduce((a, b) => a + (b.inferenceTime || 0), 0) / s.cloud) : 0;

    const avgDeviceTotal = s.device ? (deviceResults.reduce((a, b) => a + (b.totalLatency || 0), 0) / s.device) : 0;
    const avgDeviceQueue = s.device ? (deviceResults.reduce((a, b) => a + (b.queueingTime || 0), 0) / s.device) : 0;
    const avgDeviceInference = s.device ? (deviceResults.reduce((a, b) => a + (b.inferenceTime || 0), 0) / s.device) : 0;

    statsEl.innerHTML = `
        <div style="display: flex; justify-content: space-between;">
            <div>
                <h3>General Stats</h3>
                <pre>
Processed: ${s.count}
Avg total latency: ${avgTotalLatency.toFixed(1)}ms
Avg queueing time: ${avgQueueingTime.toFixed(1)}ms
Avg inference time: ${avgInferenceTime.toFixed(1)}ms
Avg correct: ${s.count ? (s.results.reduce((a, b) => a + (b.evalRes.exactMatch ? 1 : 0), 0) / s.count * 100).toFixed(1) : 0}%
Recent evaluations: ${Math.min(10, s.results.length)}
                </pre>
            </div>
            <div>
                <h3>Cloud Stats</h3>
                <pre>
Requests: ${s.cloud}
Avg total latency: ${avgCloudTotal.toFixed(1)}ms
Avg queueing time: ${avgCloudQueue.toFixed(1)}ms
Avg inference time: ${avgCloudInference.toFixed(1)}ms
Avg correct: ${s.cloud ? (cloudResults.reduce((a, b) => a + (b.evalRes.exactMatch ? 1 : 0), 0) / s.cloud * 100).toFixed(1) : 0}%
               
                </pre>
            </div>
            <div>
                <h3>On-Device Stats</h3>
                <pre>
Requests: ${s.device}
Avg total latency: ${avgDeviceTotal.toFixed(1)}ms
Avg queueing time: ${avgDeviceQueue.toFixed(1)}ms
Avg inference time: ${avgDeviceInference.toFixed(1)}ms
Avg correct: ${s.device ? (deviceResults.reduce((a, b) => a + (b.evalRes.exactMatch ? 1 : 0), 0) / s.device * 100).toFixed(1) : 0}%

                </pre>
            </div>
        </div>`;
}