import {JobScheduler} from './scheduler.js';
import {RequestManager} from './requestManager.js';
import {OnDeviceService} from './services/onDeviceService.js';
import {CloudService} from './services/cloudService.js';
import {Evaluator} from './evaluator.js';
import {logTo} from './utils.js';


// get references to html elements
const logEl = document.getElementById('log');
const statsEl = document.getElementById('stats');
const deviceStatusEl = document.getElementById('deviceStatus');


// instantiate services and components
const onDeviceInferenceService = new OnDeviceService({modelName: document.getElementById('deviceModel').value});
const cloudInferenceService = new CloudService({apiKey: document.getElementById('cloudApiKey').value, model: document.getElementById('cloudModel').value});
const evaluator = new Evaluator();


const requestManager = new RequestManager({
    deviceService: onDeviceInferenceService, cloudService: cloudInferenceService, evaluator, logger: evt => {
        logTo(logEl, `${evt.route} | latency=${evt.latency}ms | exact=${evt.evalRes.exact} | question="${evt.job.prompt.substring(0, 30)}..."`);
        updateStats();
    }
});


// instantiate the job scheduler with some mock prompts TODO: replace with real prompts
const scheduler = new JobScheduler('boolq_validation');


scheduler.onJob(async (job) => {
    await requestManager.handle(job);
});


// add event listeners for configuration inputs
document.getElementById('deviceModel').addEventListener('change', (e) =>
    onDeviceInferenceService.updateConfig({modelName: e.target.value})
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

document.getElementById('downloadStats').addEventListener('click', () => {
    downloadStats();
});
document.getElementById('loadDeviceModelBtn').addEventListener('click', () => {
    loadDeviceModel();
});


async function loadDeviceModel() {
    deviceStatusEl.textContent = 'Loading...';
    document.getElementById('loadDeviceModelBtn').disabled = true;
    document.getElementById('loadDeviceModelBtn').textContent = 'Loading Model...';
    const loadingBar = document.getElementById('deviceLoadingBar');
    const loadingText = document.getElementById('deviceLoadingText');
    loadingBar.style.width = '0%';
    loadingText.textContent = '';
    function updateModelLoadingUI(progress) {
        if (progress && progress.loaded && progress.total) {
            const percent = ((progress.loaded / progress.total) * 100).toFixed(1);
            loadingBar.style.width = percent + '%';
            loadingText.textContent = `Loading: ${percent}% (${progress.loaded}/${progress.total} bytes)`;
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
        deviceStatusEl.textContent = `Error: ${e.message}`;
        loadingText.textContent = 'Error loading model.';
        document.getElementById('loadDeviceModelBtn').disabled = false;
    }
}

function downloadStats() {
    const s = requestManager.stats;
    // add average latency to stats for device and cloud
    s.avgLatencyMs = s.count ? (s.totalLatencyMs / s.count) : 0;
    s.avgDeviceLatencyMs = s.device ? (s.evaluations.filter(e => e.route === 'device').reduce((a, b) => a + b.latency, 0) / s.device) : 0;
    s.avgCloudLatencyMs = s.cloud ? (s.evaluations.filter(e => e.route === 'cloud').reduce((a, b) => a + b.latency, 0) / s.cloud) : 0;

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(s, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", "stats.json");
    dlAnchorElem.click();
}

/**
 * Update the statistics display in the UI based on the request manager's stats
 */
function updateStats() {
    const s = requestManager.stats;

    statsEl.innerHTML = `
        <div style="display: flex; justify-content: space-between;">
            <div>
                <h3>General Stats</h3>
                <pre>
Processed: ${s.count}
Avg latency (ms): ${s.count ? (s.totalLatencyMs / s.count).toFixed(1) : 0}
Recent evaluations: ${Math.min(10, s.evaluations.length)}
                </pre>
            </div>
            <div>
                <h3>Cloud Stats</h3>
                <pre>
Requests: ${s.cloud}
Avg latency (ms): ${s.cloud ? (s.evaluations.filter(e => e.route === 'cloud').reduce((a, b) => a + b.latency, 0) / s.cloud).toFixed(1) : 0}
                </pre>
            </div>
            <div>
                <h3>On-Device Stats</h3>
                <pre>
Requests: ${s.device}
Avg latency (ms): ${s.cloud ? (s.evaluations.filter(e => e.route === 'device').reduce((a, b) => a + b.latency, 0) / s.cloud).toFixed(1) : 0}
                </pre>
            </div>
        </div>`;
}