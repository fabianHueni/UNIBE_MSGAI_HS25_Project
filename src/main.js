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
const cloudInferenceService = new CloudService({apiKey: '', model: document.getElementById('cloudModel').value});
const evaluator = new Evaluator();


const requestManager = new RequestManager({
    deviceService: onDeviceInferenceService, cloudService: cloudInferenceService, evaluator, logger: evt => {
        logTo(logEl, `${evt.job.id} -> ${evt.route} | latency=${evt.latency}ms | exact=${evt.evalRes.exact} f1=${evt.evalRes.f1.toFixed(2)}`);
        updateStats();
    }
});


// instantiate the job scheduler with some mock prompts TODO: replace with real prompts
const scheduler = new JobScheduler([
    {prompt: 'Translate to German: Hello world', groundTruth: 'Hallo Welt'},
    {
        prompt: 'What is 3*6?',
        groundTruth: '18'
    },
    {prompt: 'Answer: What is 2+2?', groundTruth: '4'},
    {prompt: 'What is the capital of switzerland?', groundTruth: 'Bern'}
]);


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
document.getElementById('apiKey').addEventListener('input', (e) =>
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


async function loadDeviceModel() {
    deviceStatusEl.textContent = 'Loading...';
    try {
        await onDeviceInferenceService.load((s) => deviceStatusEl.textContent = s);
        deviceStatusEl.textContent = 'Ready';
    } catch (e) {
        deviceStatusEl.textContent = `Error: ${e.message}`;
    }
}


function updateStats() {
    const s = requestManager.stats;
    statsEl.innerHTML = `<pre>Processed: ${s.count}\nCloud: ${s.cloud}\nDevice: ${s.device}\nAvg latency (ms): ${s.count ? (s.totalLatencyMs / s.count).toFixed(1) : 0}\nRecent evaluations: ${Math.min(10, s.evaluations.length)}</pre>`;
}