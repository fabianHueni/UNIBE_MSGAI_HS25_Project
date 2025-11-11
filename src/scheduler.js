import {sleep} from './utils.js';


/**
 * JobScheduler emits jobs based on predefined patterns.
 * Can be used to simulate different load scenarios like batch processing or on-request per second
 */
export class JobScheduler {
    constructor(promptSource = []) {
        this.promptSource = promptSource;
        this.running = false;
        this._onJob = null; // callback
    }


    onJob(cb) {
        this._onJob = cb;
    }


    /**
     * Start emitting jobs based on the selected pattern
     * TODO: Implement different patterns to simulate
     * TODO: Run different datasets instead of just simple prompts
     * @param patternName
     * @returns {Promise<void>}
     */
    async startPattern(patternName) {
        this.running = true;

        // once per second until user stopp evaluation
        if (patternName === 'once-per-sec') {
            let i = 0;
            while (this.running) {
                this._emit(i++);
                await sleep(1000);
            }
        } else if (patternName === 'ten-per-sec') {
            let i = 0;
            const interval = 100; // ms
            while (this.running) {
                this._emit(i++);
                await sleep(interval);
            }
        } else if (patternName === 'batch-10-every-5s') {
            let i = 0;
            while (this.running) {
                for (let j = 0; j < 10 && this.running; j++) this._emit(i++);
                await sleep(5000);
            }
        } else if (patternName === 'burst') {
            // single burst
            for (let i = 0; i < 50; i++) this._emit(i);
            this.running = false;
        }
    }


    /**
     * Stop emitting jobs
     */
    stop() {
        this.running = false;
    }

    _pickPrompt(id) {
        if (this.promptSource.length === 0) return {prompt: `Hello world ${id}`, groundTruth: `Hello world ${id}`};
        return this.promptSource[id % this.promptSource.length];
    }


    _emit(id) {
        if (this._onJob) {
            const p = this._pickPrompt(id);
            const job = {id: `job-${Date.now()}-${id}`, prompt: p.prompt, groundTruth: p.groundTruth};
            this._onJob(job);
        }
    }
}