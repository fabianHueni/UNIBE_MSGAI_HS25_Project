import {sleep} from './utils.js';

/**
 * JobScheduler emits jobs based on predefined patterns.
 * Can be used to simulate different load scenarios like batch processing or on-request per second
 */
export class JobScheduler {
    constructor(datasetName = 'boolq_validation') {
        // TODO implement dataset loading based on configuration parameter
        this.running = false;
        this._dataset = null;
        this._onJob = null; // callback
        this._datasetName = datasetName
        this._interArrivalTimeLambda = 2; // rate parameter for interarrival time generation in seconds

        this._loadDataset(this._datasetName);
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
            while (this._dataset.length > 0 && this.running) {
                const item = this._dataset.shift(); //shift instead of pop for FIFO
                this._emit(item);
                await sleep(1000);
            }
        } else if (patternName === 'every-ten-sec') {
            let i = 0;
            while (this._dataset.length > 0 && this.running) {
                const item = this._dataset.shift();
                this._emit(item);
                await sleep(10000);
            }
        } else if (patternName === 'exponential-arrival') {
            let i = 0;
            while (this._dataset.length > 0 && this.running) {
                const item = this._dataset.shift();
                this._emit(item);
                const timeToNextArrival = this._generateInterarrivalTime(this._interArrivalTimeLambda);
                await sleep(timeToNextArrival);
            }
        }
    }


    /**
     * Stop emitting jobs
     */
    stop() {
        this.running = false;
    }

    /**
     * Reload the dataset (useful for running multiple experiments)
     */
    async reloadDataset() {
        return new Promise((resolve, reject) => {
            this._loadDataset(this._datasetName);
            // Wait a bit for the fetch to complete
            const checkLoaded = setInterval(() => {
                if (this._dataset && this._dataset.length > 0) {
                    clearInterval(checkLoaded);
                    resolve();
                }
            }, 100);
            // Timeout after 10 seconds
            setTimeout(() => {
                clearInterval(checkLoaded);
                reject(new Error('Dataset loading timeout'));
            }, 10000);
        });
    }



    /**
     * Emit a job with the item from the dataset to process
     *
     * @param item - The dataset item containing prompt and ground truth
     * @private
     */
    _emit(item) {
        if (this._onJob) {
            const job = {
                prompt: item.prompt, 
                groundTruth: item.groundTruth,
                timestamps: {
                    jobStart: performance.now(),
                    inferenceStart: null,
                    inferenceEnd: null
                }
            };
            this._onJob(job);
        }
    }

    /**
     * Load the dataset from CSV file based on the given name
     *
     * @param name - Name of the csv dataset to load without file extension
     * @private
     */
    _loadDataset(name) {
        const path = `./dataset/${name}.csv`;

        fetch(path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Dataset file not found: ${path}`);
                }
                return response.text();
            })
            .then(data => {
                this._dataset = data.split('\n').slice(1).map(line => {
                    // Properly parse CSV with quoted fields
                    const regex = /("(?:[^"]|"")*"|[^,]*)(,|$)/g;
                    const fields = [];
                    let match;
                    while ((match = regex.exec(line)) !== null) {
                        if (match[2] === '' && match.index === line.length) break;
                        let field = match[1];
                        // Remove surrounding quotes and unescape double quotes
                        if (field.startsWith('"') && field.endsWith('"')) {
                            field = field.slice(1, -1).replace(/""/g, '"');
                        }
                        fields.push(field);
                    }
                    const [question, answer, context] = fields;
                    
                    // More explicit prompt to get concise answers
                    const full_prompt = `Question: ${question}
                                        Context: ${context}
                                        Instructions: Answer with ONLY the word "true" or "false". Do not provide any explanation or additional text.
                                        Answer:`;
                                        
                    return {prompt: full_prompt, groundTruth: answer};
                });
                console.log(`✅ Dataset '${name}' loaded with ${this._dataset.length} items.`);
            })
            .catch(error => {
                console.error(error);
            });
    }


    /**
     * Generate interarrival time based on exponential distribution
     *
     * @param lambda - rate parameter (requests per second)
     * @returns {number} - interarrival time in milliseconds
     * @private
     */
    _generateInterarrivalTime(lambda) {
        const u = Math.random(); // uniform random number between 0 and 1
        return -Math.log(u) / lambda * 1000; // convert to milliseconds
    }
}