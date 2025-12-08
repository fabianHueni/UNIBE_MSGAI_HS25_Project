import {measureAsync, sleep} from './utils.js';


/**
 * RequestManager routes inference requests to on-device or cloud services based on a routing strategy and configurations.
 * The manager does orchestrate the inference requests, collects statistics, evaluates the results and returns the final statistic.
 *
 * We provide different routing strategies:
 * - always_cloud: all requests go to cloud
 * - always_device: all requests go to device
 * - probabilistic: each request goes to cloud with a defined probability
 * - roundrobin: requests alternate between cloud and device
 * - jseq: routes to the server with the shortest expected queue time (Join the Shortest Expected Queue)
 *
 *
 */
export class RequestManager {
    constructor({
                    deviceService,
                    cloudService,
                    evaluator,
                    logger = null,
                    routeStrategy = 'roundrobin',
                    cloudProb = 0.5,
                    devicePerfModel = {slope: 0, intercept: 0},
                    cloudPerfModel = {slope: 0, intercept: 0}
                } = {}) {

        /**
         * On-device inference service
         */
        this.device = deviceService;

        /**
         * Cloud inference service
         */
        this.cloud = cloudService;

        /**
         * Evaluator instance for evaluating inference results
         */
        this.evaluator = evaluator;

        /**
         * Optional logger callback function
         * @type {null}
         */
        this.logger = logger;

        /**
         * Routing strategy (always_cloud, always_device, probabilistic, roundrobin)
         * @type {string}
         */
        this.routeStrategy = routeStrategy;

        /**
         * Probability of routing to cloud when using 'probabilistic' strategy
         * @type {number}
         */
        this.cloudProb = cloudProb;

        /**
         * Performance model for the device {slope, intercept}
         * @type {{slope: number, intercept: number}}
         */
        this.devicePerfModel = devicePerfModel;

        /**
         * Performance model for the cloud {slope, intercept}
         * @type {{slope: number, intercept: number}}
         */
        this.cloudPerfModel = cloudPerfModel;

        /**
         * Tracks when the device will be free
         * @type {number}
         */
        this.deviceFinishTime = 0;

        /**
         * Tracks when the cloud will be free
         * @type {number}
         */
        this.cloudFinishTime = 0;

        /**
         * Internal round robin counter (even = cloud, odd = device)
         * @type {number}
         * @private
         */
        this._rrCounter = 0;

        /**
         * Statistics about routing and evaluations of this job run
         * @type {{cloud: number, evaluations: *[], count: number, device: number, totalLatencyMs: number}}
         */
        this.stats = {count: 0, cloud: 0, device: 0, totalLatencyMs: 0, results: []};

        /**
         * Cloud job queue
         * @type {*[]}
         */
        this.cloud_queue = [];

        /**
         * Device job queue
         * @type {*[]}
         */
        this.device_queue = [];

        // start processing jobs from the queues
        this.runOnDeviceJobsFromQueue();
        this.runCloudJobsFromQueue();
    }

    /**
     * Push a job to the appropriate queue based on routing strategy.
     *
     * @param job - The job to be processed
     */
    pushJob(job) {
        // get routing strategy and inference service
        const route = this._choose(job);
        console.log(`Device Queue Length: ${this.device_queue.length}, \nCloud Queue Length: ${this.cloud_queue.length}`);

        if (route === 'cloud') {
            this.cloud_queue.push(job);
        } else {
            this.device_queue.push(job);
        }
    }


    /**
     * Update routing configuration
     *
     * @param routeStrategy - New routing strategy
     * @param cloudProb - New cloud probability for 'probabilistic' strategy
     * @param devicePerfModel
     * @param cloudPerfModel
     */
    updateRouting({routeStrategy, cloudProb, devicePerfModel, cloudPerfModel}) {
        if (routeStrategy) this.routeStrategy = routeStrategy;
        if (cloudProb !== undefined) this.cloudProb = cloudProb;
        if (devicePerfModel) this.devicePerfModel = devicePerfModel;
        if (cloudPerfModel) this.cloudPerfModel = cloudPerfModel;
    }


    /**
     * Handle device jobs by routing it to the appropriate service, as long as there are jobs in the queue.
     *
     * @returns {Promise<void>}
     */
    async runOnDeviceJobsFromQueue() {
        while (true) {
            if (this.device_queue.length > 0) {
                const job = this._getNextJobFromQueue(this.device_queue, 'fifo');
                const service = this.device;
                const route = 'device';

                // run the job and await until compteted
                await this._runJob(job, route, service);
            }

            // sleep for 10ms to not run into memory leak
            await sleep(10);
        }
    }

    /**
     * Handle cloud jobs by routing it to the appropriate service, as long as there are jobs in the queue.
     *
     * @returns {Promise<void>}
     */
    async runCloudJobsFromQueue() {
        while (true) {
            if (this.cloud_queue.length > 0) {
                const job = this._getNextJobFromQueue(this.cloud_queue, 'fifo');
                const service = this.cloud;
                const route = 'cloud';

                // run the job and await until it completes
                await this._runJob(job, route, service);
            }

            // sleep for 10ms to not run into memory leak
            await sleep(10);
        }
    }


    /**
     * Run the given job on the specified service and record statistics.
     *
     * @param job - The job object containing prompt and ground truth
     * @param route - The selected route ('cloud' or 'device')
     * @param service - The inference service to use
     * @returns {Promise<void>}
     * @private
     */
    async _runJob(job, route, service) {
        let full_prompt = job.prompt; // ensure string input

        // this is a little workaround to disable the thinking mode in qwen models
        if (service.getModelName().toLowerCase().includes("qwen3".toLowerCase())) {
            full_prompt = full_prompt; // + "/no_think";
            console.log("ℹ️ \"/no_think\" was added to the prompt to avoid thinking")
        }

        let response, latencyMs, cleanedResponse; // response is object with .answer and .stats
        try {
            // Mark inference start
            job.timestamps.inferenceStart = Date.now();

            const {res, ms} = await measureAsync(() => service.infer(full_prompt));
            response = res;
            latencyMs = ms;

            // Mark inference end
            job.timestamps.inferenceEnd = Date.now();
        } catch (err) {
            response = `__error__:${err.message}`;
            latencyMs = -1;
            job.timestamps.inferenceEnd = Date.now();
        }

        // Calculate timing metrics
        const queueingTime = job.timestamps.inferenceStart - job.timestamps.jobStart;
        const inferenceTime = job.timestamps.inferenceEnd - job.timestamps.inferenceStart;
        const totalLatency = job.timestamps.inferenceEnd - job.timestamps.jobStart;

        // update finish time for JSEQ
        if (route === 'device') {
            this.deviceFinishTime = job.timestamps.inferenceEnd;
        } else {
            this.cloudFinishTime = job.timestamps.inferenceEnd;
        }

        // clean response
        cleanedResponse = this._cleanResponse(response.answer);

        // evaluate result and store results
        const evalRes = this.evaluator.evaluate(cleanedResponse, job.groundTruth, latencyMs);
        this._record(route, latencyMs, evalRes, job, cleanedResponse, {queueingTime, inferenceTime, totalLatency});

        if (this.logger) {
            try {
                this.logger({job, route, latency: latencyMs, evalRes, response: cleanedResponse.answer, queueingTime, inferenceTime, totalLatency});
            } catch (error) {
                console.error("Logger encountered an error:", error);
            }
        }

        // logging on console
        console.log("🎯 Models Answer: " + response.answer +
            "; \nCleaned Answer: " + cleanedResponse.answer +
            '; \nGround Truth: ' + job.groundTruth +
            "; \nInference Time: " + inferenceTime.toFixed(2) + "ms" +
            "; \nQueueing Time: " + queueingTime.toFixed(2) + "ms" +
            "; \nTotal Latency: " + totalLatency.toFixed(2) + "ms");
    }

    _getNextJobFromQueue(queue, policy) {
        // currently only FIFO is implemented
        return queue.shift();
    }

    /**
     * Choose a route based on the configured strategy.
     *
     * @returns {string} 'cloud' or 'device'
     * @private
     */
    _choose(job) {
        switch (this.routeStrategy) {
            case 'always_cloud':
                return 'cloud';
            case 'always_device':
                return 'device';
            case 'probabilistic':
                return Math.random() < this.cloudProb ? 'cloud' : 'device';
            case 'roundrobin':
                this._rrCounter++;
                return this._rrCounter % 2 === 0 ? 'cloud' : 'device';
            case 'jseq':
                return this._decideJSEQ(job);
            default:
                return 'device';
        }
    }

    /**
     * Decide route based on Join the Shortest Expected Queue (JSEQ) policy.
     *
     * @param job
     * @returns {string}
     * @private
     */
    _decideJSEQ(job) {
        const now = Date.now();
        const input_size = job.prompt.length;

        // Predict inference time for both services
        const device_predicted_inference_time = this.devicePerfModel.intercept + this.devicePerfModel.slope * input_size;
        const cloud_predicted_inference_time = this.cloudPerfModel.intercept + this.cloudPerfModel.slope * input_size;

        // Calculate when each server will be free
        const device_free_at = Math.max(now, this.deviceFinishTime);
        const cloud_free_at = Math.max(now, this.cloudFinishTime);

        // Calculate expected finish time for the new job on both servers
        const device_expected_finish_time = device_free_at + device_predicted_inference_time;
        const cloud_expected_finish_time = cloud_free_at + cloud_predicted_inference_time;

        // Choose the server with the earlier expected finish time
        if (device_expected_finish_time <= cloud_expected_finish_time) {
            return 'device';
        } else {
            return 'cloud';
        }
    }

    /**
     * Record statistics for the given job evaluation.
     * Increases counters for total requests and cloud/device usage.
     * Updates the total latency.
     *
     * @param route - The route taken ('cloud' or 'device')
     * @param latency - Latency in milliseconds
     * @param evalRes - Evaluation result object
     * @param job - The job object
     * @param text - The inference result text
     * @param timingMetrics - Object containing queueingTime, inferenceTime, and totalLatency
     * @private
     */
    _record(route, latency, evalRes, job, text, timingMetrics) {
        this.stats.count++;
        if (route === 'cloud') this.stats.cloud++; else this.stats.device++;
        if (latency > 0) this.stats.totalLatencyMs += latency;
        this.stats.results.push({
            job: job, 
            route, 
            latency, 
            evalRes, 
            text,
            queueingTime: timingMetrics.queueingTime,
            inferenceTime: timingMetrics.inferenceTime,
            totalLatency: timingMetrics.totalLatency,
            timestamps: job.timestamps
        });
    }

    /**
     * Remove reasoning/thinking tokens and sections from a model's response.
     * Supports various formats (XML tags, special tokens, markdown, etc.) used by reasoning models.
     * Returns a cleaned response object with only the final answer for evaluation.
     *
     * @param response - The uncleaned response object (may include reasoning/thinking sections)
     * @return {object|string} - Cleaned response object or original error string
     * @private
     */
    _cleanResponse(response) {
        // If response is an error string, return as-is
        if (typeof response === 'string' && response.startsWith('__error__')) {
            return response;
        }

        // Clone the response object to avoid mutating the original
        const cleanedResponse = { ...response };
        
        if (!cleanedResponse.answer || typeof cleanedResponse.answer !== 'string') {
            return cleanedResponse;
        }

        let cleanedAnswer = cleanedResponse.answer;

        // Define patterns for thinking tokens (common formats)
        const thinkingPatterns = [
            // XML-style tags
            /<think>[\s\S]*?<\/think>/gi,
            /<thinking>[\s\S]*?<\/thinking>/gi,
            /<reasoning>[\s\S]*?<\/reasoning>/gi,
            /<thought>[\s\S]*?<\/thought>/gi,
            
            // Special tokens
            /<\|startofthinking\|>[\s\S]*?<\|endofthinking\|>/gi,
            /<\|reasoning_start\|>[\s\S]*?<\|reasoning_end\|>/gi,
            
            // Markdown-style
            /\[THINKING\][\s\S]*?\[\/THINKING\]/gi,
            /\[REASONING\][\s\S]*?\[\/REASONING\]/gi,
            /\[THOUGHT\][\s\S]*?\[\/THOUGHT\]/gi,
            
            // Other common patterns
            /\*\*Thinking:\*\*[\s\S]*?(?=\*\*Answer:\*\*|$)/gi,
            /\*\*Reasoning:\*\*[\s\S]*?(?=\*\*Answer:\*\*|$)/gi,
        ];

        // Apply all patterns to remove thinking sections
        for (const pattern of thinkingPatterns) {
            cleanedAnswer = cleanedAnswer.replace(pattern, '');
        }

        // Clean up extra whitespace
        cleanedAnswer = cleanedAnswer.trim();
        
        // If we removed everything, keep original (safety check)
        if (cleanedAnswer.length === 0 && cleanedResponse.answer.length > 0) {
            console.warn('⚠️ Thinking token removal resulted in empty answer. Keeping original.');
            return cleanedResponse;
        }

        cleanedResponse.answer = cleanedAnswer;
        return cleanedResponse;
    }
}