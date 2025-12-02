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
                    cloudProb = 0.5
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
     */
    updateRouting({routeStrategy, cloudProb}) {
        if (routeStrategy) this.routeStrategy = routeStrategy;
        if (cloudProb !== undefined) this.cloudProb = cloudProb;
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
        const full_prompt = job.prompt; // ensure string input

        let response, latencyMs, cleanedResponse; // response is object with .answer and .stats
        try {
            // Mark inference start
            job.timestamps.inferenceStart = performance.now();

            const {res, ms} = await measureAsync(() => service.infer(full_prompt));
            response = res;
            latencyMs = ms;

            // Mark inference end
            job.timestamps.inferenceEnd = performance.now();
        } catch (err) {
            response = `__error__:${err.message}`;
            latencyMs = -1;
            job.timestamps.inferenceEnd = performance.now();
        }

        // Calculate timing metrics
        const queueingTime = job.timestamps.inferenceStart - job.timestamps.jobStart;
        const inferenceTime = job.timestamps.inferenceEnd - job.timestamps.inferenceStart;
        const totalLatency = job.timestamps.inferenceEnd - job.timestamps.jobStart;

        // clean response
        cleanedResponse = this._stripThinkingTokens(response);

        // evaluate result and store results
        const evalRes = this.evaluator.evaluate(cleanedResponse, job.groundTruth, latencyMs);
        this._record(route, latencyMs, evalRes, job, cleanedResponse, {queueingTime, inferenceTime, totalLatency});

        // logging the result
        if (this.logger) this.logger({job, route, latency: latencyMs, evalRes, response: cleanedResponse, queueingTime, inferenceTime, totalLatency});

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
     * Choose the route for the given job based on the routing strategy.
     *
     * TODO: extend routing to be based on the job characteristics (e.g., prompt length, expected latency, etc.)
     *
     * @param job - The job object (not used in current strategies, could be used for more advanced routing)
     * @returns {string|string}
     * @private
     */
    _choose(job) {
        if (this.routeStrategy === 'always_cloud') return 'cloud';
        if (this.routeStrategy === 'always_device') return 'device';
        if (this.routeStrategy === 'probabilistic') return Math.random() < this.cloudProb ? 'cloud' : 'device';
        // default round robin
        this._rrCounter++;
        return (this._rrCounter % 2 === 0) ? 'cloud' : 'device';
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


    _stripThinkingTokens(response) {
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