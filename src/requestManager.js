import {measureAsync} from './utils.js';


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
     * Handle a single inference job by routing it to the appropriate service,
     * performing inference, evaluating the result, and recording statistics.
     *
     * @param job - The job object containing prompt and ground truth
     * @returns {Promise<{route: string, latency: number, text: string, job, evalRes: (*|XPathResult|{exact: *, f1: *})}>}
     */
    async handle(job) {
        // get routing strategy and inference service
        const route = this._choose(job);
        const service = this._getInferenceService(route);

        const full_prompt = job.prompt; // ensure string input

        let response, latencyMs; // response is object with .answer and .stats
        try {
            // Mark inference start
            job.timestamps.inferenceStart = performance.now(); //TODO: take this timestamp in the service.infer method
            
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

        // evaluate result and store results
        const evalRes = this.evaluator.evaluate(response, job.groundTruth, latencyMs);
        this._record(route, latencyMs, evalRes, job, response, {queueingTime, inferenceTime, totalLatency});

        // logging the result
        if (this.logger) this.logger({job, route, latency: latencyMs, evalRes, response: response, queueingTime, inferenceTime, totalLatency});

        // logging on console
        console.log("🎯 Models Answer: " + response.answer +
            '; \nGround Truth: ' + job.groundTruth +
            "; \nInference Time: " + inferenceTime.toFixed(2) + "ms" +
            "; \nQueueing Time: " + queueingTime.toFixed(2) + "ms" +
            "; \nTotal Latency: " + totalLatency.toFixed(2) + "ms");


        return {job, route, latency: latencyMs, evalRes, text: response, queueingTime, inferenceTime, totalLatency};
    }


    /**
     * Get the inference service based on the selected route.
     * Could be extended with more services in the future.
     *
     * @param route - The selected route ('cloud' or 'device')
     * @returns {*}
     * @private
     */
    _getInferenceService(route) {
        return route === 'cloud' ? this.cloud : this.device;
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
}