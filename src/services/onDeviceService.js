// OnDeviceService: uses transformers.js to run a small causal LM in browser
// Requires the transformers.js script loaded in index.html (cdn).


/**
 * On device llm inference service using transformers.js
 * TODO Implement this class!
 */
export class OnDeviceService {
    constructor({modelName = 'distilgpt2'} = {}) {
        this.modelName = modelName;
        this._ready = false;
        this._model = null;
    }


    /**
     * Load the model into memory to be ready for inference.
     * Download the model if not already cached. Cache the model for future use.
     * TODO Download models from a model hub like HuggingFace using transformers.js
     *
     * @param progressCb
     * @returns {Promise<void>}
     */
    async load(progressCb) {

    }


    /**
     * Returns if the model is loaded and ready for inference
     * @returns {boolean}
     */
    isReady() {
        return this._ready;
    }


    /**
     * Perform inference on the on-device model
     * TODO Implement inference
     *
     * @param prompt - The input prompt string
     * @param maxNewTokens - Maximum number of new tokens to generate
     * @returns {Promise<string>}
     */
    async infer(prompt, {maxNewTokens = 50} = {}) {
        return "The Answer is 42!";
    }

    /**
     * Update configuration with new values
     *
     * @param modelName - The name of the model to use
     */
    updateConfig({modelName}) {
        if (modelName) this.modelName = modelName;
    }
}