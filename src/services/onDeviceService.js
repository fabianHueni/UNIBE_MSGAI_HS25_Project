
// OnDeviceService: uses Xenova's transformers.js to run a small causal LM in browser
// Uses ES module import for Xenova's transformers.js
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.7.0/dist/transformers.min.js';


/**
 * On device llm inference service using transformers.js
 * TODO Implement this class!
 */
export class OnDeviceService {
    constructor({modelName = 'Xenova/distilgpt2'} = {}) {
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
        console.log("download model:", this.modelName);
        // Xenova's pipeline API (ES module)
        this._model = await pipeline('text-generation', this.modelName, {
            progress_callback: progressCb
        });
        console.log("model loaded");
        this._ready = true;
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
        console.log("run on device inference");
        if (!this._ready || !this._model) {
            console.log("model not ready:" , this._ready, this._model);
            throw new Error('Model not loaded. Call load() first.');
        }

        prompt = "Please answer the following question: " + prompt + "\nAnswer: "; // ensure string input
        console.log("infer on-device:", prompt);

        const output = await this._model(prompt, {
            max_new_tokens: maxNewTokens,
            temperature: 2,
            repetition_penalty: 1.5,
            no_repeat_ngram_size: 2,
            num_beams: 1,
            num_return_sequences: 2,

        });
        console.log("on-device output:", output);
        // Xenova's output is an array of objects with 'generated_text'
        return output[0]?.generated_text || '';
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