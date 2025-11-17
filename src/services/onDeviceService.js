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
        console.log("Downloading model:", this.modelName);
        // Provide a default progress callback if none is given
        const defaultProgressCb = (progress) => {
            if (progress && typeof progress === 'object') {
                if (progress.status) {
                    console.log(`[Model Loading] ${progress.status}`);
                }
                if (progress.loaded && progress.total) {
                    const percent = ((progress.loaded / progress.total) * 100).toFixed(1);
                    console.log(`[Model Loading] ${percent}% (${progress.loaded}/${progress.total} bytes)`);
                }
            } else {
                console.log(`[Model Loading] Progress:`, progress);
            }
        };
        // Xenova's pipeline API (ES module)
        this._model = await pipeline('text-generation', this.modelName, {
            progress_callback: progressCb || defaultProgressCb
        });
        console.log("Model loaded and ready.");
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
    async infer(prompt, {maxNewTokens = 100} = {}) {
        if (!this._ready || !this._model) {
            console.log("model not ready:" , this._ready, this._model);
            throw new Error('Model not loaded. Call load() first.');
        }
        prompt = "Please answer the following question: " + prompt + "\nAnswer: "; // ensure string input
        console.log("running inference on-device:\n", prompt);

        const output = await this._model(prompt, {
            max_new_tokens: maxNewTokens,
            temperature: 1.5,
            repetition_penalty: 1.5,
            no_repeat_ngram_size: 2,
            num_beams: 1,
            num_return_sequences: 1,
        });

        // Return generated text
        return output[0]?.generated_text?.trim() || '';
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