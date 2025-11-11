// CloudService: example OpenRouter integration. Replace endpoint/payload per provider.

/**
 * Cloud inference service using a remote API from OpenRouter to access different models over one API.
 *
 */
export class CloudService {
    constructor({apiKey, model} = {}) {
        this.apiKey = apiKey;
        this.model = model || 'gpt-4o-mini';
    }


    /**
     * Update configuration with new values
     *
     * @param apiKey - The API key for authentication
     * @param model - The name of the model to use
     */
    updateConfig({apiKey, model}) {
        if (apiKey) this.apiKey = apiKey;
        if (model) this.model = model;
    }


    /**
     * Perform inference on the cloud service
     *
     * @param prompt - The input prompt string
     * @returns {Promise<string>}
     */
    async infer(prompt) {
        if (!this.apiKey) throw new Error('No API key set for CloudService');

        const payload = {
            model: this.model,
            messages: [{role: 'user', content: prompt}]
        };

        // call the api
        const resp = await fetch('https://api.openrouter.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify(payload)
        });

        // check for errors during request
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(`Cloud inference failed: ${resp.status} ${text}`);
        }

        const json = await resp.json();

        // TODO check parsing of response for model provider
        let text = '';
        try {
            if (json.choices && json.choices[0]) {
                text = json.choices[0].message?.content || json.choices[0].text || '';
            } else if (json.output) {
                text = Array.isArray(json.output) ? json.output.join('\n') : json.output;
            }
        } catch (e) {
            text = JSON.stringify(json).slice(0, 200);
        }
        return text;
    }
}