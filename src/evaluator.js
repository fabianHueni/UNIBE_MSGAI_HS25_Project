/**
 * Evaluator class to run multiple evaluation metrics for a tasks such as exact text matching.
 */
export class Evaluator{
    constructor(){ }

/**
 * Run multiple metrics for a prediction against the ground truth and return the results.
 *
 * @param pred - Predicted string
 * @param truth - Ground truth string
 * @param latencyMs - Latency measured during inference
 * @returns {{exact: number, f1: number, rouge1: number, totalTokens: number, tokensPerSecond: number}}
 */
evaluate(pred, truth, latencyMs) {
    const total_tokens = this._countTokens(pred);
    return {
            exactMatch: this._exactMatch(pred, truth),
            f1WordLevel: this._f1WordLevel(pred, truth),
            rouge1: this._rouge1(pred, truth),
            totalTokens: total_tokens,
            tokensPerSecond: this._tokensPerSecond(total_tokens, latencyMs)
        };
    }

    /**
     * F1 score at word level (precision, recall, f1)
     * @param pred
     * @param truth
     * @returns {number}
     */
    _f1WordLevel(pred, truth) {
        const predTokens = this._normalize(pred).split(/\s+/).filter(Boolean);
        const truthTokens = this._normalize(truth).split(/\s+/).filter(Boolean);
        const predSet = new Set(predTokens);
        const truthSet = new Set(truthTokens);
        const common = predTokens.filter(t => truthSet.has(t));
        const numCommon = common.length;
        if (numCommon === 0) return 0;
        const precision = numCommon / predTokens.length;
        const recall = numCommon / truthTokens.length;
        return (2 * precision * recall) / (precision + recall);
    }

    /**
     * ROUGE-1 (unigram overlap recall)
     * @param pred
     * @param truth
     * @returns {number}
     */
    _rouge1(pred, truth) {
        const predTokens = this._normalize(pred).split(/\s+/).filter(Boolean);
        const truthTokens = this._normalize(truth).split(/\s+/).filter(Boolean);
        const truthTokenCounts = {};
        for (const t of truthTokens) {
            truthTokenCounts[t] = (truthTokenCounts[t] || 0) + 1;
        }
        let overlap = 0;
        for (const t of predTokens) {
            if (truthTokenCounts[t]) {
                overlap++;
                truthTokenCounts[t]--;
            }
        }
        return truthTokens.length > 0 ? overlap / truthTokens.length : 0;
    }

    /**
     * Check the prediction for exact match against the ground truth
     *
     * @param pred - Predicted string
     * @param truth- Ground truth string
     * @returns {number}
     * @private
     */
    _exactMatch(pred, truth){
        return this._normalize(pred) === this._normalize(truth) ? 1 : 0;
    }


    /**
     * Normalize a string to avoid false negatives due to spaces or capitalization
     * Convert input to a string in case it is not already
     *
     * @param s - Input string
     * @returns {string}
     * @private
     */
    _normalize(s){
        return String(s||'').trim().toLowerCase();
    }

    /**
     * Count the number of tokens (words) in a string
     * TODO: verify if this is the right method to count tokens.
     * @param s - Input string
     * @returns {number}
     */
    _countTokens(s) {
        return String(s||'').trim().split(/\s+/).filter(Boolean).length;
    }

    /**
     * Calculate tokens per second given token count and latency in ms
     * @param tokenCount - Number of tokens
     * @param latencyMs - Latency in milliseconds
     * @returns {number}
     */
    _tokensPerSecond(tokenCount, latencyMs) {
        return latencyMs > 0 ? tokenCount / (latencyMs / 1000) : 0;
    }

    /**
     * TODO: Implement more custom metrics for classification or NER task.
     *
     * @param pred - Predicted string
     * @param truth - Ground truth string
     * @private
     */
    _myMetric(pred, truth){
        return 0;
    }

}