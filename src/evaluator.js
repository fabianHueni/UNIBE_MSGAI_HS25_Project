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
     * @returns {{exact: number, f1: (number|*)}}
     */
    evaluate(pred, truth){
        return { exact: this._exactMatch(pred, truth), f1: this._myMetric(pred, truth) };
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
     * TODO: Implement custom metric for classification or NER task.
     *
     * @param pred - Predicted string
     * @param truth - Ground truth string
     * @private
     */
    _myMetric(pred, truth){
        return 0;
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

}