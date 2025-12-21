
/**
 * DatasetLoader loads a dataset and prepare it for the processing.
 */
export class DatasetLoader {
    constructor(datasetName = 'boolq_validation') {
        this.running = false;
        this._dataset = null;
        this._datasetName = datasetName

        this.loadDataset(this._datasetName);
    }

    /**
     * Load the dataset from CSV file based on the given name
     * If a comma appears inside a quote (context) it is not interpreted as a delimiter
     *
     * @param name - Name of the csv dataset to load without file extension
     * @private
     */
    loadDataset(name) {
        const path = `./dataset/${name}.csv`;

        return fetch(path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Dataset file not found: ${path}`);
                }
                return response.text();
            })
            .then(data => {
                const lines = data.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
                // drop header
                lines.shift();

                this._dataset = lines
                    .filter(l => l.trim().length > 0)
                    .map(line => {
                        let id, answer, full_prompt, question, context, text;

                        // load different datasets based on name
                        if (name === 'boolq_validation') {
                            // parse line into fields handling quoted commas
                            [id, question, answer, context] = this._parseCSVLine(line);

                            // set the prompt
                            full_prompt = `Question: ${question}
                                        Context: ${context}
                                        Instructions: Answer with ONLY the word "true" or "false". Do not provide any explanation or additional text.
                                        Answer:`;
                        } else if (name === 'spam_ham_dataset') {
                            [id, text, answer] = this._parseCSVLine(line);

                            // convert answer to string boolean
                            answer = (answer.toLowerCase() === 'spam') ? 'true' : 'false';

                            // set the prompt
                            full_prompt = `Task: Determine whether the following message is spam or not.
                                        Instructions: Answer with ONLY the word "true" or "false". Do not provide any explanation or additional text.
                                        Message: ${text}
                                        Answer:`;
                        }

                        return {id: id, prompt: full_prompt, groundTruth: answer};
                    });

                console.log(`✅ Dataset '${name}' loaded with ${this._dataset.length} items.`);
                console.log(this._dataset.slice(0, 2)); // log first 2 items for verification
                return this._dataset;
            })
            .catch(error => {
                console.error(error);
            });
    }

    /**
     * Parse a single CSV line into fields, handling quoted fields with commas
     *
     * @param line - A single line from a CSV file
     * @private
     */
    _parseCSVLine(line) {

        // inline CSV parse with quotes support
        const fields = [];
        let cur = '';
        let inQuotes = false;

        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (inQuotes) { // if we are in a quote we just look for the quote ending
                if (ch === '"') {
                    // escaped quote ""
                    if (i + 1 < line.length && line[i + 1] === '"') {
                        cur += '"';
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    cur += ch;
                }
            } else {   // only if we are not in a quote we count the comma as e delimiter
                if (ch === ',') {
                    fields.push(cur);
                    cur = '';
                } else if (ch === '"') {
                    inQuotes = true;
                } else {
                    cur += ch;
                }
            }
        }
        fields.push(cur);
        return fields;
    }
}

