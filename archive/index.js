import {
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForQuestionAnswering,
    RawImage,
    TextStreamer,
    pipeline
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

const modelLoaderOverlay = document.getElementById("model-loader-overlay");
const processingIndicator = document.getElementById("processing-indicator");

const promptInput = document.getElementById("prompt-input");
const generateBtn = document.getElementById("process-btn");
let model, processor;
let currentImage = null;


/**
 * Loads and initializes the model and processor.
 */
async function initializeModel() {
    try {
        const model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct";
        // processor = await AutoProcessor.from_pretrained(model_id);
        const progress = {};

        model ??= pipeline('translation', 'Xenova/nllb-200-distilled-600M', {
            progress_callback: (data) => {
                if (data.status === "progress" && data.file?.endsWith?.("onnx_data")) {
                    progress[data.file] = data;
                    if (Object.keys(progress).length !== 3) return;
                    let sum = 0;
                    let total = 0;
                    for (const [key, val] of Object.entries(progress)) {
                        sum += val.loaded;
                        total += val.total;
                    }
                    const overallPercent = Math.round((sum / total) * 100);
                    document.getElementById("model-progress").value = overallPercent;
                    document.getElementById("progress-text").textContent = overallPercent + "%";
                }
            }
        });


        /*
        model = await AutoModelForQuestionAnswering.from_pretrained(model_id, {
            dtype: {
                embed_tokens: "fp16",
                decoder_model_merged: "fp32",
            },
            device: "webgpu",
            progress_callback: (data) => {
                if (data.status === "progress" && data.file?.endsWith?.("onnx_data")) {
                    progress[data.file] = data;
                    if (Object.keys(progress).length !== 3) return;
                    let sum = 0;
                    let total = 0;
                    for (const [key, val] of Object.entries(progress)) {
                        sum += val.loaded;
                        total += val.total;
                    }
                    const overallPercent = Math.round((sum / total) * 100);
                    document.getElementById("model-progress").value = overallPercent;
                    document.getElementById("progress-text").textContent = overallPercent + "%";
                }
            },
        });
        */
        modelLoaderOverlay.style.display = "none";
        console.log("Model loaded successfully.");
    } catch (error) {
        console.error("Failed to load model:", error);
        modelLoaderOverlay.innerHTML = `
            <h2 class="text-center text-red-500 text-xl font-semibold">Failed to Load Model</h2>
            <p class="text-center text-white text-md mt-2">Please refresh the page to try again. Check the console for errors.</p>
        `;
    }
}

/**
 * Processes an image and generates Docling text.
 * @param {ImageBitmap|HTMLImageElement} imageObject An image object to process.
 */
async function process(imageObject) {

}


/**
 * Manages the visibility of UI components based on the app state.
 * @param {'initial'|'processing'|'result'} state The current state.
 */
function setUiState(state) {
    processingIndicator.classList.add("hidden");
    if (state === "initial") {
        // Clear previous results when going back to initial
        // document.getElementById('detection-stats').innerHTML = '';
        // document.getElementById('drug-matches').innerHTML = '';
        generateBtn.disabled = true;
    } else if (state === "processing") {
        // Keep stats visible during processing, but clear matches while streaming
        // document.getElementById('drug-matches').innerHTML = '';
        processingIndicator.classList.remove("hidden");
        generateBtn.disabled = true;
    } else if (state === "result") {
        // Preserve the populated stats and matches on result
        generateBtn.disabled = false;
    }
}


// Event Listeners
generateBtn.addEventListener("click", () => {
    if (currentImage) {
        processImage(currentImage);
    }
});

document.addEventListener("DOMContentLoaded", () => {
    setUiState("initial");
    initializeModel();
});