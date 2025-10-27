import {
    AutoProcessor,
    AutoModelForVision2Seq,
    RawImage,
    TextStreamer,
    load_image
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";
import {doclingToHtml} from "./docling-html-parser.js";

const modelLoaderOverlay = document.getElementById("model-loader-overlay");
const imageDropArea = document.getElementById("image-drop-area");
const imagePlaceholder = document.getElementById("image-placeholder");
const imagePreviewContainer = document.getElementById("image-preview-container");
const imagePreview = document.getElementById("image-preview");
const removeImageBtn = document.getElementById("remove-image-btn");
const fileInput = document.getElementById("file-input");
const exampleImages = document.querySelectorAll(".example-image");
const examplesContainer = document.getElementById("examples-container");
const examplesTitle = document.getElementById("examples-title");
const processingIndicator = document.getElementById("processing-indicator");
const welcomeMessage = document.getElementById("welcome-message");
const doclingView = document.getElementById("docling-view");
const htmlView = document.getElementById("html-view");
const doclingOutput = document.getElementById("docling-output");
const htmlIframe = document.getElementById("html-iframe");
const viewToggle = document.getElementById("view-toggle");
const hiddenCanvas = document.getElementById("hidden-canvas");
const promptInput = document.getElementById("prompt-input");
const generateBtn = document.getElementById("generate-btn");
let model, processor;
let currentImageWidth, currentImageHeight;
let currentImage = null;

/**
 * Loads and initializes the model and processor.
 */
async function initializeModel() {
    try {
        const model_id = "onnx-community/granite-docling-258M-ONNX";
        processor = await AutoProcessor.from_pretrained(model_id);
        const progress = {};
        model = await AutoModelForVision2Seq.from_pretrained(model_id, {
            dtype: {
                embed_tokens: "fp16", // fp32 (231 MB) | fp16 (116 MB)
                vision_encoder: "fp32", // fp32 (374 MB)
                decoder_model_merged: "fp32", // fp32 (658 MB) | q4 (105 MB), q4 sometimes into repetition issues
            },
            device: "webgpu",
            progress_callback: (data) => {
                if (data.status === "progress" && data.file?.endsWith?.("onnx_data")) {
                    progress[data.file] = data;
                    const progressPercent = Math.round(data.progress);
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
async function processImage(imageObject) {
    if (!model || !processor) {
        alert("Model is not loaded yet. Please wait.");
        return;
    }
    // Reset UI
    setUiState("processing");
    clearOverlays();
    let fullText = "";
    doclingOutput.textContent = "";
    htmlIframe.srcdoc = "";
    try {
        // 1. Draw image to canvas and get RawImage
        const ctx = hiddenCanvas.getContext("2d");
        hiddenCanvas.width = imageObject.width;
        hiddenCanvas.height = imageObject.height;
        ctx.drawImage(imageObject, 0, 0);
        const image = RawImage.fromCanvas(hiddenCanvas);
        // 2. Create input messages
        const messages = [
            {
                role: "user",
                content: [{type: "image"}, {type: "text", text: promptInput.value}],
            },
        ];
        // 3. Prepare inputs for the model
        const text = processor.apply_chat_template(messages, {
            add_generation_prompt: true,
        });
        const inputs = await processor(text, [image], {
            do_image_splitting: true,
        });
        // 5. Generate output
        await model.generate({
            ...inputs,
            max_new_tokens: 4096,
            streamer: new TextStreamer(processor.tokenizer, {
                skip_prompt: true,
                skip_special_tokens: false,
                callback_function: (streamedText) => {
                    fullText += streamedText;
                    doclingOutput.textContent += streamedText;
                },
            }),
        });
        // Strip <|end_of_text|> from the end
        fullText = fullText.replace(/<\|end_of_text\|>$/, "");
        doclingOutput.textContent = fullText;
        // Parse loc tags and create overlays
        const tagRegex = /<(\w+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/g;
        const overlays = [];
        let match;
        while ((match = tagRegex.exec(fullText)) !== null) {
            const tagType = match[1];
            const locs = [parseInt(match[2]), parseInt(match[3]), parseInt(match[4]), parseInt(match[5])];
            overlays.push({tagType, locs});
        }
        const colorMap = {};

        function getRandomColor() {
            return `rgb(${Math.floor(Math.random() * 256)}, ${Math.floor(Math.random() * 256)}, ${Math.floor(Math.random() * 256)})`;
        }

        const imgRect = imagePreview.getBoundingClientRect();
        const containerRect = imagePreviewContainer.getBoundingClientRect();
        const imageOffsetLeft = imgRect.left - containerRect.left;
        const imageOffsetTop = imgRect.top - containerRect.top;
        const scaleX = imgRect.width / currentImageWidth;
        const scaleY = imgRect.height / currentImageHeight;
        overlays.forEach(({tagType, locs}) => {
            const color = colorMap[tagType] || (colorMap[tagType] = getRandomColor());
            const [leftLoc, topLoc, rightLoc, bottomLoc] = locs;
            const left = imageOffsetLeft + (leftLoc / 500) * currentImageWidth * scaleX;
            const top = imageOffsetTop + (topLoc / 500) * currentImageHeight * scaleY;
            const width = ((rightLoc - leftLoc) / 500) * currentImageWidth * scaleX;
            const height = ((bottomLoc - topLoc) / 500) * currentImageHeight * scaleY;
            const overlay = document.createElement("div");
            overlay.className = "overlay";
            overlay.style.setProperty('--overlay-color', color);
            const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
            overlay.style.setProperty('--overlay-color-rgb', `${rgbMatch[1]},${rgbMatch[2]},${rgbMatch[3]}`);
            overlay.style.position = "absolute";
            overlay.style.left = left + "px";
            overlay.style.top = top + "px";
            overlay.style.width = width + "px";
            overlay.style.height = height + "px";
            imagePreviewContainer.appendChild(overlay);
        });
        // After generation, create the HTML iframe
        htmlIframe.srcdoc = doclingToHtml(fullText);
    } catch (error) {
        console.error("Error during image processing:", error);
        doclingOutput.textContent = `An error occurred: ${error.message}`;
    } finally {
        setUiState("result");
    }
}

/**
 * Handles the selection of an image file.
 * @param {File|string} source The image file or URL.
 */
function handleImageSelection(source) {
    const reader = new FileReader();
    const img = new Image();
    img.onload = () => {
        currentImageWidth = img.naturalWidth;
        currentImageHeight = img.naturalHeight;
        currentImage = img;
        imagePreview.src = img.src;
        imagePlaceholder.classList.add("hidden");
        imagePreviewContainer.classList.remove("hidden");
        examplesContainer.classList.add("hidden");
        examplesTitle.classList.add("hidden");
        processImage(img);
    };
    img.onerror = () => {
        alert("Failed to load image.");
    };
    if (typeof source === "string") {
        // It's a URL
        // To avoid CORS issues with canvas, we can try to fetch it first
        fetch(source)
            .then((res) => res.blob())
            .then((blob) => {
                img.src = URL.createObjectURL(blob);
            })
            .catch((e) => {
                console.error("CORS issue likely. Trying proxy or direct load.", e);
                // Fallback to direct load which might taint the canvas
                img.crossOrigin = "anonymous";
                img.src = source;
            });
    } else {
        // It's a File object
        reader.onload = (e) => {
            img.src = e.target.result;
        };
        reader.readAsDataURL(source);
    }
}

/**
 * Manages the visibility of UI components based on the app state.
 * @param {'initial'|'processing'|'result'} state The current state.
 */
function setUiState(state) {
    welcomeMessage.style.display = "none";
    processingIndicator.classList.add("hidden");
    doclingView.classList.add("hidden");
    htmlView.classList.add("hidden");
    if (state === "initial") {
        welcomeMessage.style.display = "flex";
        generateBtn.disabled = true;
    } else if (state === "processing") {
        viewToggle.checked = false;
        processingIndicator.classList.remove("hidden");
        doclingView.classList.remove("hidden");
        generateBtn.disabled = true;
    } else if (state === "result") {
        viewToggle.checked = true;
        htmlView.classList.remove("hidden");
        generateBtn.disabled = false;
    }
}

/**
 * Clears all overlay divs from the image preview container.
 */
function clearOverlays() {
    document.querySelectorAll(".overlay").forEach((el) => el.remove());
}

// Drag and Drop
imageDropArea.addEventListener("click", () => fileInput.click());
imageDropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    imageDropArea.classList.add("border-indigo-500", "bg-indigo-50");
});
imageDropArea.addEventListener("dragleave", () => {
    imageDropArea.classList.remove("border-indigo-500", "bg-indigo-50");
});
imageDropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    imageDropArea.classList.remove("border-indigo-500", "bg-indigo-50");
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith("image/")) {
        handleImageSelection(files[0]);
    }
});
// File input
fileInput.addEventListener("change", (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleImageSelection(files[0]);
    }
});
// Example images
exampleImages.forEach((img) => {
    img.addEventListener("click", () => {
        promptInput.value = img.dataset.prompt;
        handleImageSelection(img.src);
    });
});
// Remove image
removeImageBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    currentImage = null;
    imagePreview.src = "";
    fileInput.value = ""; // Reset file input
    imagePlaceholder.classList.remove("hidden");
    imagePreviewContainer.classList.add("hidden");
    examplesContainer.classList.remove("hidden");
    examplesTitle.classList.remove("hidden");
    setUiState("initial");
    doclingOutput.textContent = "";
    htmlIframe.srcdoc = "";
    clearOverlays();
});
// View toggle
viewToggle.addEventListener("change", () => {
    const isHtmlView = viewToggle.checked;
    htmlView.classList.toggle("hidden", !isHtmlView);
    doclingView.classList.toggle("hidden", isHtmlView);
});
// Generate button
generateBtn.addEventListener("click", () => {
    if (currentImage) {
        processImage(currentImage);
    }
});
document.addEventListener("DOMContentLoaded", () => {
    setUiState("initial"); // Set initial view correctly
    initializeModel();
});