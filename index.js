import {
    AutoProcessor,
    AutoModelForVision2Seq,
    RawImage,
    TextStreamer,
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
 * Updates the statistics and drug matches display
 * @param {Object} stats - Statistics object containing counts
 * @param {Array} drugMatches - Array of drug matches found
 */
function updateStatsAndMatches(stats, drugMatches) {
    // Update the statistics
    const statsHtml = `
        <div class="divide-y divide-gray-200">
            <div class="py-3 flex justify-between items-center">
                <span class="text-gray-600">Total Overlays</span>
                <span class="font-semibold text-indigo-600">${stats.totalOverlays}</span>
            </div>
            <div class="py-3 flex justify-between items-center">
                <span class="text-gray-600">Tag Types</span>
                <span class="font-semibold text-indigo-600">${stats.tagTypes}</span>
            </div>
            <div class="py-3 flex justify-between items-center">
                <span class="text-gray-600">Drug Matches Found</span>
                <span class="font-semibold text-indigo-600">${stats.totalDrugMatches}</span>
            </div>
        </div>
    `;
    document.getElementById('detection-stats').innerHTML = statsHtml;

    // Update drug matches display
    const drugMatchesHtml = drugMatches.length === 0 
        ? '<div class="text-gray-500 text-center py-4">No medication matches found</div>'
        : `
            <div class="grid grid-cols-2 gap-4">
                ${drugMatches.map(match => `
                    <div class="bg-white p-3 rounded-lg shadow-sm border border-gray-100">
                        <div class="font-medium text-indigo-600">${match.drug}</div>
                    </div>
                `).join('')}
            </div>
        `;
    document.getElementById('drug-matches').innerHTML = drugMatchesHtml;
}

// Helper: escape regex special chars
function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Helper: parse CSV text into array of objects (handles quoted fields)
function parseCSV(csvText) {
    const lines = csvText.split(/\r?\n/).filter(l => l.trim() !== '');
    if (lines.length === 0) return [];
    const splitLine = (line) => line.split(/,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)/).map(s => {
        let val = s.trim();
        if (val.startsWith('"') && val.endsWith('"')) {
            val = val.slice(1, -1).replace(/""/g, '"');
        }
        return val;
    });
    const headers = splitLine(lines[0]);
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        const parts = splitLine(lines[i]);
        if (parts.length === 0) continue;
        const obj = {};
        for (let j = 0; j < headers.length; j++) {
            obj[headers[j]] = parts[j] || '';
        }
        rows.push(obj);
    }
    return rows;
}

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
                embed_tokens: "fp16",
                vision_encoder: "fp32",
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
        // Draw image to canvas and get RawImage
        const ctx = hiddenCanvas.getContext("2d");
        hiddenCanvas.width = imageObject.width;
        hiddenCanvas.height = imageObject.height;
        ctx.drawImage(imageObject, 0, 0);
        const image = RawImage.fromCanvas(hiddenCanvas);
        
        // Create input messages
        const messages = [{
            role: "user",
            content: [{type: "image"}, {type: "text", text: promptInput.value}],
        }];
        
        // Prepare inputs for the model
        const text = processor.apply_chat_template(messages, {
            add_generation_prompt: true,
        });
        const inputs = await processor(text, [image], {
            do_image_splitting: true,
        });
        
        // Generate output
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
        
        // Create overlays on the image
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

        // Initial results and stats
        const results = {
            doclingText: fullText,
            overlays: overlays,
            htmlContent: doclingToHtml(fullText)
        };

        const stats = {
            totalOverlays: overlays.length,
            tagTypes: [...new Set(overlays.map(o => o.tagType))].length,
            totalDrugMatches: 0
        };

        // Update stats with initial values (no drug matches yet)
        updateStatsAndMatches(stats, []);
        console.log('Initial stats updated:', stats);

        // Load and process drug names from CSV
        try {
            console.log('Starting to load drugs.csv...');
            const response = await fetch('drugs.csv');
            const csvText = await response.text();
            console.log('CSV loaded, first 100 characters:', csvText.substring(0, 100));
            
            // Parse CSV robustly
                const rows = parseCSV(csvText);
            const drugNames = new Set();

            // Helper function to validate a potential drug name
            const isValidDrugName = (name) => {
                if (!name || name.length < 4) return false; // Must be at least 4 characters
                if (/^\d/.test(name)) return false; // Must not start with a number
                if (/^\d+\.?\d*$/.test(name)) return false; // Must not be just a number
                if (!/[a-zA-Z]/.test(name)) return false; // Must contain at least one letter
                return true;
            };

            rows.forEach(row => {
                // Process brand names
                const brand = (row['brand_name'] || row['brand'] || '').trim();
                if (brand) {
                    // Add the full brand name
                    const cleanBrand = brand
                        .replace(/\s*\([^)]*\)/g, '') // Remove parenthetical content
                        .replace(/\d+\s*(?:mg|ml|mcg|g)\b/gi, '') // Remove dosage amounts
                        .trim();
                    if (isValidDrugName(cleanBrand)) {
                        drugNames.add(cleanBrand.toLowerCase());
                    }

                    // For generic names with salts (e.g., "ATORVASTATIN CALCIUM"), add base name too
                    const baseName = cleanBrand.split(/\s+/)[0]; // Get first word
                    if (isValidDrugName(baseName)) {
                        drugNames.add(baseName.toLowerCase());
                    }
                }

                // Process active ingredients
                const ai = (row['active_ingredients'] || row['active_ingredient'] || '').trim();
                if (ai) {
                    // Split on commas and semicolons
                    const parts = ai.split(/[;,]/)
                        .map(p => {
                            const cleaned = p.trim()
                                .replace(/\s*\([^)]*\)/g, '') // Remove parenthetical content
                                .replace(/\d+\s*(?:mg|ml|mcg|g)\b/gi, '') // Remove dosage amounts
                                .trim();
                            
                            // Add both full name and base name (without salt)
                            const results = new Set();
                            if (isValidDrugName(cleaned)) {
                                results.add(cleaned.toLowerCase());
                            }
                            
                            // Add base name (first word) if it's valid
                            const baseName = cleaned.split(/\s+/)[0];
                            if (isValidDrugName(baseName)) {
                                results.add(baseName.toLowerCase());
                            }
                            
                            return Array.from(results);
                        })
                        .flat();
                    
                    parts.forEach(p => drugNames.add(p));
                }
            });            // Look for drug matches using word-boundary regex
            const drugMatches = [];
            const detectedText = doclingOutput.textContent.toLowerCase();
            for (const drugName of drugNames) {
                try {
                    const pattern = new RegExp('\\b' + escapeRegex(drugName) + '\\b', 'i');
                    const m = detectedText.match(pattern);
                    if (m) {
                        const idx = detectedText.search(pattern);
                        drugMatches.push({ drug: drugName, found: true });
                    }
                } catch (e) {
                    // fallback to simple includes if regex fails
                    if (detectedText.includes(drugName)) {
                        drugMatches.push({ drug: drugName, found: true });
                    }
                }
            }

            // Update results and display
            results.drugMatches = drugMatches;
            stats.totalDrugMatches = drugMatches.length;
            console.log('Found drug matches:', drugMatches.length);
            console.log('Updated stats:', stats);
            updateStatsAndMatches(stats, drugMatches);
            console.log('Stats and matches updated in UI');

            // Create HTML iframe
            htmlIframe.srcdoc = results.htmlContent;
        } catch (error) {
            console.error('Error processing drug matches:', error);
            updateStatsAndMatches(stats, []);
        }
    } catch (error) {
        console.error("Error during image processing:", error);
        doclingOutput.textContent = `An error occurred: ${error.message}`;
    } finally {
        setUiState("result");
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
        // Clear previous results when going back to initial
        document.getElementById('detection-stats').innerHTML = '';
        document.getElementById('drug-matches').innerHTML = '';
        welcomeMessage.style.display = "flex";
        generateBtn.disabled = true;
    } else if (state === "processing") {
        // Keep stats visible during processing, but clear matches while streaming
        document.getElementById('drug-matches').innerHTML = '';
        viewToggle.checked = false;
        processingIndicator.classList.remove("hidden");
        doclingView.classList.remove("hidden");
        generateBtn.disabled = true;
    } else if (state === "result") {
        // Preserve the populated stats and matches on result
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
        fetch(source)
            .then((res) => res.blob())
            .then((blob) => {
                img.src = URL.createObjectURL(blob);
            })
            .catch((e) => {
                console.error("CORS issue likely. Trying proxy or direct load.", e);
                img.crossOrigin = "anonymous";
                img.src = source;
            });
    } else {
        reader.onload = (e) => {
            img.src = e.target.result;
        };
        reader.readAsDataURL(source);
    }
}

// Event Listeners
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

fileInput.addEventListener("change", (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleImageSelection(files[0]);
    }
});

exampleImages.forEach((img) => {
    img.addEventListener("click", () => {
        promptInput.value = img.dataset.prompt;
        handleImageSelection(img.src);
    });
});

removeImageBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    currentImage = null;
    imagePreview.src = "";
    fileInput.value = "";
    imagePlaceholder.classList.remove("hidden");
    imagePreviewContainer.classList.add("hidden");
    examplesContainer.classList.remove("hidden");
    examplesTitle.classList.remove("hidden");
    setUiState("initial");
    doclingOutput.textContent = "";
    htmlIframe.srcdoc = "";
    clearOverlays();
});

viewToggle.addEventListener("change", () => {
    const isHtmlView = viewToggle.checked;
    htmlView.classList.toggle("hidden", !isHtmlView);
    doclingView.classList.toggle("hidden", isHtmlView);
});

generateBtn.addEventListener("click", () => {
    if (currentImage) {
        processImage(currentImage);
    }
});

document.addEventListener("DOMContentLoaded", () => {
    setUiState("initial");
    initializeModel();
});