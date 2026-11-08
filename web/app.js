// Classify photos in the browser with ONNX Runtime Web. Nothing is sent to a server.
// The image preparation must match src/data.py's eval_transform, or predictions get worse.

const CLASS_NAMES = ["Cat", "Dog"];
const IMAGE_SIZE = 224;
const RESIZE_TO = 256;  // shorter side is resized to 256, then the centre 224 x 224 is cropped
const MEAN = [0.485, 0.456, 0.406];  // ImageNet colour statistics, the same as in training
const STD = [0.229, 0.224, 0.225];

// Running locally (python -m http.server from the project folder) uses the freshly exported file;
// the live site downloads it from the Hugging Face model repo.
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const MODEL_URL = IS_LOCAL
  ? "../models/model.onnx"
  : "https://huggingface.co/bebird/cats-vs-dogs-efficientnet-b0/resolve/main/model.onnx";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

const el = (id) => document.getElementById(id);
const fileInput = el("file-input");
const dropzone = el("dropzone");
const preview = el("preview");
const statusText = el("status");

let session = null;
const sessionReady = ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] })
  .then((s) => { session = s; setStatus("ready"); })
  .catch((err) => { console.error(err); setStatus("could not load the model"); });

function setStatus(text) { statusText.textContent = text; }

/** Turn an image into the 1 x 3 x 224 x 224 array of numbers the model expects. */
function preprocess(bitmap) {
  // Crop the centre square that Resize(256) + CenterCrop(224) would keep, scaled straight to 224
  const side = Math.min(bitmap.width, bitmap.height) * IMAGE_SIZE / RESIZE_TO;
  const sx = (bitmap.width - side) / 2;
  const sy = (bitmap.height - side) / 2;

  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const { data } = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);  // RGBA, one byte each

  // Rearrange from pixel-by-pixel RGBA to three separate colour planes, and normalise
  const pixels = IMAGE_SIZE * IMAGE_SIZE;
  const input = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * pixels + i] = (data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return new ort.Tensor("float32", input, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

/** Return {Cat: p, Dog: p} for an image file or blob. */
async function classify(file) {
  await sessionReady;
  if (!session) throw new Error("model not loaded");
  const bitmap = await createImageBitmap(file);
  const output = await session.run({ image: preprocess(bitmap) });
  const probabilities = output.probabilities.data;
  return Object.fromEntries(CLASS_NAMES.map((name, i) => [name, probabilities[i]]));
}

function showResult(probabilities) {
  const ranked = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  el("result-label").textContent = ranked[0][0];
  el("bars").innerHTML = ranked.map(([name, p]) => `
    <div class="bar-row">
      <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
      <div class="bar-text"><span>${name}</span><span>${(p * 100).toFixed(1)}%</span></div>
    </div>`).join("");
}

function clearResult() {
  el("result-label").textContent = "?";
  el("bars").innerHTML = "";
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  el("dz-text").hidden = true;
  el("clear-btn").disabled = false;
  clearResult();
  setStatus("thinking...");
  try {
    showResult(await classify(file));
    setStatus("ready");
  } catch (err) {
    console.error(err);
    setStatus("could not read that photo");
  }
}

fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
el("choose-btn").addEventListener("click", () => fileInput.click());
el("clear-btn").addEventListener("click", () => {
  fileInput.value = "";
  preview.hidden = true;
  preview.removeAttribute("src");
  el("dz-text").hidden = false;
  el("clear-btn").disabled = true;
  clearResult();
});

// Drag and drop
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  handleFile(e.dataTransfer.files[0]);
});

// Exposed for automated checks that compare the browser's answers with PyTorch's
window.classifyBlob = classify;
