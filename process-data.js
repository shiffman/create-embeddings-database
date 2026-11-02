import fs from 'fs';
import csv from 'csv-parser';
import axios from 'axios';
import path from 'path';
import { fileURLToPath } from 'url';
import { pipeline, env, LogLevel } from '@huggingface/transformers';

// Only show Transformers.js errors, not warnings, so the progress below stays readable
env.logLevel = LogLevel.ERROR;

// Get directory path
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function readTSV(filePath) {
  const results = [];
  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(csv({ separator: '\t' }))
      .on('data', (data) => results.push(data))
      .on('end', () => resolve(results))
      .on('error', (err) => reject(err));
  });
}

// Helper function to download images asynchronously
async function downloadImage(url, filename) {
  const imagesDir = path.resolve(__dirname, 'images');

  // Ensure the images directory exists
  if (!fs.existsSync(imagesDir)) {
    fs.mkdirSync(imagesDir);
  }

  const filePath = path.join(imagesDir, filename);
  const writer = fs.createWriteStream(filePath);

  const response = await axios.get(url, { responseType: 'stream' });
  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on('finish', () => resolve(filePath));
    writer.on('error', reject);
  });
}

// if partially downloaded try
// rm -rf node_modules/@huggingface/transformers/.cache

// The image-feature-extraction pipeline gives CLIP's image embedding (512 numbers)
console.log('Loading CLIP (Xenova/clip-vit-base-patch16)');
const extractor = await pipeline('image-feature-extraction', 'Xenova/clip-vit-base-patch16', {
  // device: 'webgpu',
  dtype: 'fp32',
  // Show the download of each model file on one line
  progress_callback: (x) => {
    if (x.status === 'progress') {
      log(`Loading ${x.file}: ${x.progress.toFixed(0)}%`);
    }
  },
});
log('Model ready\n');

const photosTSV = await readTSV('unsplash/photos.tsv000');
console.log(`Processing ${photosTSV.length} photos from Unsplash`);
const startTime = Date.now();

let photos = [];
let embeddings = [];

for (let i = 0; i < photosTSV.length; i++) {
  const photo = photosTSV[i];

  // Adding URL parameters to resize and compress the image
  const imageUrl = `${photo.photo_image_url}?w=1024&h=1024&fit=max&q=90`;
  const filename = `${photo.photo_id}.jpg`;
  const filePath = path.resolve(__dirname, 'images', filename);

  // Only download the image if it doesn't exist
  // No need to actually down the images but good to have the option
  // if (!fs.existsSync(filePath)) {
  //   log(`Downloading ${filename}: ${i + 1}/${photosTSV.length}\r`);
  //   await downloadImage(imageUrl, filename);
  // }

  // Process the embedding
  try {
    const output = await extractor(imageUrl);
    const image_embeddings = output.normalize().tolist();

    // Add to arrays
    embeddings.push(image_embeddings[0]);
    photos.push({ id: photo.photo_id, url: photo.photo_image_url });
  } catch (e) {
    // If an image can't be loaded, skip it and keep going
    log(`Skipped ${photo.photo_id}: ${e.message}\n`);
  }

  // Show progress on one line, with an estimate of the time left
  const done = i + 1;
  const secondsPerPhoto = (Date.now() - startTime) / 1000 / done;
  const minutesLeft = (secondsPerPhoto * (photosTSV.length - done)) / 60;
  log(`Computed embeddings: ${done}/${photosTSV.length}, about ${minutesLeft.toFixed(0)} minutes left`);

  // Save every 500 cycles since it takes a long time to run
  if ((i + 1) % 500 == 0) {
    writeData();
  }
}

// If we get to the end
writeData();
console.log('Done!');

function writeData() {
  // Step 1: Flatten the nested array of embeddings into one long array
  // Each embedding is a 512-dimensional array, and we have many of them
  // The flat() method concatenates all subarrays into one long array
  let flattened = embeddings.flat();

  // Step 2: Convert to binary format (Float32Array)
  // This creates a binary representation of our floating point numbers
  // Each number uses exactly 4 bytes (32 bits) of memory
  // Much more efficient than storing as strings in JSON
  const embeddingsBuffer = Buffer.from(new Float32Array(flattened).buffer);

  // Step 3: Write the binary data to a file
  // The resulting .bin file is much smaller than equivalent JSON would be

  // Create the embeddings directory if it doesn't exist
  if (!fs.existsSync('embeddings')) fs.mkdirSync('embeddings');
  fs.writeFileSync('embeddings/embeddings.bin', embeddingsBuffer);

  // Save image metadata separately as JSON
  // This is kept as JSON since it's smaller and we need the structure
  fs.writeFileSync('embeddings/photo.json', JSON.stringify(photos));

  log(`Saved ${embeddings.length} embeddings to embeddings/embeddings.bin and photo info to embeddings/photo.json\n`);
}

// Write a message on the current line, replacing what was there
// (end the message with \n to keep it and move to the next line)
function log(message) {
  process.stdout.clearLine();
  process.stdout.cursorTo(0);
  process.stdout.write(message);
}
