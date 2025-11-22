
// A modern, lightweight web server to handle API requests for a 3D model-to-tactile-display pipeline.

const express = require('express');
const { spawn } = require('child_process');
const cors = require('cors');
const path = require('path');

const app = express();
const port = 3000;

// Middleware to parse JSON bodies for POST requests
app.use(express.json());

// Serve the generated image files statically
app.use('/assets', express.static(path.join(__dirname, 'serve/assets')));

// Enable Cross-Origin Resource Sharing (CORS) for all routes
app.use(cors());

// --- Helper Function to run the Python script ---
function runPythonScript(scriptPath, args) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [scriptPath, ...args]);

        let stdout = '';
        let stderr = '';

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error(`Python script exited with code ${code}`);
                console.error('Stderr:', stderr);
                // Try to parse stderr as JSON for a more structured error
                try {
                    const errorJson = JSON.parse(stderr);
                    return reject(new Error(errorJson.error || 'Python script error.'));
                } catch (e) {
                    return reject(new Error(`Python Error: ${stderr}`));
                }
            }
            // The output from Python is now a JSON string, so we parse it
            try {
                const result = JSON.parse(stdout);
                resolve(result);
            } catch (e) {
                console.error("Raw stdout from Python:", stdout);
                reject(new Error('Failed to parse Python script output as JSON.'));
            }
        });
    });
}


// --- API Endpoints ---

/**
 * @route GET /
 * @description Serves the main frontend HTML file.
 */
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

/**
 * @route GET /api/slice
 * @description Retrieves a single slice of a 3D model.
 */
app.get('/api/slice', async (req, res) => {
    const requiredParams = ['file', 'poseX', 'poseY', 'poseZ', 'dirX', 'dirY', 'dirZ', 'n', 'i'];
    const missingParams = requiredParams.filter(param => !req.query[param]);

    if (missingParams.length > 0) {
        return res.status(400).json({ error: 'Missing required query parameters.', missing: missingParams });
    }

    try {
        const { file, poseX, poseY, poseZ, dirX, dirY, dirZ, n, i } = req.query;
        const args = ['retrieve', file, poseX, poseY, poseZ, dirX, dirY, dirZ, n, i];

        console.log(`Requesting slice ${i} of ${n}. Running script: python slicer.py ${args.join(' ')}`);

        const result = await runPythonScript('slicer.py', args);

        // Check if the python script returned a specific error
        if (result.error) {
            return res.status(500).json({ error: 'An error occurred while processing the model in Python.', details: result.error });
        }

        // Construct the full URL to the generated image using the path from the JSON object
        const fileUrl = `${req.protocol}://${req.get('host')}/${result.path.replace(/\\/g, '/')}`;

        // Send the entire result object back, now including the URL
        res.status(200).json({ ...result, url: fileUrl });

    } catch (error) {
        console.error('Failed to execute Python script or process its output:', error);
        res.status(500).json({ error: 'An error occurred on the server.', details: error.message });
    }
});

/**
 * @route POST /api/send-to-device
 * @description Receives raw slice data and simulates sending it to the tactile pin device.
 */
app.post('/api/send-to-device', (req, res) => {
    const { sliceData, width, height } = req.body;
    if (!sliceData || !width || !height) {
        return res.status(400).json({ error: 'Incomplete slice data provided. Required: sliceData, width, height.' });
    }

    // In a real application, this is where you would process the raw sliceData array
    // and send the appropriate signals to the hardware.
    console.log('--- Pin Device Simulation ---');
    console.log(`Received ${width}x${height} data grid to send to the tactile device.`);
    // console.log('Slice Data Grid:', sliceData); // This can be very long, so log selectively.
    console.log('---------------------------');

    res.status(200).json({ message: `Data grid successfully sent to the pin device.` });
});


// --- Server Start ---
app.listen(port, () => {
    console.log(`✅ Server is running at http://localhost:${port}`);
    console.log('Static file serving from:', path.join(__dirname, 'serve/assets'));
    console.log('Waiting for API requests...');
});
