const form = document.getElementById('slicer-form');
const loader = document.getElementById('loader');
const resultImage = document.getElementById('result-image');
const placeholderText = document.getElementById('placeholder-text');
const errorMessage = document.getElementById('error-message');

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    loader.classList.remove('hidden');
    resultImage.classList.add('hidden');
    placeholderText.classList.remove('hidden');
    errorMessage.textContent = '';

    const formData = new FormData(form);
    const params = new URLSearchParams(formData);

    const requestUrl = `/api/slice?${params.toString()}`;
    console.log("Fetching:", requestUrl);

    try {
        const response = await fetch(requestUrl);
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || "Server error.");

        console.log("Slice Data:", data);

        if (data.url) {
            // Cache-busting so browser reloads the image
            resultImage.src = data.url + "?t=" + new Date().getTime();
            resultImage.classList.remove('hidden');
            placeholderText.classList.add('hidden');
        } else {
            placeholderText.textContent = "No slice generated.";
        }

    } catch (err) {
        console.error("Fetch error:", err);
        errorMessage.textContent = `Error: ${err.message}`;
    } finally {
        loader.classList.add('hidden');
    }
});
