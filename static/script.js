// -----------------------------------------------------------------------------
// ELEMENTS
// -----------------------------------------------------------------------------
const form = document.getElementById("slicer-form");
const resultContainer = document.getElementById("result-container");

const loader = document.getElementById("loader");
const resultImage = document.getElementById("result-image");
const placeholderText = document.getElementById("placeholder-text");
const errorMessage = document.getElementById("error-message");

const fullscreenBtn = document.getElementById("fullscreenBtn");

const progressWrapper = document.getElementById("progress-wrapper");
const progressBar = document.getElementById("progress-bar");

const viewerOverlay = document.getElementById("viewerOverlay");
const viewerImage = document.getElementById("viewerImage");
const closeViewer = document.getElementById("closeViewer");
const sliceLabel = document.getElementById("sliceLabel");
const sliceAnnouncer = document.getElementById("sliceAnnouncer");

const openSettings = document.getElementById("openSettings");
const settingsModal = document.getElementById("settingsModal");
const closeSettings = document.getElementById("closeSettings");
const settingsForm = document.getElementById("settingsForm");
const modalI = document.getElementById("modal-i");
const modalN = document.getElementById("modal-n");


// -----------------------------------------------------------------------------
// PROGRESS BAR LOGIC
// -----------------------------------------------------------------------------
function startProgressBar() {
    progressWrapper.classList.remove("hidden");
    progressBar.style.width = "0%";

    let p = 0;
    window.progressInterval = setInterval(() => {
        p += Math.random() * 20;
        if (p > 90) p = 90;
        progressBar.style.width = p + "%";
    }, 300);
}

function stopProgressBar() {
    clearInterval(window.progressInterval);
    progressBar.style.width = "100%";

    setTimeout(() => {
        progressWrapper.classList.add("hidden");
        progressBar.style.width = "0%";
    }, 400);
}


// -----------------------------------------------------------------------------
// UPDATE SLICE LABEL (visual + screen reader)
// -----------------------------------------------------------------------------
function updateSliceLabel() {
    const i = parseInt(document.getElementById("i").value);
    const n = parseInt(document.getElementById("n").value);

    // Visible label
    sliceLabel.textContent = `Slice ${i} / ${n}`;

    // Screen-reader live region
    sliceAnnouncer.textContent = `Slice ${i} of ${n}`;
}


// -----------------------------------------------------------------------------
// LOAD SLICE (main functionality)
// -----------------------------------------------------------------------------
form.addEventListener("submit", async (event) => {
    event.preventDefault();

    fullscreenBtn.classList.add("hidden");
    resultImage.classList.add("hidden");
    placeholderText.classList.remove("hidden");
    errorMessage.textContent = "";

    startProgressBar();

    const formData = new FormData(form);
    const params = new URLSearchParams(formData);

    try {
        const response = await fetch(`/api/slice?${params.toString()}`);
        const data = await response.json();

        stopProgressBar();

        if (!response.ok) throw new Error(data.error || "Server error occurred");

        if (data.url) {
            resultImage.src = data.url + "?t=" + Date.now();
            resultImage.classList.remove("hidden");
            placeholderText.classList.add("hidden");
            fullscreenBtn.classList.remove("hidden");

            if (!viewerOverlay.classList.contains("hidden")) {
                viewerImage.src = resultImage.src;
            }

            updateSliceLabel();

            resultContainer.scrollIntoView({ behavior: "smooth" });

            modalI.value = formData.get("i");
            modalN.value = formData.get("n");
        }

    } catch (err) {
        stopProgressBar();
        errorMessage.textContent = err.message;
        sliceAnnouncer.textContent = "Error loading slice.";
    }
});


// -----------------------------------------------------------------------------
// FULLSCREEN VIEWER ACCESSIBILITY + FOCUS TRAP
// -----------------------------------------------------------------------------

// elements that can receive focus
const focusableSelector =
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

function trapFocus(container) {
    const focusables = container.querySelectorAll(focusableSelector);
    if (!focusables.length) return;

    let first = focusables[0];
    let last = focusables[focusables.length - 1];

    container.addEventListener("keydown", (e) => {
        if (e.key !== "Tab") return;

        if (e.shiftKey) {
            if (document.activeElement === first) {
                e.preventDefault();
                last.focus();
            }
        } else {
            if (document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        }
    });
}


// -----------------------------------------------------------------------------
// OPEN FULLSCREEN
// -----------------------------------------------------------------------------
function openFullscreen() {
    if (!resultImage.src) return;

    viewerImage.src = resultImage.src;
    viewerOverlay.classList.remove("hidden");

    updateSliceLabel();

    sliceAnnouncer.textContent = "Fullscreen viewer opened.";

    closeViewer.focus();
    trapFocus(viewerOverlay);

    if (viewerOverlay.requestFullscreen) {
        viewerOverlay.requestFullscreen();
    }
}


// -----------------------------------------------------------------------------
// CLOSE FULLSCREEN
// -----------------------------------------------------------------------------
function closeFullscreen() {
    viewerOverlay.classList.add("hidden");

    if (document.fullscreenElement) {
        document.exitFullscreen();
    }

    fullscreenBtn.focus();
    sliceAnnouncer.textContent = "Fullscreen viewer closed.";
}


// -----------------------------------------------------------------------------
// EVENT HANDLERS
// -----------------------------------------------------------------------------
document.getElementById("image-display").onclick = openFullscreen;
fullscreenBtn.onclick = openFullscreen;
closeViewer.onclick = closeFullscreen;


// -----------------------------------------------------------------------------
// KEYBOARD NAVIGATION IN FULLSCREEN
// -----------------------------------------------------------------------------
document.addEventListener("keydown", (e) => {
    if (viewerOverlay.classList.contains("hidden")) return;

    const idx = document.getElementById("i");
    const total = document.getElementById("n");

    let current = parseInt(idx.value);
    let max = parseInt(total.value);

    if (e.key === "Escape") {
        closeFullscreen();
        return;
    }

    if (e.key === "ArrowRight") {
        if (current <= max) {
            idx.value = current + 1;
            form.dispatchEvent(new Event("submit"));
        }
    }

    if (e.key === "ArrowLeft") {
        if (current > 0) {
            idx.value = current - 1;
            form.dispatchEvent(new Event("submit"));
        }
    }
});


// -----------------------------------------------------------------------------
// SETTINGS MODAL
// -----------------------------------------------------------------------------
openSettings.onclick = () => {
    modalI.value = document.getElementById("i").value;
    modalN.value = document.getElementById("n").value;
    settingsModal.classList.remove("hidden");
};

closeSettings.onclick = () => {
    settingsModal.classList.add("hidden");
};

settingsForm.addEventListener("submit", (event) => {
    event.preventDefault();

    document.getElementById("i").value = modalI.value;
    document.getElementById("n").value = modalN.value;

    settingsModal.classList.add("hidden");
    form.dispatchEvent(new Event("submit"));
});
