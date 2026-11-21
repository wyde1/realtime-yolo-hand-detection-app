(() => {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const output = document.getElementById("output");
    const webcamEmpty = document.getElementById("webcam-empty");

    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const statusPill = document.getElementById("connection-status");
    const statusText = document.getElementById("connection-status-text");
    const fpsCounter = document.getElementById("fps-counter");
    const latencyCounter = document.getElementById("latency-counter");
    const modelWarning = document.getElementById("model-warning");
    const modelWarningText = document.getElementById("model-warning-text");
    const modelBadge = document.getElementById("model-badge");
    const inlineError = document.getElementById("inline-error");

    const resolutionSelect = document.getElementById("resolution-select");
    const qualitySlider = document.getElementById("quality-slider");
    const qualityValue = document.getElementById("quality-value");
    const confidenceSlider = document.getElementById("confidence-slider");
    const confidenceValue = document.getElementById("confidence-value");
    const fpsLimitSelect = document.getElementById("fps-limit");

    const tabWebcam = document.getElementById("tab-webcam");
    const tabUpload = document.getElementById("tab-upload");
    const panelWebcam = document.getElementById("panel-webcam");
    const panelUpload = document.getElementById("panel-upload");
    const webcamActions = document.getElementById("webcam-actions");
    const uploadActions = document.getElementById("upload-actions");
    const webcamOnlyControls = document.getElementById("webcam-only-controls");
    const stageTitle = document.getElementById("stage-title");
    const stageMeta = document.getElementById("stage-meta");
    const endpointTag = document.getElementById("endpoint-tag");

    const uploadInput = document.getElementById("upload-input");
    const uploadDropzone = document.getElementById("upload-dropzone");
    const uploadBrowseBtn = document.getElementById("upload-browse-btn");
    const uploadTriggerBtn = document.getElementById("upload-trigger-btn");
    const uploadRerunBtn = document.getElementById("upload-rerun-btn");
    const uploadOutput = document.getElementById("upload-output");

    const digitEmpty = document.getElementById("digit-empty");
    const detectionList = document.getElementById("detection-list");

    const ctx = canvas.getContext("2d");

    let activeMode = "webcam";
    let stream = null;
    let socket = null;
    let isConnected = false;
    let isStreaming = false;
    let isProcessing = false;
    let animationFrameId = null;
    let cameraSessionId = 0;

    let frameCount = 0;
    let lastFpsTimestamp = performance.now();
    let lastSentTimestamp = 0;
    let fpsLimit = parseInt(fpsLimitSelect.value, 10);
    let frameInterval = fpsLimit > 0 ? 1000 / fpsLimit : 0;

    let selectedFile = null;
    let uploadObjectUrl = null;
    let uploadAbortController = null;
    let uploadRequestId = 0;

    function setStatus(state, label) {
        statusPill.dataset.state = state;
        statusText.textContent = label;
    }

    function showError(message) {
        inlineError.textContent = message;
        inlineError.classList.remove("hidden");
    }

    function clearError() {
        inlineError.textContent = "";
        inlineError.classList.add("hidden");
    }

    function stopTracks(mediaStream) {
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
        }
    }

    function updateStageMeta() {
        if (activeMode === "webcam") {
            const [w, h] = resolutionSelect.value.split("x");
            stageTitle.textContent = "Live WebSocket Feed";
            stageMeta.textContent = `${w} \u00d7 ${h} \u00b7 WS /ws`;
            endpointTag.textContent = "WS /ws";
        } else {
            stageTitle.textContent = "Static Image Inference";
            stageMeta.textContent = selectedFile ? selectedFile.name : "POST /detect";
            endpointTag.textContent = "POST /detect";
        }
    }

    function renderDetections(detections, emptyCaption = "No hand detected") {
        detectionList.innerHTML = "";

        if (!detections || detections.length === 0) {
            detectionList.classList.add("hidden");
            digitEmpty.classList.remove("hidden");
            digitEmpty.querySelector(".digit-caption").textContent = emptyCaption;
            return;
        }

        digitEmpty.classList.add("hidden");
        detectionList.classList.remove("hidden");

        for (const det of detections) {
            const pct = Math.round((det.confidence || 0) * 100);
            const item = document.createElement("div");
            item.className = "detection-item";

            const tile = document.createElement("div");
            tile.className = "digit-tile";
            tile.textContent = det.label;

            const meta = document.createElement("div");
            meta.className = "detection-meta";

            const row = document.createElement("div");
            row.className = "detection-row";

            const nameSpan = document.createElement("span");
            nameSpan.textContent = `Class "${det.label}"`;

            const confSpan = document.createElement("span");
            confSpan.className = "detection-conf";
            confSpan.textContent = `${pct}%`;

            row.append(nameSpan, confSpan);

            const track = document.createElement("div");
            track.className = "conf-bar-track";

            const fill = document.createElement("div");
            fill.className = "conf-bar-fill";
            fill.style.width = `${Math.max(4, Math.min(100, pct))}%`;
            track.appendChild(fill);

            meta.append(row, track);
            item.append(tile, meta);
            detectionList.appendChild(item);
        }
    }

    async function checkHealth() {
        try {
            const response = await fetch("/health");
            if (!response.ok) {
                throw new Error(`Health check returned ${response.status}`);
            }
            const data = await response.json();
            if (typeof data.default_conf === "number") {
                const clamped = Math.max(0.05, Math.min(0.95, data.default_conf));
                confidenceSlider.value = clamped.toFixed(2);
                confidenceValue.textContent = clamped.toFixed(2);
            }
            if (data.model_loaded) {
                modelBadge.textContent = `YOLO11s (${data.classes.length} classes)`;
                modelBadge.dataset.ready = "true";
                modelWarning.classList.add("hidden");
            } else {
                modelBadge.textContent = "Weights missing";
                modelBadge.dataset.ready = "false";
                modelWarning.classList.remove("hidden");
                if (data.detail) {
                    modelWarningText.textContent = data.detail;
                }
            }
        } catch {
            modelBadge.textContent = "Server unreachable";
            modelBadge.dataset.ready = "false";
            showError("Could not reach /health endpoint. Verify the FastAPI server is running.");
        }
    }

    function switchMode(mode) {
        if (mode === activeMode) {
            return;
        }
        clearError();
        if (mode === "upload") {
            // Unconditionally cancel any active or pending camera startup session.
            stopCamera();
        } else {
            if (uploadAbortController) {
                uploadAbortController.abort();
                uploadAbortController = null;
            }
            setStatus("idle", "Idle");
            fpsCounter.textContent = "--";
            latencyCounter.textContent = "-- ms";
            renderDetections([], "Awaiting frame");
        }
        activeMode = mode;

        const isWebcam = mode === "webcam";
        tabWebcam.classList.toggle("active", isWebcam);
        tabWebcam.setAttribute("aria-selected", String(isWebcam));
        tabUpload.classList.toggle("active", !isWebcam);
        tabUpload.setAttribute("aria-selected", String(!isWebcam));

        panelWebcam.classList.toggle("hidden", !isWebcam);
        panelUpload.classList.toggle("hidden", isWebcam);
        webcamActions.classList.toggle("hidden", !isWebcam);
        uploadActions.classList.toggle("hidden", isWebcam);
        webcamOnlyControls.classList.toggle("hidden", !isWebcam);

        updateStageMeta();
    }

    function detachAndCloseSocket() {
        if (!socket) {
            return;
        }
        const oldSocket = socket;
        socket = null;
        oldSocket.onopen = null;
        oldSocket.onmessage = null;
        oldSocket.onclose = null;
        oldSocket.onerror = null;
        if (oldSocket.readyState === WebSocket.OPEN || oldSocket.readyState === WebSocket.CONNECTING) {
            oldSocket.close();
        }
    }

    function connectWebSocket() {
        detachAndCloseSocket();

        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        const ws = new WebSocket(wsUrl);
        socket = ws;

        ws.onopen = () => {
            if (ws !== socket) {
                return;
            }
            isConnected = true;
            isProcessing = false;
            frameCount = 0;
            lastFpsTimestamp = performance.now();
            setStatus("connected", "Live WS");
            clearError();
        };

        ws.onmessage = (event) => {
            if (ws !== socket) {
                return;
            }
            isProcessing = false;
            if (!isStreaming) {
                return;
            }

            let payload;
            try {
                payload = JSON.parse(event.data);
            } catch {
                output.src = event.data;
                output.classList.remove("hidden");
                video.classList.remove("preview-visible");
                return;
            }

            if (payload.error) {
                showError(payload.error);
                return;
            }

            if (payload.image) {
                output.src = payload.image;
                output.classList.remove("hidden");
                video.classList.remove("preview-visible");
            }

            if (typeof payload.inference_ms === "number") {
                latencyCounter.textContent = `${payload.inference_ms.toFixed(1)} ms`;
            }

            renderDetections(payload.detections || [], "No hand in frame");

            frameCount += 1;
            const now = performance.now();
            if (now - lastFpsTimestamp >= 1000) {
                const elapsedSec = (now - lastFpsTimestamp) / 1000;
                fpsCounter.textContent = (frameCount / elapsedSec).toFixed(1);
                frameCount = 0;
                lastFpsTimestamp = now;
            }
        };

        ws.onclose = () => {
            if (ws !== socket) {
                return;
            }
            isConnected = false;
            isProcessing = false;
            if (isStreaming) {
                setStatus("disconnected", "Disconnected");
            } else {
                setStatus("idle", "Idle");
            }
        };

        ws.onerror = () => {
            if (ws !== socket) {
                return;
            }
            isConnected = false;
            isProcessing = false;
            setStatus("error", "WS Error");
            showError("WebSocket connection error. Check that the FastAPI backend is running.");
        };
    }

    function sendFrameLoop() {
        if (!isStreaming) {
            animationFrameId = null;
            return;
        }

        animationFrameId = requestAnimationFrame(sendFrameLoop);

        const now = performance.now();
        if (frameInterval > 0 && now - lastSentTimestamp < frameInterval) {
            return;
        }

        if (
            video.readyState >= video.HAVE_CURRENT_DATA &&
            video.videoWidth > 0 &&
            isConnected &&
            socket &&
            socket.readyState === WebSocket.OPEN &&
            !isProcessing
        ) {
            lastSentTimestamp = now;
            isProcessing = true;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const quality = parseInt(qualitySlider.value, 10);
            const conf = parseFloat(confidenceSlider.value);
            const dataUri = canvas.toDataURL("image/jpeg", quality / 100);

            try {
                socket.send(
                    JSON.stringify({
                        image: dataUri,
                        conf,
                        quality,
                    })
                );
            } catch {
                isProcessing = false;
            }
        }
    }

    function getVideoConstraints() {
        const [width, height] = resolutionSelect.value.split("x").map(Number);
        return {
            facingMode: { ideal: "user" },
            width: { ideal: width },
            height: { ideal: height },
        };
    }

    async function acquireCameraStream() {
        try {
            return await navigator.mediaDevices.getUserMedia({
                video: getVideoConstraints(),
                audio: false,
            });
        } catch (err) {
            if (err && (err.name === "OverconstrainedError" || err.name === "NotFoundError")) {
                return await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false,
                });
            }
            throw err;
        }
    }

    async function startCamera() {
        if (isStreaming || activeMode !== "webcam") {
            return;
        }
        clearError();

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError("Webcam API is unavailable in this browser context. Use Upload Image instead.");
            return;
        }

        const sessionId = ++cameraSessionId;
        startBtn.disabled = true;
        setStatus("idle", "Starting\u2026");

        let acquiredStream = null;
        try {
            acquiredStream = await acquireCameraStream();
            if (sessionId !== cameraSessionId || activeMode !== "webcam") {
                stopTracks(acquiredStream);
                return;
            }

            stream = acquiredStream;
            video.srcObject = acquiredStream;
            await video.play();

            if (sessionId !== cameraSessionId || activeMode !== "webcam") {
                stopTracks(acquiredStream);
                if (stream === acquiredStream) {
                    stream = null;
                    video.srcObject = null;
                }
                return;
            }

            isStreaming = true;
            isProcessing = false;
            frameCount = 0;
            lastSentTimestamp = 0;
            lastFpsTimestamp = performance.now();

            webcamEmpty.classList.add("hidden");
            video.classList.add("preview-visible");
            output.classList.add("hidden");

            stopBtn.disabled = false;
            setStatus("idle", "Connecting\u2026");

            connectWebSocket();

            if (animationFrameId !== null) {
                cancelAnimationFrame(animationFrameId);
            }
            animationFrameId = requestAnimationFrame(sendFrameLoop);
        } catch (err) {
            stopTracks(acquiredStream);
            if (stream === acquiredStream) {
                stream = null;
                video.srcObject = null;
            }
            if (sessionId !== cameraSessionId) {
                return;
            }
            isStreaming = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            setStatus("error", "Camera Error");
            showError(`Unable to access webcam: ${err.message || "permission denied"}`);
        }
    }

    function stopCamera() {
        cameraSessionId += 1;
        isStreaming = false;
        isProcessing = false;
        isConnected = false;

        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }

        stopTracks(stream);
        stream = null;
        video.srcObject = null;
        video.classList.remove("preview-visible");

        detachAndCloseSocket();

        startBtn.disabled = false;
        stopBtn.disabled = true;
        output.classList.add("hidden");
        output.removeAttribute("src");
        webcamEmpty.classList.remove("hidden");

        fpsCounter.textContent = "--";
        latencyCounter.textContent = "-- ms";
        setStatus("idle", "Idle");
        renderDetections([], "Awaiting frame");
    }

    async function extractErrorDetail(response) {
        let errDetail = `Detection failed (${response.status})`;
        try {
            const errJson = await response.json();
            if (errJson && errJson.detail) {
                errDetail = errJson.detail;
            }
        } catch {
            // ignore JSON parse failure on non-JSON error
        }
        return errDetail;
    }

    async function runUploadDetection(file) {
        if (!file) {
            return;
        }
        clearError();
        selectedFile = file;
        uploadRerunBtn.disabled = false;
        updateStageMeta();
        setStatus("idle", "Inferring\u2026");

        if (uploadAbortController) {
            uploadAbortController.abort();
        }
        const controller = new AbortController();
        uploadAbortController = controller;
        const reqId = ++uploadRequestId;

        const conf = parseFloat(confidenceSlider.value).toFixed(2);
        const imgForm = new FormData();
        imgForm.append("file", file);
        const jsonForm = new FormData();
        jsonForm.append("file", file);

        try {
            const [imgResp, jsonResp] = await Promise.all([
                fetch(`/detect?conf=${encodeURIComponent(conf)}`, {
                    method: "POST",
                    body: imgForm,
                    signal: controller.signal,
                }),
                fetch(`/detect/json?conf=${encodeURIComponent(conf)}`, {
                    method: "POST",
                    body: jsonForm,
                    signal: controller.signal,
                }),
            ]);

            if (reqId !== uploadRequestId) {
                return;
            }

            if (!imgResp.ok) {
                throw new Error(await extractErrorDetail(imgResp));
            }

            const blob = await imgResp.blob();
            if (reqId !== uploadRequestId) {
                return;
            }

            if (uploadObjectUrl) {
                URL.revokeObjectURL(uploadObjectUrl);
            }
            uploadObjectUrl = URL.createObjectURL(blob);
            uploadOutput.src = uploadObjectUrl;
            uploadOutput.classList.remove("hidden");
            uploadDropzone.classList.add("hidden");

            if (!jsonResp.ok) {
                renderDetections([], "Metadata unavailable");
                showError(await extractErrorDetail(jsonResp));
                setStatus("error", "Partial");
                return;
            }

            const jsonData = await jsonResp.json();
            if (reqId !== uploadRequestId) {
                return;
            }

            if (typeof jsonData.inference_ms === "number") {
                latencyCounter.textContent = `${jsonData.inference_ms.toFixed(1)} ms`;
            }
            renderDetections(jsonData.detections || [], "No hand detected in image");
            setStatus("connected", "Ready");
        } catch (err) {
            if (err && err.name === "AbortError") {
                return;
            }
            if (reqId !== uploadRequestId) {
                return;
            }
            renderDetections([], "Detection failed");
            setStatus("error", "Error");
            showError(err.message || "Failed to process uploaded image.");
        }
    }

    // Event bindings
    tabWebcam.addEventListener("click", () => switchMode("webcam"));
    tabUpload.addEventListener("click", () => switchMode("upload"));

    [tabWebcam, tabUpload].forEach((tab) => {
        tab.addEventListener("keydown", (event) => {
            if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
                event.preventDefault();
                const nextMode = activeMode === "webcam" ? "upload" : "webcam";
                switchMode(nextMode);
                (nextMode === "webcam" ? tabWebcam : tabUpload).focus();
            }
        });
    });

    startBtn.addEventListener("click", startCamera);
    stopBtn.addEventListener("click", stopCamera);

    qualitySlider.addEventListener("input", () => {
        qualityValue.textContent = `${qualitySlider.value}%`;
    });

    confidenceSlider.addEventListener("input", () => {
        confidenceValue.textContent = Number(confidenceSlider.value).toFixed(2);
    });

    fpsLimitSelect.addEventListener("change", () => {
        fpsLimit = parseInt(fpsLimitSelect.value, 10);
        frameInterval = fpsLimit > 0 ? 1000 / fpsLimit : 0;
    });

    resolutionSelect.addEventListener("change", async () => {
        updateStageMeta();
        if (isStreaming) {
            stopCamera();
            await startCamera();
        }
    });

    uploadBrowseBtn.addEventListener("click", () => uploadInput.click());
    uploadTriggerBtn.addEventListener("click", () => uploadInput.click());
    uploadRerunBtn.addEventListener("click", () => {
        if (selectedFile) {
            runUploadDetection(selectedFile);
        }
    });

    uploadInput.addEventListener("change", () => {
        const file = uploadInput.files && uploadInput.files[0];
        uploadInput.value = "";
        if (file) {
            runUploadDetection(file);
        }
    });

    // Bind drag-and-drop on panelUpload so subsequent images can also be dropped.
    panelUpload.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadDropzone.classList.add("dragover");
    });

    panelUpload.addEventListener("dragleave", (event) => {
        if (!panelUpload.contains(event.relatedTarget)) {
            uploadDropzone.classList.remove("dragover");
        }
    });

    panelUpload.addEventListener("drop", (event) => {
        event.preventDefault();
        uploadDropzone.classList.remove("dragover");
        const file = event.dataTransfer && event.dataTransfer.files[0];
        if (file) {
            runUploadDetection(file);
        }
    });

    window.addEventListener("beforeunload", () => {
        stopCamera();
        if (uploadAbortController) {
            uploadAbortController.abort();
        }
        if (uploadObjectUrl) {
            URL.revokeObjectURL(uploadObjectUrl);
        }
    });

    updateStageMeta();
    checkHealth();
})();
