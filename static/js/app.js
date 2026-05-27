/* ═══════════════════════════════════════════
   SmartBin Dashboard — Interactive Logic
   ═══════════════════════════════════════════ */

(function () {
    "use strict";

    // ─── DOM References ───
    const $ = (sel) => document.querySelector(sel);
    const videoFeed    = $("#video-feed");
    const videoHolder  = $("#video-placeholder");
    const btnStart     = $("#btn-start");
    const btnStop      = $("#btn-stop");
    const statusBadge  = $("#status-badge");
    const statusText   = $("#status-text");
    const totalCount   = $("#total-count");
    const dryCount     = $("#dry-count");
    const wetCount     = $("#wet-count");
    const metalCount   = $("#metal-count");
    const lastTime     = $("#last-detection-time");
    const tbody        = $("#detection-tbody");
    const donutCanvas  = $("#donut-chart");
    const chartLegend  = $("#chart-legend");
    const droidcamUrl  = $("#droidcam-url");
    const confSlider   = $("#conf-slider");
    const confValue    = $("#conf-value");
    const useWebcam    = $("#use-webcam");
    const btnSave      = $("#btn-save-settings");
    const btnClear     = $("#btn-clear-history");

    // EcoChat
    const ecoChatPanel   = $("#ecochat-panel");
    const ecoChatOverlay = $("#ecochat-overlay");
    const ecoChatBody    = $("#ecochat-body");
    const ecoChatInput   = $("#ecochat-input");
    const btnChatToggle  = $("#btn-ecochat-toggle");
    const btnChatClose   = $("#btn-ecochat-close");
    const btnChatSend    = $("#btn-ecochat-send");

    const toastContainer = $("#toast-container");

    let isStreaming = false;
    let pollInterval = null;
    let prevTotal = 0;

    // ─── Confidence slider ───
    confSlider.addEventListener("input", () => {
        confValue.textContent = Math.round(confSlider.value * 100) + "%";
    });

    // ─── Stream Controls ───
    btnStart.addEventListener("click", startStream);
    btnStop.addEventListener("click", stopStream);

    function startStream() {
        // Save settings first, then connect
        saveSettings().then(() => {
            videoFeed.src = "/video_feed?" + Date.now();
            videoFeed.style.display = "block";
            videoHolder.style.display = "none";
            isStreaming = true;
            btnStart.disabled = true;
            btnStop.disabled = false;
            setStatus(true);
            showToast("🎥", "Stream started — YOLO detection active");
            startPolling();
        });
    }

    function stopStream() {
        videoFeed.src = "";
        videoFeed.style.display = "none";
        videoHolder.style.display = "flex";
        isStreaming = false;
        btnStart.disabled = false;
        btnStop.disabled = true;
        setStatus(false);
        stopPolling();
        showToast("⏹", "Stream stopped");
    }

    function setStatus(online) {
        statusBadge.className = "status-badge " + (online ? "online" : "offline");
        statusText.textContent = online ? "Live" : "Offline";
    }

    // ─── Settings ───
    btnSave.addEventListener("click", () => {
        saveSettings().then(() => showToast("✅", "Settings saved"));
    });

    function saveSettings() {
        return fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                droidcam_url: droidcamUrl.value.trim(),
                conf_threshold: parseFloat(confSlider.value),
                use_webcam: useWebcam.checked,
            }),
        }).catch(() => {});
    }

    // ─── Polling ───
    function startPolling() {
        if (pollInterval) return;
        fetchData();
        pollInterval = setInterval(fetchData, 2000);
    }

    function stopPolling() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }

    function fetchData() {
        Promise.all([
            fetch("/api/detections?limit=20").then(r => r.json()).catch(() => []),
            fetch("/api/stats").then(r => r.json()).catch(() => ({ total: 0, categories: {}, last_time: "--:--:--" })),
        ]).then(([detections, stats]) => {
            updateTable(detections);
            updateStats(stats);
            updateChart(stats.categories);

            // Toast on new detections
            if (stats.total > prevTotal && prevTotal > 0) {
                const newest = detections[0];
                if (newest) {
                    showToast("🎯", `Detected: ${newest.item} (${newest.category})`);
                }
            }
            prevTotal = stats.total;
        });
    }

    // ─── Table ───
    function updateTable(detections) {
        if (!detections.length) return;

        tbody.innerHTML = "";
        detections.forEach((d, i) => {
            const tr = document.createElement("tr");
            if (i === 0) tr.classList.add("row-new");

            const catClass = d.category.toLowerCase();
            tr.innerHTML = `
                <td style="color:var(--text-primary);font-weight:500;">${escHtml(d.item)}</td>
                <td><span class="cat-badge ${catClass}">${d.category}</span></td>
                <td>
                    <span class="degrad-badge ${d.degradability === 'Biodegradable' ? 'bio' : 'non-bio'}">
                        ${d.degradability}
                    </span>
                </td>
                <td>
                    <span class="conf-bar">
                        <span class="conf-bar-bg"><span class="conf-bar-fill" style="width:${d.confidence}%"></span></span>
                        ${d.confidence}%
                    </span>
                </td>
                <td style="font-variant-numeric:tabular-nums;color:var(--text-muted);">${d.time}</td>
                <td>
                    <button class="btn btn-xs recycle-tip-btn" title="Get recycling tip" data-item="${escHtml(d.item)}" data-bin="${escHtml(d.category)}" data-deg="${escHtml(d.degradability)}">♻️ Tip</button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        // Attach recycle tip button listeners
        tbody.querySelectorAll(".recycle-tip-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                sendRecycleTip(btn.dataset.item, btn.dataset.bin, btn.dataset.deg);
            });
        });
    }

    // ─── Stats ───
    function updateStats(stats) {
        animateValue(totalCount, stats.total);
        animateValue(dryCount, stats.categories["Dry"] || 0);
        animateValue(wetCount, stats.categories["Wet"] || 0);
        animateValue(metalCount, stats.categories["Metal"] || 0);
        lastTime.textContent = stats.last_time;
    }

    function animateValue(el, newVal) {
        const cur = parseInt(el.textContent) || 0;
        if (cur !== newVal) {
            el.textContent = newVal;
            el.classList.remove("pop");
            void el.offsetWidth; // trigger reflow
            el.classList.add("pop");
        }
    }

    // ─── Donut Chart ───
    const ctx = donutCanvas.getContext("2d");
    const chartColors = {
        "Dry":   "#3b82f6",
        "Wet":   "#10b981",
        "Metal": "#f59e0b",
    };

    let currentAngles = {}; // for smooth transition

    function updateChart(categories) {
        const entries = Object.entries(categories);
        const total = entries.reduce((s, [, v]) => s + v, 0) || 1;

        // Clear canvas
        const W = donutCanvas.width;
        const H = donutCanvas.height;
        const cx = W / 2;
        const cy = H / 2;
        const outerR = Math.min(W, H) / 2 - 10;
        const innerR = outerR * 0.62;

        ctx.clearRect(0, 0, W, H);

        if (entries.length === 0) {
            // Empty state
            ctx.beginPath();
            ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
            ctx.arc(cx, cy, innerR, 0, Math.PI * 2, true);
            ctx.fillStyle = "rgba(255,255,255,0.04)";
            ctx.fill();

            ctx.fillStyle = "rgba(255,255,255,0.15)";
            ctx.font = "600 14px Inter";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("No data", cx, cy);

            chartLegend.innerHTML = "";
            return;
        }

        let startAngle = -Math.PI / 2;
        entries.forEach(([cat, count]) => {
            const sweep = (count / total) * Math.PI * 2;
            const endAngle = startAngle + sweep;
            const color = chartColors[cat] || "#8b5cf6";

            ctx.beginPath();
            ctx.arc(cx, cy, outerR, startAngle, endAngle);
            ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();

            // Subtle gap between segments
            ctx.strokeStyle = "rgba(6,11,20,0.8)";
            ctx.lineWidth = 2;
            ctx.stroke();

            startAngle = endAngle;
        });

        // Center text
        ctx.fillStyle = "#f0f4f8";
        ctx.font = "800 28px Outfit";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(total, cx, cy - 8);

        ctx.fillStyle = "rgba(255,255,255,0.35)";
        ctx.font = "600 10px Inter";
        ctx.fillText("DETECTIONS", cx, cy + 14);

        // Legend
        chartLegend.innerHTML = entries.map(([cat, count]) => {
            const pct = Math.round((count / total) * 100);
            const clr = chartColors[cat] || "#8b5cf6";
            return `
                <div class="legend-item">
                    <span class="legend-dot" style="background:${clr}"></span>
                    <span class="legend-label">${cat}</span>
                    <span class="legend-value">${count} <span style="font-weight:400;color:var(--text-muted);font-size:0.75rem;">(${pct}%)</span></span>
                </div>
            `;
        }).join("");
    }

    // Initialize empty chart
    updateChart({});

    // ─── Clear History ───
    btnClear.addEventListener("click", () => {
        fetch("/api/detections/clear", { method: "POST" })
            .then(() => {
                tbody.innerHTML = '<tr class="empty-row"><td colspan="6">No detections yet — start the live stream</td></tr>';
                totalCount.textContent = "0";
                dryCount.textContent = "0";
                wetCount.textContent = "0";
                metalCount.textContent = "0";
                prevTotal = 0;
                updateChart({});
                lastTime.textContent = "--:--:--";
                showToast("🗑️", "Detection history cleared");
            })
            .catch(() => {});
    });

    // ─── EcoChat ───
    function openChat() {
        ecoChatPanel.classList.add("open");
        ecoChatOverlay.classList.add("show");
        ecoChatInput.focus();
    }

    function closeChat() {
        ecoChatPanel.classList.remove("open");
        ecoChatOverlay.classList.remove("show");
    }

    btnChatToggle.addEventListener("click", openChat);
    btnChatClose.addEventListener("click", closeChat);
    ecoChatOverlay.addEventListener("click", closeChat);

    function addChatMessage(text, isUser) {
        const div = document.createElement("div");
        div.className = "chat-message " + (isUser ? "user" : "bot");
        div.innerHTML = `<div class="chat-bubble">${escHtml(text)}</div>`;
        ecoChatBody.appendChild(div);
        ecoChatBody.scrollTop = ecoChatBody.scrollHeight;
    }

    function addChatBotMessage(text) {
        const div = document.createElement("div");
        div.className = "chat-message bot";
        // Render basic markdown: **bold**, *italic*, bullet points, newlines
        let html = escHtml(text)
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.+?)\*/g, "<em>$1</em>")
            .replace(/^[-•]\s+/gm, "&#8226; ")
            .replace(/\n/g, "<br>");
        div.innerHTML = `<div class="chat-bubble">${html}</div>`;
        ecoChatBody.appendChild(div);
        ecoChatBody.scrollTop = ecoChatBody.scrollHeight;
    }

    function addTypingIndicator() {
        const div = document.createElement("div");
        div.className = "chat-message bot";
        div.id = "typing-indicator";
        div.innerHTML = `<div class="chat-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
        ecoChatBody.appendChild(div);
        ecoChatBody.scrollTop = ecoChatBody.scrollHeight;
    }

    function removeTypingIndicator() {
        const el = document.getElementById("typing-indicator");
        if (el) el.remove();
    }

    function sendChat() {
        const msg = ecoChatInput.value.trim();
        if (!msg) return;

        addChatMessage(msg, true);
        ecoChatInput.value = "";
        addTypingIndicator();

        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 90000);

        fetch("/api/ecochat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: msg }),
            signal: ctrl.signal,
        })
            .then(r => { clearTimeout(timer); return r.json(); })
            .then(data => {
                removeTypingIndicator();
                addChatBotMessage(data.reply || "⚠️ No response");
            })
            .catch(err => {
                clearTimeout(timer);
                removeTypingIndicator();
                if (err.name === 'AbortError') {
                    addChatBotMessage("⏳ EcoChat is waiting on rate limit. Please resend your message in ~30 seconds.");
                } else {
                    addChatBotMessage("⚠️ Could not reach EcoChat. Check server.");
                }
            });
    }

    btnChatSend.addEventListener("click", sendChat);
    ecoChatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendChat();
        }
    });

    // ─── Recycle Tip via EcoChat ───
    function sendRecycleTip(item, binType, degradability) {
        openChat();
        addChatMessage(`♻️ Give me recycling tips for: ${item}`, true);
        addTypingIndicator();

        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 90000); // 90s timeout

        fetch("/api/ecochat/recycle", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ item, bin_type: binType, degradability }),
            signal: ctrl.signal,
        })
            .then(r => { clearTimeout(timer); return r.json(); })
            .then(data => {
                removeTypingIndicator();
                addChatBotMessage(data.reply || "⚠️ No response");
            })
            .catch(err => {
                clearTimeout(timer);
                removeTypingIndicator();
                if (err.name === 'AbortError') {
                    addChatBotMessage("⚠️ Request timed out. The API may be rate-limited. Please try again in a moment.");
                } else {
                    addChatBotMessage("⚠️ Could not reach EcoChat. Check server.");
                }
            });
    }

    // ─── Toasts ───
    function showToast(icon, message) {
        const toast = document.createElement("div");
        toast.className = "toast";
        toast.innerHTML = `<span class="toast-icon">${icon}</span><span>${escHtml(message)}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add("removing");
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // ─── Helpers ───
    function escHtml(s) {
        const d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
    }

    // ─── Load initial settings ───
    fetch("/api/settings")
        .then(r => r.json())
        .then(s => {
            if (s.droidcam_url) droidcamUrl.value = s.droidcam_url;
            if (s.conf_threshold) {
                confSlider.value = s.conf_threshold;
                confValue.textContent = Math.round(s.conf_threshold * 100) + "%";
            }
            if (s.use_webcam) useWebcam.checked = true;
        })
        .catch(() => {});

})();
