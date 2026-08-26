const API = "http://127.0.0.1:8000";

// Global Chart & Timer State
let fitnessChart = null;
let statusInterval = null;
let availableFeatures = [];

// DOM Elements
const generateBtn = document.getElementById("generateBtn");
const runBtn = document.getElementById("runBtn");

const statusBadge = document.getElementById("statusBadge");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const genText = document.getElementById("genText");
const liveFitness = document.getElementById("liveFitness");
const datasetStatus = document.getElementById("datasetStatus");

const solutionPlaceholder = document.getElementById("solutionPlaceholder");
const solutionContent = document.getElementById("solutionContent");
const resFitness = document.getElementById("resFitness");
const resAlpha = document.getElementById("resAlpha");
const resL1 = document.getElementById("resL1");
const resCount = document.getElementById("resCount");
const featureChips = document.getElementById("featureChips");

const datasetPlaceholder = document.getElementById("datasetPlaceholder");
const tableWrapper = document.getElementById("tableWrapper");
const tableHeader = document.getElementById("tableHeader");
const tableBody = document.getElementById("tableBody");
const datasetBadge = document.getElementById("datasetBadge");
const toast = document.getElementById("toast");

// --------------------------------------------------
// Helper Toast Notification
// --------------------------------------------------
function showToast(message, type = "success") {
    toast.textContent = message;
    toast.className = `toast toast-${type}`;
    setTimeout(() => {
        toast.className = "toast hidden";
    }, 4000);
}

// --------------------------------------------------
// 1. Generate Synthetic Dataset
// --------------------------------------------------
async function generateDataset() {
    const numSamples = document.getElementById("numSamples").value || 500;

    generateBtn.disabled = true;
    datasetStatus.textContent = "Generating...";

    try {
        const response = await fetch(`${API}/generate-data?samples=${numSamples}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to generate dataset");
        }

        const data = await response.json();

        datasetStatus.textContent = `${data.samples} Students`;
        datasetStatus.className = "stat-value text-success";
        runBtn.disabled = false;
        showToast(`Successfully generated ${data.samples} student records!`, "success");

        await loadDatasetPreview();
    } catch (err) {
        showToast(err.message, "error");
        datasetStatus.textContent = "Error";
    } finally {
        generateBtn.disabled = false;
    }
}

// --------------------------------------------------
// 2. Render Dataset Table Preview
// --------------------------------------------------
async function loadDatasetPreview() {
    try {
        const response = await fetch(`${API}/dataset?limit=15`);
        const result = await response.json();

        if (!result.data || result.data.length === 0) return;

        datasetBadge.textContent = `${result.samples} Records`;
        availableFeatures = result.columns.filter(c => c !== "Final_Marks");

        // Render Table Headers
        tableHeader.innerHTML = result.columns.map(col => {
            const isTarget = col === "Final_Marks";
            return `<th class="${isTarget ? 'target-col' : ''}">${col} ${isTarget ? '(Target)' : ''}</th>`;
        }).join("");

        // Render Table Rows
        tableBody.innerHTML = result.data.map(row => {
            return `<tr>${result.columns.map(col => {
                const isTarget = col === "Final_Marks";
                return `<td class="${isTarget ? 'target-col' : ''}">${row[col]}</td>`;
            }).join("")}</tr>`;
        }).join("");

        datasetPlaceholder.classList.add("hidden");
        tableWrapper.classList.remove("hidden");
    } catch (err) {
        console.error("Failed to load dataset preview:", err);
    }
}

// --------------------------------------------------
// 3. Start Genetic Algorithm Optimization
// --------------------------------------------------
async function runOptimization() {
    const generations = parseInt(document.getElementById("generations").value) || 40;
    const population_size = parseInt(document.getElementById("populationSize").value) || 30;
    const mutation_rate = parseFloat(document.getElementById("mutationRate").value) || 0.10;
    const crossover_rate = parseFloat(document.getElementById("crossoverRate").value) || 0.80;
    const gamma = parseFloat(document.getElementById("gammaPenalty").value) || 0.02;

    const payload = {
        generations,
        population_size,
        mutation_rate,
        crossover_rate,
        gamma
    };

    runBtn.disabled = true;
    generateBtn.disabled = true;

    statusBadge.textContent = "Running";
    statusBadge.className = "status-badge status-running";

    try {
        const response = await fetch(`${API}/run-ga`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to start optimization");
        }

        showToast("Genetic Algorithm optimization started!", "success");
        startStatusPolling();
    } catch (err) {
        showToast(err.message, "error");
        runBtn.disabled = false;
        generateBtn.disabled = false;
        statusBadge.textContent = "Failed";
        statusBadge.className = "status-badge status-idle";
    }
}

// --------------------------------------------------
// 4. Poll Status & Update Progress Bar
// --------------------------------------------------
function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);

    statusInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API}/status`);
            const data = await response.json();

            const pct = data.progress_percent || 0;
            progressBar.style.width = `${pct}%`;
            progressText.textContent = `${pct.toFixed(0)}% Complete`;
            genText.textContent = `Gen ${data.current_generation} / ${data.total_generations}`;

            if (data.current_best_fitness !== null && data.current_best_fitness !== undefined) {
                liveFitness.textContent = data.current_best_fitness.toFixed(4);
            }

            if (!data.running) {
                clearInterval(statusInterval);
                statusInterval = null;

                progressBar.style.width = "100%";
                progressText.textContent = "100% Complete";
                statusBadge.textContent = "Completed";
                statusBadge.className = "status-badge status-complete";

                runBtn.disabled = false;
                generateBtn.disabled = false;

                showToast("Optimization finished!", "success");
                await loadBestResult();
            }
        } catch (err) {
            console.error("Error checking GA status:", err);
        }
    }, 400);
}

// --------------------------------------------------
// 5. Load & Display Best Solution
// --------------------------------------------------
async function loadBestResult() {
    try {
        const response = await fetch(`${API}/best`);
        const best = await response.json();

        if (best.error) {
            showToast(best.error, "error");
            return;
        }

        resFitness.textContent = best.fitness.toFixed(4);
        resAlpha.textContent = best.alpha < 0.001 ? best.alpha.toExponential(3) : best.alpha.toFixed(4);
        resL1.textContent = best.l1_ratio.toFixed(4);
        resCount.textContent = `${best.num_selected} / ${best.total_features}`;

        // Render Feature Chips
        const selectedSet = new Set(best.selected_features || []);
        const allFeatures = availableFeatures.length > 0 ? availableFeatures : best.selected_features;

        featureChips.innerHTML = allFeatures.map(feat => {
            const isSel = selectedSet.has(feat);
            return `<span class="chip ${isSel ? 'chip-selected' : 'chip-rejected'}">
                <i class="fa-solid ${isSel ? 'fa-check' : 'fa-xmark'}"></i> ${feat}
            </span>`;
        }).join("");

        solutionPlaceholder.classList.add("hidden");
        solutionContent.classList.remove("hidden");

        await loadHistoryChart();
    } catch (err) {
        showToast("Failed to load best solution: " + err.message, "error");
    }
}

// --------------------------------------------------
// 6. Fitness History Convergence Chart
// --------------------------------------------------
async function loadHistoryChart() {
    try {
        const response = await fetch(`${API}/history`);
        const history = await response.json();

        if (!history || history.length === 0) return;

        const ctx = document.getElementById("fitnessChart").getContext("2d");

        // Destroy existing chart instance to prevent canvas reuse error
        if (fitnessChart) {
            fitnessChart.destroy();
        }

        // Gradient for chart background
        const gradient = ctx.createLinearGradient(0, 0, 0, 250);
        gradient.addColorStop(0, "rgba(79, 70, 229, 0.4)");
        gradient.addColorStop(1, "rgba(79, 70, 229, 0.0)");

        fitnessChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: history.map((_, i) => `Gen ${i + 1}`),
                datasets: [{
                    label: "Best Generation Fitness (R² - Penalty)",
                    data: history,
                    borderColor: "#6366f1",
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: "#818cf8",
                    pointRadius: 3,
                    pointHoverRadius: 6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } }
                    },
                    tooltip: {
                        backgroundColor: "#0f172a",
                        titleColor: "#f8fafc",
                        bodyColor: "#818cf8",
                        borderColor: "rgba(255, 255, 255, 0.1)",
                        borderWidth: 1,
                    }
                },
                scales: {
                    x: {
                        ticks: { color: "#64748b", font: { family: "Inter", size: 11 } },
                        grid: { color: "rgba(255, 255, 255, 0.04)" }
                    },
                    y: {
                        ticks: { color: "#64748b", font: { family: "Inter", size: 11 } },
                        grid: { color: "rgba(255, 255, 255, 0.04)" }
                    }
                }
            }
        });
    } catch (err) {
        console.error("Failed to render history chart:", err);
    }
}

// --------------------------------------------------
// 7. Initial App Sync on Page Load
// --------------------------------------------------
async function initApp() {
    try {
        // Sync Dataset
        const dsResp = await fetch(`${API}/dataset?limit=15`);
        if (dsResp.ok) {
            const ds = await dsResp.json();
            if (ds.samples > 0) {
                datasetStatus.textContent = `${ds.samples} Students`;
                datasetStatus.className = "stat-value text-success";
                runBtn.disabled = false;
                await loadDatasetPreview();
            }
        }

        // Sync GA Status
        const stResp = await fetch(`${API}/status`);
        if (stResp.ok) {
            const st = await stResp.json();
            if (st.running) {
                runBtn.disabled = true;
                generateBtn.disabled = true;
                statusBadge.textContent = "Running";
                statusBadge.className = "status-badge status-running";
                startStatusPolling();
            } else if (st.progress_percent === 100) {
                progressBar.style.width = "100%";
                progressText.textContent = "100% Complete";
                genText.textContent = `Gen ${st.total_generations} / ${st.total_generations}`;
                if (st.current_best_fitness !== null) {
                    liveFitness.textContent = st.current_best_fitness.toFixed(4);
                }
                statusBadge.textContent = "Completed";
                statusBadge.className = "status-badge status-complete";
            }
        }

        // Sync Best Result & Chart
        const bestResp = await fetch(`${API}/best`);
        if (bestResp.ok) {
            const best = await bestResp.json();
            if (!best.error) {
                await loadBestResult();
            }
        }
    } catch (err) {
        console.error("Initialization sync error:", err);
    }
}

// --------------------------------------------------
// Event Listeners
// --------------------------------------------------
generateBtn.addEventListener("click", generateDataset);
runBtn.addEventListener("click", runOptimization);
document.addEventListener("DOMContentLoaded", initApp);