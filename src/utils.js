// helpers: timing and small utilities
export function nowMs() {
    return performance.now();
}

export function sleep(ms) {
    return new Promise(res => setTimeout(res, ms));
}


export function measureAsync(fn) {
    const start = nowMs();
    return Promise.resolve()
        .then(() => fn())
        .then(res => ({res, ms: nowMs() - start}));
}


/**
 * Log text to a given HTML element with timestamp to show the log in the UI
 *
 * @param el - HTML element to log to
 * @param evt - Event object with job, route, latency, response, and timing metrics
 */
export function logTo(el, evt) {
    if (!el) return;
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${new Date().toLocaleTimeString()}</td>
        <td>${evt.route}</td>
        <td>${evt.totalLatency?.toFixed(2) || evt.latency?.toFixed(2) || 0}ms</td>
        <td>${evt.queueingTime?.toFixed(2) || 0}ms</td>
        <td>${evt.inferenceTime?.toFixed(2) || evt.latency?.toFixed(2) || 0}ms</td>
        <td title="${evt.job.prompt}">${evt.job.prompt.substring(0, 30)}...</td>
        <td title="${evt.response.answer}">${evt.response.answer.substring(0, 30)}</td>
        <td>${evt.evalRes.exactMatch}</td>
    `;
    el.appendChild(row);
    el.scrollTop = el.scrollHeight;
}