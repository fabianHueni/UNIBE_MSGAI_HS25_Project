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
 * @param text - Text to log
 */
export function logTo(el, text) {
    if (!el) return;
    const p = document.createElement('div');
    p.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    el.appendChild(p);
    el.scrollTop = el.scrollHeight;
}