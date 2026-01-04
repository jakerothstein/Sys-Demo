// Text-to-SQL Research Pipeline - Web UI Application

const API_BASE = '';
const SESSION_ID = 'session_' + Date.now();

// DOM Elements
const queryInput = document.getElementById('query-input');
const submitBtn = document.getElementById('submit-btn');
const statusSection = document.getElementById('status-section');
const statusIcon = document.getElementById('status-icon');
const statusText = document.getElementById('status-text');
const hitlSection = document.getElementById('hitl-section');
const hitlMessage = document.getElementById('hitl-message');
const hitlConfidence = document.getElementById('hitl-confidence');
const hitlReasoning = document.getElementById('hitl-reasoning');
const feedbackInput = document.getElementById('feedback-input');
const feedbackBtn = document.getElementById('feedback-btn');
const resultsSection = document.getElementById('results-section');
const sqlOutput = document.getElementById('sql-output').querySelector('code');
const dataOutput = document.getElementById('data-output');
const planOutput = document.getElementById('plan-output');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');

// Event Listeners
submitBtn.addEventListener('click', handleSubmit);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSubmit();
});

feedbackBtn.addEventListener('click', handleFeedback);
feedbackInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleFeedback();
});

// Example button handlers
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        queryInput.value = btn.dataset.query;
        handleSubmit();
    });
});

// Main submit handler
async function handleSubmit() {
    const query = queryInput.value.trim();
    if (!query) return;

    setLoading(true);
    hideAllSections();
    showStatus('processing', '⏳', 'Processing your query...');

    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, session_id: SESSION_ID })
        });

        const result = await response.json();
        handleResult(result);
    } catch (error) {
        showError('Failed to connect to the server. Make sure the backend is running.');
    } finally {
        setLoading(false);
    }
}

// Handle HITL feedback
async function handleFeedback() {
    const feedback = feedbackInput.value.trim();
    if (!feedback) return;

    setLoading(true);
    showStatus('processing', '⏳', 'Processing with your feedback...');
    hitlSection.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feedback, session_id: SESSION_ID })
        });

        const result = await response.json();
        handleResult(result);
    } catch (error) {
        showError('Failed to submit feedback.');
    } finally {
        setLoading(false);
        feedbackInput.value = '';
    }
}

// Handle API result
function handleResult(result) {
    hideAllSections();

    if (result.status === 'SUCCESS') {
        showStatus('success', '✅', 'Query executed successfully!');
        showResults(result);
    } else if (result.status === 'PAUSED_HITL') {
        showStatus('warning', '⚠️', 'Clarification needed');
        showHitl(result);
    } else if (result.status === 'FAILED') {
        showStatus('error', '❌', `Query failed after ${result.retry_count || 0} retries`);
        showError(result.error || result.message || 'Unknown error');
        // Show debug info if available
        if (result.debug_analysis) {
            showDebugInfo(result);
        }
    } else if (result.error) {
        showError(result.error);
    }
}

// UI Helper Functions
function setLoading(loading) {
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoading = submitBtn.querySelector('.btn-loading');

    if (loading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
        submitBtn.disabled = true;
    } else {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
        submitBtn.disabled = false;
    }
}

function hideAllSections() {
    hitlSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

function showStatus(type, icon, text) {
    statusSection.style.display = 'block';
    statusSection.className = 'status-section ' + type;
    statusIcon.textContent = icon;
    statusText.textContent = text;
}

function showHitl(result) {
    hitlSection.style.display = 'block';
    hitlMessage.textContent = result.message || 'The query seems ambiguous.';
    hitlConfidence.textContent = ((result.confidence || 0) * 100).toFixed(0) + '%';

    // Show ambiguity reasons
    const reasons = result.ambiguity_reasons || [];
    if (reasons.length > 0) {
        hitlReasoning.innerHTML = reasons.map(r => `<li>${r}</li>`).join('');
    } else {
        hitlReasoning.textContent = 'Query requires clarification.';
    }

    feedbackInput.focus();
}

function showResults(result) {
    resultsSection.style.display = 'flex';

    // SQL Output with variations
    let sqlHtml = result.sql || 'No SQL generated';
    if (result.sql_variations && result.sql_variations.length > 1) {
        sqlHtml += '\n\n/* Other variations considered:\n';
        result.sql_variations.slice(1).forEach((v, i) => {
            sqlHtml += `   ${i + 2}. ${v}\n`;
        });
        sqlHtml += '*/';
    }
    sqlOutput.textContent = sqlHtml;

    // Data Output
    const data = result.data || [];
    if (data.length > 0) {
        const columns = result.columns || Object.keys(data[0]);
        let tableHtml = '<table class="data-table"><thead><tr>';

        columns.forEach(col => {
            tableHtml += `<th>${col}</th>`;
        });
        tableHtml += '</tr></thead><tbody>';

        data.forEach(row => {
            tableHtml += '<tr>';
            if (Array.isArray(row)) {
                row.forEach(cell => {
                    tableHtml += `<td>${cell !== null ? cell : '<em>null</em>'}</td>`;
                });
            } else {
                columns.forEach(col => {
                    const cell = row[col];
                    tableHtml += `<td>${cell !== null ? cell : '<em>null</em>'}</td>`;
                });
            }
            tableHtml += '</tr>';
        });

        tableHtml += '</tbody></table>';
        tableHtml += `<p class="row-count">${result.row_count || data.length} row(s) returned</p>`;
        dataOutput.innerHTML = tableHtml;
    } else {
        dataOutput.innerHTML = '<p style="color: var(--text-muted);">No data returned</p>';
    }

    // Plan Output - Enhanced with consistency info
    const confidence = result.confidence || 0;
    const confidenceClass = confidence >= 0.8 ? 'confidence-high' :
        confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';

    let planHtml = `
        <div class="plan-item">
            <div class="plan-item-label">Confidence</div>
            <div class="plan-item-value ${confidenceClass}">${(confidence * 100).toFixed(0)}%</div>
        </div>
        <div class="plan-item">
            <div class="plan-item-label">Intent</div>
            <div class="plan-item-value">${result.detected_intent || 'select'}</div>
        </div>
        <div class="plan-item">
            <div class="plan-item-label">Tables</div>
            <div class="plan-item-value">${(result.detected_tables || []).join(', ') || 'auto-detected'}</div>
        </div>
        <div class="plan-item">
            <div class="plan-item-label">Consistency</div>
            <div class="plan-item-value ${result.consistency_passed ? 'confidence-high' : 'confidence-low'}">
                ${result.consistency_passed ? '✓ Passed' : '✗ Failed'}
            </div>
        </div>
    `;

    // Show few-shot examples if used
    if (result.few_shot_examples && result.few_shot_examples.length > 0) {
        planHtml += `
            <div class="plan-item" style="grid-column: 1 / -1;">
                <div class="plan-item-label">Few-Shot Examples Used</div>
                <div class="plan-item-value" style="font-size: 0.8rem;">
                    ${result.few_shot_examples.map(ex => `"${ex.question}"`).join(', ')}
                </div>
            </div>
        `;
    }

    planOutput.innerHTML = planHtml;
}

function showDebugInfo(result) {
    // Append debug info to error section
    if (result.debug_analysis) {
        errorMessage.innerHTML += `<br><br><strong>Debug Analysis:</strong><br><pre>${result.debug_analysis}</pre>`;
    }
    if (result.sql) {
        errorMessage.innerHTML += `<br><strong>Last SQL Attempt:</strong><br><code>${result.sql}</code>`;
    }
}

function showError(message) {
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
}

// Initial focus
queryInput.focus();

// Add row count styling
const style = document.createElement('style');
style.textContent = `
    .row-count {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 8px;
        text-align: right;
    }
    #hitl-reasoning {
        list-style: disc;
        padding-left: 20px;
        margin: 8px 0;
    }
    #hitl-reasoning li {
        margin: 4px 0;
        color: var(--text-secondary);
    }
`;
document.head.appendChild(style);
