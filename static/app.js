// Text-to-SQL Research Pipeline - Web UI Application

const API_BASE = '';
const SESSION_ID = 'session_' + Date.now();

// Toggle collapsible card sections
function toggleCard(headerElement) {
    const card = headerElement.closest('.collapsible');
    const content = card.querySelector('.card-content');
    const icon = card.querySelector('.toggle-icon');

    if (card.classList.contains('collapsed')) {
        card.classList.remove('collapsed');
        content.style.display = 'block';
        icon.textContent = '▼';
    } else {
        card.classList.add('collapsed');
        content.style.display = 'none';
        icon.textContent = '▶';
    }
}

// Make toggleCard available globally for onclick handlers
window.toggleCard = toggleCard;

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
const sqlOutputContainer = document.getElementById('sql-output-container');
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
    let reasoningHtml = '';
    
    if (reasons.length > 0) {
        reasoningHtml += '<strong>Semantic Review:</strong><ul>';
        reasoningHtml += reasons.map(r => `<li>${r}</li>`).join('');
        reasoningHtml += '</ul>';
    }
    
    // Add Mixture of Experts Evaluation if present
    const evaluations = result.evaluation_results || [];
    if (evaluations.length > 0) {
        reasoningHtml += '<strong>Expert Evaluations:</strong><ul>';
        evaluations.forEach(expertEval => {
            const statusIcon = expertEval.is_approved ? '✅' : '❌';
            const penaltyText = expertEval.confidence_penalty > 0 ? ` <span style="color: var(--error); font-weight: 500;">(-${(expertEval.confidence_penalty * 100).toFixed(0)}% penalty)</span>` : '';
            reasoningHtml += `<li style="margin-bottom: 6px;">${statusIcon} <strong>${expertEval.expert}</strong>: ${expertEval.reasoning}${penaltyText}</li>`;
        });
        reasoningHtml += '</ul>';
    }

    if (reasoningHtml) {
        hitlReasoning.innerHTML = reasoningHtml;
    } else {
        hitlReasoning.textContent = 'Query requires clarification.';
    }

    feedbackInput.focus();
}

function showResults(result) {
    resultsSection.style.display = 'flex';

    // Reset rating buttons for new results
    const thumbsUpBtn = document.querySelector('.btn-thumbs-up');
    const thumbsDownBtn = document.querySelector('.btn-thumbs-down');
    if (thumbsUpBtn) {
        thumbsUpBtn.classList.remove('selected');
        thumbsUpBtn.disabled = false;
    }
    if (thumbsDownBtn) {
        thumbsDownBtn.classList.remove('selected');
        thumbsDownBtn.disabled = false;
    }

    // SQL Output with variations
    let fallbackSqlHtml = result.sql || 'No SQL generated';
    if (result.sql_variations && result.sql_variations.length > 1) {
        fallbackSqlHtml += '\n\n/* Other variations considered:\n';
        result.sql_variations.slice(1).forEach((v, i) => {
            fallbackSqlHtml += `   ${i + 2}. ${v}\n`;
        });
        fallbackSqlHtml += '*/';
    }
    
    if (result.token_confidence_map && result.token_confidence_map.length > 0) {
        renderSqlHeatmap(result.token_confidence_map, sqlOutputContainer);
    } else {
        sqlOutputContainer.textContent = fallbackSqlHtml;
    }

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
            <div class="plan-item-label">Total Confidence</div>
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

    // Show Mixture of Experts Extracted Evaluations
    const evaluations = result.evaluation_results || [];
    if (evaluations.length > 0) {
        let evalList = evaluations.map(expertEval => {
             const statusIcon = expertEval.is_approved ? '✅' : '❌';
             const penaltyText = expertEval.confidence_penalty > 0 ? ` <span style="color: var(--error)">(-${(expertEval.confidence_penalty * 100).toFixed(0)}% penalty)</span>` : '';
             return `<li style="margin-bottom: 6px;">${statusIcon} <strong>${expertEval.expert}</strong>: ${expertEval.reasoning}${penaltyText}</li>`;
        }).join('');
        
        planHtml += `
            <div class="plan-item" style="grid-column: 1 / -1;">
                <div class="plan-item-label">Confidence Breakdown (Expert Evaluations)</div>
                <div class="plan-item-value" style="font-size: 0.85rem; font-weight: normal; margin-top: 8px;">
                    <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0;">
                        ${evalList}
                    </ul>
                </div>
            </div>
        `;
    }

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

    // Show clarification if one was used
    if (currentClarification || result.user_feedback) {
        const clarificationText = currentClarification || result.user_feedback;
        planHtml += `
            <div class="user-clarification" style="grid-column: 1 / -1;">
                <div class="user-clarification-label">Clarification Used</div>
                <div class="user-clarification-text">"${clarificationText}"</div>
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

function renderSqlHeatmap(tokenData, container) {
    container.innerHTML = '';
    
    tokenData.forEach(item => {
        const span = document.createElement('span');
        span.textContent = item.token;
        
        if (item.confidence < 0.95) {
            const intensity = 1.0 - item.confidence;
            span.style.backgroundColor = `rgba(255, 99, 71, ${intensity})`;
            const percent = (item.confidence * 100).toFixed(1);
            span.title = `Confidence: ${percent}%`;
        }
        
        container.appendChild(span);
    });
}

function showError(message) {
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
}

// Initial focus
queryInput.focus();

// Database Management
const databaseSelect = document.getElementById('database-select');
const dbModeBadge = document.getElementById('db-mode-badge');
const dbNameDisplay = document.getElementById('db-name-display');
const schemaContainer = document.getElementById('schema-container');

// Table icons mapping
const tableIcons = {
    customers: '👤',
    products: '📦',
    orders: '🛒',
    order_items: '📋',
    employees: '👥',
    departments: '🏢',
    projects: '📊',
    project_assignments: '📌',
    default: '📁'
};

// Load available databases on startup
async function loadDatabases() {
    try {
        const response = await fetch(`${API_BASE}/api/databases`);
        const data = await response.json();

        // Clear existing options
        databaseSelect.innerHTML = '<option value="demo">Demo (E-Commerce)</option>';

        // Add BIRD databases
        if (data.databases.bird && data.databases.bird.length > 0) {
            const birdGroup = document.createElement('optgroup');
            birdGroup.label = '🐦 BIRD-bench';
            data.databases.bird.forEach(db => {
                const opt = document.createElement('option');
                opt.value = `bird:${db}`;
                opt.textContent = db;
                birdGroup.appendChild(opt);
            });
            databaseSelect.appendChild(birdGroup);
        }

        // Add Spider databases
        if (data.databases.spider && data.databases.spider.length > 0) {
            const spiderGroup = document.createElement('optgroup');
            spiderGroup.label = '🕷️ Spider';
            data.databases.spider.forEach(db => {
                const opt = document.createElement('option');
                opt.value = `spider:${db}`;
                opt.textContent = db;
                spiderGroup.appendChild(opt);
            });
            databaseSelect.appendChild(spiderGroup);
        }

        // Add Custom databases
        if (data.databases.custom && data.databases.custom.length > 0) {
            const customGroup = document.createElement('optgroup');
            customGroup.label = '📂 Custom';
            data.databases.custom.forEach(db => {
                const opt = document.createElement('option');
                opt.value = `custom:${db}`;
                opt.textContent = db;
                customGroup.appendChild(opt);
            });
            databaseSelect.appendChild(customGroup);
        }

        // Update current selection
        if (data.current_mode === 'benchmark' && data.current_database) {
            // Try to find and select current database
            const options = databaseSelect.querySelectorAll('option');
            options.forEach(opt => {
                if (opt.value.endsWith(`:${data.current_database}`)) {
                    opt.selected = true;
                }
            });
        }

        // Load initial schema
        loadCurrentSchema();

    } catch (error) {
        console.error('Failed to load databases:', error);
        schemaContainer.innerHTML = '<div class="schema-error">Failed to load databases</div>';
    }
}

// Switch database
async function switchDatabase(value) {
    schemaContainer.innerHTML = '<div class="schema-loading">Switching database...</div>';

    let mode, dbId, dataset;

    if (value === 'demo') {
        mode = 'demo';
    } else {
        const parts = value.split(':');
        dataset = parts[0];
        dbId = parts[1];
        mode = 'benchmark';
    }

    try {
        const response = await fetch(`${API_BASE}/api/databases/switch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, db_id: dbId, dataset })
        });

        const data = await response.json();

        if (data.success) {
            updateSchemaDisplay(data.schema, mode);
            showNotification(`Switched to ${dbId || 'demo'} database`, 'success');
        } else {
            schemaContainer.innerHTML = `<div class="schema-error">${data.error}</div>`;
            showNotification(data.error, 'error');
        }
    } catch (error) {
        console.error('Failed to switch database:', error);
        schemaContainer.innerHTML = '<div class="schema-error">Failed to switch database</div>';
    }
}

// Load current schema
async function loadCurrentSchema() {
    try {
        const response = await fetch(`${API_BASE}/api/databases/schema`);
        const data = await response.json();
        updateSchemaDisplay(data.schema, data.mode);
    } catch (error) {
        console.error('Failed to load schema:', error);
        schemaContainer.innerHTML = '<div class="schema-error">Failed to load schema</div>';
    }
}

// Update schema display
function updateSchemaDisplay(schema, mode) {
    if (!schema || !schema.tables) {
        schemaContainer.innerHTML = '<div class="schema-error">No schema available</div>';
        return;
    }

    // Update mode badge
    dbModeBadge.textContent = mode === 'demo' ? 'Demo' : 'Benchmark';
    dbModeBadge.className = `db-badge mode-${mode}`;
    dbNameDisplay.textContent = schema.database_name || 'unknown';

    // Render tables
    let html = '';
    const tables = Object.entries(schema.tables);

    tables.forEach(([tableName, tableInfo]) => {
        const icon = tableIcons[tableName.toLowerCase()] || tableIcons.default;
        const rowCount = tableInfo.row_count !== undefined ? ` (${tableInfo.row_count} rows)` : '';

        html += `<div class="schema-table">`;
        html += `<div class="table-name">${icon} ${tableName}${rowCount}</div>`;
        html += `<ul class="columns">`;

        if (tableInfo.columns) {
            Object.entries(tableInfo.columns).forEach(([colName, colInfo]) => {
                let typeLabel = colInfo.type || 'TEXT';
                if (colInfo.is_primary_key) typeLabel += ' PK';
                if (colInfo.foreign_key) typeLabel = `FK → ${colInfo.foreign_key.split('.')[0]}`;

                html += `<li>`;
                html += `<span class="col-name">${colName}</span> `;
                html += `<span class="col-type">${typeLabel}</span>`;
                html += `</li>`;
            });
        }

        html += `</ul></div>`;
    });

    schemaContainer.innerHTML = html;
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => notification.classList.add('show'), 10);
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Database select change handler
databaseSelect.addEventListener('change', (e) => {
    switchDatabase(e.target.value);
});

// Initialize on page load
loadDatabases();

// Add dynamic styles
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
    .schema-header {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 12px;
    }
    .schema-header h3 {
        margin: 0;
    }
    .database-selector select {
        width: 100%;
        padding: 8px 12px;
        border: 1px solid var(--border-color, rgba(255,255,255,0.1));
        border-radius: 8px;
        background: rgba(0,0,0,0.2);
        color: var(--text-primary, #fff);
        font-size: 0.85rem;
        cursor: pointer;
    }
    .database-selector select:focus {
        outline: none;
        border-color: var(--primary-color, #6366f1);
    }
    .database-selector optgroup {
        font-weight: 600;
        color: var(--text-secondary, #aaa);
    }
    .database-info {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-color, rgba(255,255,255,0.1));
    }
    .db-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .mode-demo {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
    }
    .mode-benchmark {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
    }
    #db-name-display {
        font-size: 0.85rem;
        color: var(--text-secondary, #aaa);
        font-family: 'JetBrains Mono', monospace;
    }
    .schema-tables {
        max-height: calc(100vh - 280px);
        overflow-y: auto;
    }
    .schema-loading, .schema-error {
        padding: 16px;
        text-align: center;
        color: var(--text-muted, #666);
        font-size: 0.9rem;
    }
    .schema-error {
        color: #f87171;
    }
    .notification {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 0.9rem;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.3s ease;
        z-index: 1000;
    }
    .notification.show {
        opacity: 1;
        transform: translateY(0);
    }
    .notification-success {
        background: rgba(34, 197, 94, 0.9);
        color: white;
    }
    .notification-error {
        background: rgba(239, 68, 68, 0.9);
        color: white;
    }
    .notification-info {
        background: rgba(99, 102, 241, 0.9);
        color: white;
    }
    
    /* Benchmark Panel Styles */
    .header-actions {
        display: flex;
        gap: 10px;
        margin-top: 12px;
    }
    .btn-secondary {
        padding: 8px 16px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        color: #a5b4fc;
        font-size: 0.85rem;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.2s;
    }
    .btn-secondary:hover {
        background: rgba(99, 102, 241, 0.3);
    }
    .benchmark-panel {
        margin: 20px 0;
        padding: 20px;
    }
    .benchmark-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .benchmark-header h3 {
        margin: 0;
    }
    .close-btn {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: var(--text-muted);
    }
    .benchmark-status {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        margin-bottom: 16px;
    }
    .benchmark-progress {
        flex: 1;
    }
    .benchmark-stats {
        display: flex;
        gap: 15px;
    }
    .stat-success { color: #4ade80; }
    .stat-failed { color: #f87171; }
    .stat-hitl { color: #fbbf24; }
    .progress-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin-top: 8px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .benchmark-question {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .question-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .current-question {
        font-size: 1.1rem;
        font-weight: 500;
    }
    .difficulty-badge {
        display: inline-block;
        margin-top: 8px;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
    }
    .benchmark-actions {
        display: flex;
        gap: 10px;
    }
    
    /* Save Example Button Styles */
    .card-actions {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .btn-small {
        padding: 4px 10px;
        font-size: 0.75rem;
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 6px;
        color: #4ade80;
        cursor: pointer;
        transition: all 0.2s;
    }
    .btn-small:hover {
        background: rgba(34, 197, 94, 0.3);
    }
    .btn-small:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
`;
document.head.appendChild(style);


// ============== SAVE AS EXAMPLE ==============

// Store the current result for saving
let currentResult = null;
let currentClarification = null;  // Track HITL clarification

// Override handleResult to store the result
const originalHandleResult = handleResult;
handleResult = function (result) {
    currentResult = result;
    originalHandleResult(result);
};

// Track when feedback is submitted (capture before it's cleared)
feedbackBtn.addEventListener('click', () => {
    currentClarification = feedbackInput.value.trim();
}, true);

// Save current query as example
async function saveAsExample() {
    const query = queryInput.value.trim();
    const sql = currentResult?.sql;

    if (!query || !sql) {
        showNotification('No query to save', 'error');
        return;
    }

    const btn = document.getElementById('save-example-btn');
    btn.disabled = true;
    btn.textContent = '💾 Saving...';

    try {
        const payload = {
            question: query,
            sql: sql,
            intent: currentResult?.detected_intent || 'user_saved',
            tables: currentResult?.detected_tables || [],
            difficulty: 'medium'
        };

        // Include clarification if one was provided
        if (currentClarification) {
            payload.clarification = currentClarification;
        }

        const response = await fetch(`${API_BASE}/api/examples/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.success) {
            const msg = currentClarification
                ? `Example saved with clarification! (${result.total_examples} total)`
                : `Example saved! (${result.total_examples} total)`;
            showNotification(msg, 'success');
            btn.textContent = '✓ Saved!';
            currentClarification = null;  // Reset after saving
            setTimeout(() => {
                btn.textContent = '💾 Save as Example';
                btn.disabled = false;
            }, 2000);
        } else {
            showNotification(result.error || 'Failed to save', 'error');
            btn.textContent = '💾 Save as Example';
            btn.disabled = false;
        }
    } catch (error) {
        console.error('Failed to save example:', error);
        showNotification('Failed to save example', 'error');
        btn.textContent = '💾 Save as Example';
        btn.disabled = false;
    }
}

// Make saveAsExample available globally
window.saveAsExample = saveAsExample;

// Submit rating feedback (thumbs up/down)
async function submitRating(rating) {
    const query = queryInput.value.trim();
    const sql = currentResult?.sql;

    if (!query || !sql) {
        showNotification('No query to rate', 'error');
        return;
    }

    // Get all rating buttons
    const thumbsUpBtn = document.querySelector('.btn-thumbs-up');
    const thumbsDownBtn = document.querySelector('.btn-thumbs-down');

    // Prompt for comment, especially if negative feedback
    let comment = '';
    if (rating === 0) {
        comment = prompt('What was wrong with this result? (optional)');
        if (comment === null) return; // User cancelled
    } else {
        // Optional comment for positive feedback
        comment = prompt('Any additional comments? (optional)') || '';
        if (comment === null) comment = '';
    }

    // Disable buttons while submitting
    if (thumbsUpBtn) thumbsUpBtn.disabled = true;
    if (thumbsDownBtn) thumbsDownBtn.disabled = true;

    try {
        const payload = {
            session_id: SESSION_ID,
            rating: rating,
            question: query,
            sql: sql,
            comment: comment
        };

        // Include clarification if one was provided during HITL
        if (currentClarification) {
            payload.clarification = currentClarification;
        }

        const response = await fetch(`${API_BASE}/api/feedback/rate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.success) {
            // Mark the selected button as selected
            if (rating === 1 && thumbsUpBtn) {
                thumbsUpBtn.classList.add('selected');
            } else if (rating === 0 && thumbsDownBtn) {
                thumbsDownBtn.classList.add('selected');
            }

            const ratingText = rating === 1 ? 'Positive' : 'Negative';
            showNotification(`${ratingText} feedback recorded! Thank you.`, 'success');
        } else {
            showNotification(result.error || 'Failed to submit feedback', 'error');
            // Re-enable buttons on error
            if (thumbsUpBtn) thumbsUpBtn.disabled = false;
            if (thumbsDownBtn) thumbsDownBtn.disabled = false;
        }
    } catch (error) {
        console.error('Failed to submit rating:', error);
        showNotification('Failed to submit feedback', 'error');
        // Re-enable buttons on error
        if (thumbsUpBtn) thumbsUpBtn.disabled = false;
        if (thumbsDownBtn) thumbsDownBtn.disabled = false;
    }
}

// Make submitRating available globally
window.submitRating = submitRating;


// ============== BENCHMARK MODE ==============

const benchmarkBtn = document.getElementById('benchmark-btn');
const benchmarkPanel = document.getElementById('benchmark-panel');
const closeBenchmarkBtn = document.getElementById('close-benchmark');
const startBenchmarkBtn = document.getElementById('start-benchmark-btn');
const nextQuestionBtn = document.getElementById('next-question-btn');
const benchProgressText = document.getElementById('bench-progress-text');
const benchProgressFill = document.getElementById('bench-progress-fill');
const benchSuccess = document.getElementById('bench-success');
const benchFailed = document.getElementById('bench-failed');
const benchHitl = document.getElementById('bench-hitl');
const benchQuestionContainer = document.getElementById('benchmark-question-container');
const benchCurrentQuestion = document.getElementById('bench-current-question');
const benchDifficulty = document.getElementById('bench-difficulty');

let benchmarkState = {
    runId: null,
    totalQuestions: 10,
    currentIndex: 0,
    successCount: 0,
    failedCount: 0,
    hitlCount: 0,
    waitingFeedback: false
};

// Toggle benchmark panel
benchmarkBtn.addEventListener('click', () => {
    benchmarkPanel.style.display = benchmarkPanel.style.display === 'none' ? 'block' : 'none';
});

closeBenchmarkBtn.addEventListener('click', () => {
    benchmarkPanel.style.display = 'none';
});

// Start benchmark
startBenchmarkBtn.addEventListener('click', async () => {
    startBenchmarkBtn.disabled = true;
    startBenchmarkBtn.textContent = 'Starting...';

    try {
        const response = await fetch(`${API_BASE}/api/benchmark/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ count: 10 })
        });

        const data = await response.json();

        benchmarkState = {
            runId: data.run_id,
            totalQuestions: data.total_questions,
            currentIndex: 0,
            successCount: 0,
            failedCount: 0,
            hitlCount: 0,
            waitingFeedback: false
        };

        // Update UI
        startBenchmarkBtn.style.display = 'none';
        nextQuestionBtn.style.display = 'inline-block';
        benchProgressText.textContent = `Question 0/${data.total_questions}`;

        showNotification('Benchmark started!', 'success');

        // Auto-run first question
        runNextQuestion();

    } catch (error) {
        console.error('Failed to start benchmark:', error);
        showNotification('Failed to start benchmark', 'error');
    } finally {
        startBenchmarkBtn.disabled = false;
        startBenchmarkBtn.textContent = 'Start Benchmark';
    }
});

// Next question button
nextQuestionBtn.addEventListener('click', runNextQuestion);

async function runNextQuestion(feedback = null) {
    nextQuestionBtn.disabled = true;
    nextQuestionBtn.textContent = 'Processing...';

    try {
        const body = { run_id: benchmarkState.runId };
        if (feedback) {
            body.feedback = feedback;
        }

        const response = await fetch(`${API_BASE}/api/benchmark/next`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const result = await response.json();

        if (result.benchmark_complete) {
            // Benchmark finished
            benchProgressText.textContent = 'Benchmark Complete!';
            benchProgressFill.style.width = '100%';
            benchQuestionContainer.style.display = 'none';
            nextQuestionBtn.style.display = 'none';
            startBenchmarkBtn.style.display = 'inline-block';
            startBenchmarkBtn.textContent = 'Run Another Benchmark';

            showNotification('Benchmark complete! View results in Analytics.', 'success');
            return;
        }

        // Update progress
        const idx = (result.question_index || 0) + 1;
        benchmarkState.currentIndex = idx;

        benchProgressText.textContent = `Question ${idx}/${benchmarkState.totalQuestions}`;
        benchProgressFill.style.width = `${(idx / benchmarkState.totalQuestions) * 100}%`;

        // Show current question
        benchQuestionContainer.style.display = 'block';
        benchCurrentQuestion.textContent = result.question || 'Processing...';
        benchDifficulty.textContent = result.difficulty || '';

        if (result.needs_feedback) {
            // HITL needed - show in normal UI
            benchmarkState.waitingFeedback = true;
            benchmarkState.hitlCount++;
            benchHitl.textContent = `${benchmarkState.hitlCount} 🤔`;

            // Show HITL section
            showHitl(result);
            hitlSection.style.display = 'block';

            nextQuestionBtn.textContent = 'Waiting for feedback...';
            nextQuestionBtn.disabled = true;

            // Add evidence to HITL section
            if (result.evidence) {
                hitlMessage.textContent = `${result.message}\n\nHint: ${result.evidence}`;
            }
        } else {
            // Normal result
            if (result.status === 'SUCCESS') {
                benchmarkState.successCount++;
                benchSuccess.textContent = `${benchmarkState.successCount} ✓`;
            } else {
                benchmarkState.failedCount++;
                benchFailed.textContent = `${benchmarkState.failedCount} ✗`;
            }

            // Show results
            handleResult(result);

            nextQuestionBtn.textContent = 'Next Question →';
            nextQuestionBtn.disabled = false;
        }

    } catch (error) {
        console.error('Failed to get next question:', error);
        showNotification('Error processing question', 'error');
        nextQuestionBtn.disabled = false;
        nextQuestionBtn.textContent = 'Next Question →';
    }
}

// Override feedback handler for benchmark mode
const originalHandleFeedback = handleFeedback;
handleFeedback = async function () {
    const feedback = feedbackInput.value.trim();
    if (!feedback) return;

    if (benchmarkState.waitingFeedback && benchmarkState.runId) {
        // Benchmark mode feedback
        setLoading(true);
        hitlSection.style.display = 'none';

        await runNextQuestion(feedback);

        feedbackInput.value = '';
        benchmarkState.waitingFeedback = false;
        setLoading(false);

        nextQuestionBtn.textContent = 'Next Question →';
        nextQuestionBtn.disabled = false;
    } else {
        // Normal feedback
        return originalHandleFeedback.call(this);
    }
};
