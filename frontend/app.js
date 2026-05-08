/**
 * Valura AI — Frontend Logic
 * Handles chat interactions, SSE streaming, and UI updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const welcomeMsg = document.getElementById('welcome-msg');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const statusBar = document.getElementById('status-bar');
    
    // Status items
    const statusSafety = document.getElementById('status-safety');
    const statusIntent = document.getElementById('status-intent');
    const statusAgent = document.getElementById('status-agent');
    
    let currentAiBubble = null;

    // Right sidebar elements
    const topHoldingPct = document.getElementById('top-holding-pct');
    const concentrationStatus = document.getElementById('concentration-status');
    const concentrationBar = document.getElementById('concentration-bar');
    const perfPortfolio = document.getElementById('perf-portfolio');
    const perfBenchmark = document.getElementById('perf-benchmark');
    const perfAlpha = document.getElementById('perf-alpha');
    const actionItems = document.getElementById('action-items');

    const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    const userId = 'usr_001'; // Default test user

    // Fetch initial summary and start polling
    fetchSummary();
    setInterval(fetchSummary, 15000); // Refresh every 15 seconds

    async function fetchSummary() {
        try {
            const response = await fetch(`http://localhost:8000/portfolio/summary?user_id=${userId}`);
            const data = await response.json();
            
            if (data.error) return;

            // Update Sidebar
            document.getElementById('sidebar-total-value').textContent = `$${data.total_value.toLocaleString()}`;
            document.getElementById('sidebar-total-change').innerHTML = `${data.total_return_pct >= 0 ? '+' : ''}${data.total_return_pct}% <i data-lucide="${data.total_return_pct >= 0 ? 'trending-up' : 'trending-down'}"></i>`;
            document.getElementById('sidebar-total-change').className = `change ${data.total_return_pct >= 0 ? 'positive' : 'negative'}`;
            
            document.getElementById('sidebar-risk-profile').textContent = data.risk_profile.charAt(0).toUpperCase() + data.risk_profile.slice(1);
            document.getElementById('sidebar-diversification-score').textContent = `${data.diversification_score}/100`;
            document.getElementById('sidebar-diversification-bar').style.width = `${data.diversification_score}%`;
            
            const holdingsContainer = document.getElementById('sidebar-top-holdings');
            holdingsContainer.innerHTML = data.top_holdings.map(h => `
                <div class="holding-item">
                    <span class="ticker">${h.ticker}</span>
                    <span class="pct">${h.pct}%</span>
                </div>
            `).join('');
            
            // Update Timestamp from Server
            document.getElementById('last-updated-text').innerHTML = `<span class="live-pulse"></span> Last synced: ${data.last_updated || new Date().toLocaleTimeString()}`;
            
            // Sync any existing chat cards with the new data
            document.querySelectorAll('.live-total-return').forEach(el => {
                el.textContent = `${data.total_return_pct}%`;
                el.className = `value live-total-return ${data.total_return_pct >= 0 ? 'positive' : ''}`;
            });
            
            if (typeof lucide !== 'undefined') lucide.createIcons();
        } catch (err) {
            console.error('Failed to fetch summary', err);
            document.getElementById('last-updated-text').textContent = 'Offline — Sync Error';
        }
    }

    function addMessage(role, content, isHtml = false) {
        if (welcomeMsg) welcomeMsg.style.display = 'none';
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        if (isHtml) {
            bubble.innerHTML = content;
        } else {
            // Use marked for markdown if available, else plain text
            if (typeof marked !== 'undefined') {
                bubble.innerHTML = marked.parse(content);
            } else {
                bubble.textContent = content;
            }
        }
        
        messageDiv.appendChild(bubble);
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return bubble;
    }

    function updateStatus(step, state) {
        // state: 'idle' | 'active' | 'complete'
        const item = {
            'safety': statusSafety,
            'intent': statusIntent,
            'agent': statusAgent
        }[step];

        if (!item) return;

        item.classList.remove('active', 'complete');
        const spinner = item.querySelector('.spinner');

        if (state === 'active') {
            item.classList.add('active');
            if (spinner) spinner.style.display = 'block';
        } else if (state === 'complete') {
            item.classList.add('complete');
            if (spinner) spinner.style.display = 'none';
        } else {
            if (spinner) spinner.style.display = 'none';
        }
    }

    function resetStatus() {
        statusBar.style.display = 'flex';
        updateStatus('safety', 'idle');
        updateStatus('intent', 'idle');
        updateStatus('agent', 'idle');
    }

    async function handleSend(query) {
        if (!query) return;
        
        currentAiBubble = null;
        addMessage('user', query);
        chatInput.value = '';
        resetStatus();

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    user_id: userId,
                    query: query
                })
            });

            if (!response.ok) throw new Error('API unreachable');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const rawData = JSON.parse(line.substring(6));
                            const { event, data } = rawData;

                            if (event === 'status') {
                                handleStatusUpdate(data.message);
                            } else if (event === 'classification') {
                                updateStatus('intent', 'complete');
                            } else if (event === 'chunk') {
                                if (!currentAiBubble) {
                                    currentAiBubble = addMessage('ai', '');
                                }
                                currentAiBubble.textContent += data.text;
                                chatWindow.scrollTop = chatWindow.scrollHeight;
                            } else if (event === 'result') {
                                statusBar.style.display = 'none';
                                renderResult(data);
                            } else if (event === 'blocked') {
                                statusBar.style.display = 'none';
                                addMessage('ai', `⚠️ **Safety Warning:** ${data.message}`);
                            } else if (event === 'error') {
                                statusBar.style.display = 'none';
                                addMessage('ai', `❌ **Error:** ${data.error}`);
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data', e);
                        }
                    }
                }
            }

        } catch (err) {
            statusBar.style.display = 'none';
            addMessage('ai', "I'm having trouble connecting to the Valura services. Please ensure the backend is running at localhost:8000.");
            console.error(err);
        }
    }

    function handleStatusUpdate(message) {
        const msg = message.toLowerCase();
        if (msg.includes('safety')) {
            updateStatus('safety', 'active');
        } else if (msg.includes('classifying') || msg.includes('intent')) {
            updateStatus('safety', 'complete');
            updateStatus('intent', 'active');
        } else if (msg.includes('routing') || msg.includes('analyzing') || msg.includes('calculating') || msg.includes('researching') || msg.includes('fetching')) {
            updateStatus('intent', 'complete');
            updateStatus('agent', 'active');
        }
        
        // Show partial message in status bar if it happens
        if (msg.includes('longer than expected')) {
            const statusText = document.querySelector('#status-agent span');
            if (statusText) statusText.textContent = "Data Fetch Delayed (Partial Mode)";
        }
    }

    function renderResult(data) {
        // If it's a portfolio health result
        if (data.agent === 'portfolio_health') {
            const html = `
                <div class="specialist-card ${data.partial ? 'partial-mode' : ''}">
                    ${data.partial ? '<div class="partial-banner">Showing Partial Insights — Market Data Delayed</div>' : ''}
                    <h4>Portfolio Health Analysis</h4>
                    <div class="card-grid">
                        <div class="grid-item">
                            <span class="label">Total Return</span>
                            <span class="value live-total-return ${(data.performance?.total_return_pct ?? 0) >= 0 ? 'positive' : ''}">${data.performance?.total_return_pct ?? 0}%</span>
                        </div>
                        <div class="grid-item">
                            <span class="label">Annualized</span>
                            <span class="value live-annualized">${data.performance?.annualized_return_pct ?? 0}%</span>
                        </div>
                        <div class="grid-item">
                            <span class="label">Top Position</span>
                            <span class="value live-top-pos">${data.concentration_risk?.top_position_pct ?? 0}%</span>
                        </div>
                        <div class="grid-item">
                            <span class="label">Benchmark Alpha</span>
                            <span class="value live-alpha ${(data.benchmark_comparison?.alpha_pct ?? 0) >= 0 ? 'positive' : ''}">${data.benchmark_comparison?.alpha_pct ?? 0}%</span>
                        </div>
                    </div>
                    <div class="observations">
                        ${data.observations.map(obs => `
                            <div class="observation-item ${obs.severity}">
                                <i data-lucide="${obs.severity === 'warning' ? 'alert-circle' : 'info'}"></i>
                                <span>${obs.text}</span>
                            </div>
                        `).join('')}
                    </div>
                    <p style="font-size: 10px; color: #9CA3AF; margin-top: 10px; line-height: 1.4;">${data.disclaimer || 'Analysis performed using deterministic financial modeling. Past performance does not guarantee future results.'}</p>
                </div>
            `;
            addMessage('ai', html, true);
            if (typeof lucide !== 'undefined') lucide.createIcons();
            
            // Update sidebar
            updateSidebar(data);
        } else if (data.agent === 'market_research' && data.ticker) {
            // Market Research Card
            const html = `
                <div class="specialist-card market-research-card">
                    <div class="research-header">
                        <div class="ticker-badge">${data.ticker}</div>
                        <h4>${data.name}</h4>
                    </div>
                    
                    <div class="research-price-block">
                        <div class="main-price">$${data.price?.toLocaleString()}</div>
                        <div class="price-change ${data.change >= 0 ? 'positive' : 'negative'}">
                            ${data.change >= 0 ? '+' : ''}${data.change} (${data.change_pct}%)
                            <i data-lucide="${data.change >= 0 ? 'trending-up' : 'trending-down'}"></i>
                        </div>
                    </div>

                    <div class="research-metrics">
                        <div class="metric">
                            <span class="label">Market Cap</span>
                            <span class="value">$${(data.metrics.market_cap / 1e12).toFixed(2)}T</span>
                        </div>
                        <div class="metric">
                            <span class="label">P/E Ratio</span>
                            <span class="value">${data.metrics.pe_ratio || 'N/A'}</span>
                        </div>
                    </div>

                    <div class="research-summary">
                        <p>${data.summary.length > 250 ? data.summary.substring(0, 250) + '...' : data.summary}</p>
                    </div>
                    
                    <div class="research-footer">
                        <span>Source: Yahoo Finance</span>
                        <span>Currency: ${data.currency}</span>
                    </div>
                </div>
            `;
            addMessage('ai', html, true);
            if (typeof lucide !== 'undefined') lucide.createIcons();
        } else if (data.message && !currentAiBubble) {
            // Only add message if it wasn't already streamed
            addMessage('ai', data.message);
        }
        
        // Finalize the streamed message with markdown if it exists
        if (currentAiBubble && data.message) {
            if (typeof marked !== 'undefined') {
                currentAiBubble.innerHTML = marked.parse(data.message);
            } else {
                currentAiBubble.textContent = data.message;
            }
        }
    }

    function updateSidebar(data) {
        if (data.concentration_risk) {
            topHoldingPct.textContent = `${data.concentration_risk.top_position_pct ?? 0}%`;
            const flag = data.concentration_risk.flag || 'low';
            concentrationStatus.textContent = flag.charAt(0).toUpperCase() + flag.slice(1) + ' Risk';
            
            concentrationBar.className = `risk-fill risk-${data.concentration_risk.flag}`;
            concentrationBar.style.width = `${Math.min(data.concentration_risk.top_position_pct, 100)}%`;
        }

        if (data.performance && data.benchmark_comparison) {
            const annReturn = data.performance.annualized_return_pct ?? 0;
            const benchReturn = data.benchmark_comparison.benchmark_return_pct ?? 0;
            const alpha = data.benchmark_comparison.alpha_pct ?? 0;
            
            perfPortfolio.textContent = `${annReturn >= 0 ? '+' : ''}${annReturn}%`;
            perfBenchmark.textContent = `${benchReturn >= 0 ? '+' : ''}${benchReturn}%`;
            perfAlpha.textContent = `${alpha >= 0 ? '+' : ''}${alpha}%`;
        }

        if (data.observations) {
            actionItems.innerHTML = '';
            const items = data.observations
                .filter(o => o.severity === 'warning' || o.text.includes('Consider'))
                .slice(0, 3);
            
            if (items.length > 0) {
                items.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = item.text.split('.')[0] + '.'; 
                    actionItems.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = "Maintain current strategy.";
                actionItems.appendChild(li);
            }
        }
    }

    // Event listeners
    sendBtn.addEventListener('click', () => handleSend(chatInput.value));
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend(chatInput.value);
    });

    analyzeBtn.addEventListener('click', () => handleSend("Analyze my portfolio health"));

    document.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => handleSend(chip.textContent));
    });
});
