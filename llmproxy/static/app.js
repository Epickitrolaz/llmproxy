/**
 * LLMProxy Management Dashboard Application
 * Modern Vanilla JavaScript (ES6+) • Zero External Dependencies • Offline-First
 */

(function () {
  "use strict";

  // Application State
  const state = {
    currentTab: "overview",
    plugins: [],
    templates: [],
    accounts: [],
    accountProviderFilter: "",
    selectedAccountId: null,
    stats: {},
    logs: [],
    activeSnippetLang: "python",
    selectedSnippetModel: "gpt-4o",
    selectedEditorPluginId: null,
    selectedPlaygroundModel: null,
    chatMessages: [],
    autoRefreshLogs: false,
    autoRefreshInterval: null,
  };

  // DOM Elements Cache
  const el = {
    navTabs: document.querySelectorAll(".nav-tab"),
    tabPanels: document.querySelectorAll(".tab-panel"),
    // Stats
    statRequests: document.getElementById("stat-total-requests"),
    statSuccessRate: document.getElementById("stat-success-rate"),
    statActiveModels: document.getElementById("stat-active-models"),
    statActivePlugins: document.getElementById("stat-active-plugins"),
    statAvgLatency: document.getElementById("stat-avg-latency"),
    // Overview Snippet
    snippetCode: document.getElementById("snippet-code"),
    snippetModelSelect: document.getElementById("snippet-model-select"),
    copySnippetBtn: document.getElementById("copy-snippet-btn"),
    copyEndpointBtn: document.getElementById("copy-endpoint-btn"),
    // Providers
    providersGrid: document.getElementById("providers-grid"),
    btnNewProvider: document.getElementById("btn-new-provider"),
    // Accounts
    statTotalAccounts: document.getElementById("stat-total-accounts"),
    statActiveAccounts: document.getElementById("stat-active-accounts"),
    statCooldownAccounts: document.getElementById("stat-cooldown-accounts"),
    statRoutingStrategy: document.getElementById("stat-routing-strategy"),
    accountsTableBody: document.getElementById("accounts-table-body"),
    accountsProviderFilter: document.getElementById("accounts-provider-filter"),
    btnConnectAccount: document.getElementById("btn-connect-account"),
    modalAccount: document.getElementById("modal-account"),
    modalAccountTitle: document.getElementById("modal-account-title"),
    accountFormId: document.getElementById("account-form-id"),
    accountProviderSelect: document.getElementById("account-provider-select"),
    accountName: document.getElementById("account-name"),
    accountApiKey: document.getElementById("account-api-key"),
    btnToggleAccountKey: document.getElementById("btn-toggle-account-key"),
    accountBaseUrl: document.getElementById("account-base-url"),
    accountPriority: document.getElementById("account-priority"),
    accountWeight: document.getElementById("account-weight"),
    accountEnabledToggle: document.getElementById("account-enabled-toggle"),
    btnSaveAccount: document.getElementById("btn-save-account"),
    // Editor
    editorPluginSelect: document.getElementById("editor-plugin-select"),
    editorTemplateSelect: document.getElementById("editor-template-select"),
    editorTextarea: document.getElementById("editor-code"),
    editorLineNumbers: document.getElementById("editor-line-numbers"),
    editorSaveBtn: document.getElementById("editor-save-btn"),
    editorSwapBtn: document.getElementById("editor-swap-btn"),
    editorFileInput: document.getElementById("editor-file-input"),
    editorFeedback: document.getElementById("editor-feedback"),
    editorFilenameBadge: document.getElementById("editor-filename-badge"),
    // Models
    modelsTableBody: document.getElementById("models-table-body"),
    modelsSearchInput: document.getElementById("models-search-input"),
    btnAddModel: document.getElementById("btn-add-model"),
    // Playground
    playgroundModelSelect: document.getElementById("playground-model-select"),
    playgroundTempInput: document.getElementById("playground-temp-input"),
    playgroundTempDisplay: document.getElementById("playground-temp-display"),
    playgroundMaxTokensInput: document.getElementById("playground-tokens-input"),
    playgroundSystemInput: document.getElementById("playground-system-input"),
    playgroundHistory: document.getElementById("playground-history"),
    playgroundPrompt: document.getElementById("playground-prompt"),
    playgroundSendBtn: document.getElementById("playground-send-btn"),
    playgroundClearBtn: document.getElementById("playground-clear-btn"),
    // Logs
    logsTableBody: document.getElementById("logs-table-body"),
    logsRefreshBtn: document.getElementById("logs-refresh-btn"),
    logsAutoToggle: document.getElementById("logs-auto-toggle"),
    // Modals
    modalNewProvider: document.getElementById("modal-new-provider"),
    modalVariables: document.getElementById("modal-variables"),
    modalModel: document.getElementById("modal-model"),
    toastContainer: document.getElementById("toast-container"),
  };

  // =========================================================================
  // Toast Notifications
  // =========================================================================

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = "toast";
    let icon = "✓";
    if (type === "error") icon = "✕";
    if (type === "warning") icon = "⚠";

    toast.innerHTML = `<span style="color: ${type === 'error' ? '#f43f5e' : '#10b981'}; font-weight: bold;">${icon}</span> <span>${message}</span>`;
    el.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      toast.style.transition = "all 0.2s ease";
      setTimeout(() => toast.remove(), 200);
    }, 3200);
  }

  // =========================================================================
  // API Calls
  // =========================================================================

  async function api(path, options = {}) {
    try {
      const res = await fetch(path, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        ...options,
      });

      if (!res.ok) {
        let errorDetail = `HTTP ${res.status}`;
        try {
          const errData = await res.json();
          errorDetail = errData.detail || errorDetail;
        } catch (_) {}
        throw new Error(errorDetail);
      }

      const contentType = res.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        return await res.json();
      }
      return await res.text();
    } catch (err) {
      console.error(`API Error on ${path}:`, err);
      throw err;
    }
  }

  // =========================================================================
  // Navigation & Tab Management
  // =========================================================================

  function switchTab(tabId) {
    state.currentTab = tabId;

    el.navTabs.forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === tabId);
    });

    el.tabPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === `tab-${tabId}`);
    });

    // Tab specific refresh actions
    if (tabId === "overview") {
      fetchStats();
      fetchPlugins();
      fetchAccounts();
    } else if (tabId === "providers") {
      fetchPlugins();
      fetchAccounts();
    } else if (tabId === "accounts") {
      fetchPlugins();
      fetchAccounts();
    } else if (tabId === "editor") {
      if (state.plugins.length > 0 && !state.selectedEditorPluginId) {
        selectEditorPlugin(state.plugins[0].id);
      }
    } else if (tabId === "models") {
      renderModelsTable();
    } else if (tabId === "playground") {
      syncPlaygroundModels();
    } else if (tabId === "logs") {
      fetchLogs();
    }
  }

  el.navTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  // =========================================================================
  // Data Fetching & Sync
  // =========================================================================

  async function fetchStats() {
    try {
      const data = await api("/api/stats");
      state.stats = data;
      renderStats();
    } catch (err) {
      console.error("Failed to fetch stats", err);
    }
  }

  async function fetchPlugins() {
    try {
      const data = await api("/api/plugins");
      state.plugins = Array.isArray(data) ? data : data.plugins || [];
      syncAccountProviderSelects();
      renderProvidersGrid();
      renderModelsTable();
      syncEditorPluginSelect();
      syncSnippetModels();
      syncPlaygroundModels();
      renderStats();
    } catch (err) {
      console.error("Failed to fetch plugins", err);
    }
  }

  async function fetchAccounts() {
    try {
      const data = await api("/api/accounts");
      state.accounts = Array.isArray(data) ? data : [];
      renderAccountsTable();
      renderAccountsStats();
      renderProvidersGrid();
    } catch (err) {
      console.error("Failed to fetch accounts", err);
    }
  }

  function syncAccountProviderSelects() {
    if (el.accountProviderSelect) {
      const current = el.accountProviderSelect.value;
      el.accountProviderSelect.innerHTML = "";
      state.plugins.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = `${p.name} (${p.id})`;
        el.accountProviderSelect.appendChild(opt);
      });
      if (current) el.accountProviderSelect.value = current;
    }

    if (el.accountsProviderFilter) {
      const current = el.accountsProviderFilter.value;
      el.accountsProviderFilter.innerHTML = '<option value="">All Providers</option>';
      state.plugins.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.name;
        el.accountsProviderFilter.appendChild(opt);
      });
      if (current) el.accountsProviderFilter.value = current;
    }
  }

  async function fetchTemplates() {
    try {
      const data = await api("/api/templates");
      state.templates = Array.isArray(data) ? data : [];
      syncTemplateSelects();
    } catch (err) {
      console.error("Failed to fetch templates", err);
    }
  }

  async function fetchLogs() {
    try {
      const data = await api("/api/logs?limit=100");
      state.logs = Array.isArray(data) ? data : data.logs || [];
      renderLogsTable();
    } catch (err) {
      console.error("Failed to fetch logs", err);
    }
  }

  // =========================================================================
  // Overview Tab & Snippets
  // =========================================================================

  function renderStats() {
    const s = state.stats;
    if (el.statRequests) el.statRequests.textContent = s.total_requests ?? 0;
    if (el.statSuccessRate) el.statSuccessRate.textContent = `${s.success_rate ?? 100}%`;
    if (el.statActiveModels) el.statActiveModels.textContent = s.active_models_count ?? 0;
    if (el.statActivePlugins) el.statActivePlugins.textContent = s.active_plugins_count ?? 0;
    if (el.statAvgLatency) el.statAvgLatency.textContent = `${Math.round(s.avg_latency_ms ?? 0)} ms`;
  }

  function syncSnippetModels() {
    if (!el.snippetModelSelect) return;
    const currentVal = el.snippetModelSelect.value;
    el.snippetModelSelect.innerHTML = "";

    const allModels = [];
    state.plugins.forEach((p) => {
      if (p.enabled) {
        p.models.forEach((m) => {
          if (m.enabled) allModels.push(m.id);
        });
      }
    });

    if (allModels.length === 0) {
      allModels.push("gpt-4o");
    }

    allModels.forEach((modelId) => {
      const opt = document.createElement("option");
      opt.value = modelId;
      opt.textContent = modelId;
      el.snippetModelSelect.appendChild(opt);
    });

    if (allModels.includes(currentVal)) {
      el.snippetModelSelect.value = currentVal;
    } else {
      el.snippetModelSelect.value = allModels[0];
    }
    state.selectedSnippetModel = el.snippetModelSelect.value;
    updateSnippetCode();
  }

  function updateSnippetCode() {
    const model = state.selectedSnippetModel || "gpt-4o";
    const lang = state.activeSnippetLang;
    const baseUrl = `${window.location.origin}/v1`;

    let code = "";
    if (lang === "python") {
      code = `from openai import OpenAI

# Connect to local LLMProxy router
client = OpenAI(
    base_url="${baseUrl}",
    api_key="llmproxy",  # Router routes credentials automatically
)

response = client.chat.completions.create(
    model="${model}",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)`;
    } else if (lang === "curl") {
      code = `curl -X POST "${baseUrl}/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer llmproxy" \\
  -d '{
    "model": "${model}",
    "messages": [
      {"role": "user", "content": "Hello from LLMProxy!"}
    ],
    "temperature": 0.7
  }'`;
    } else if (lang === "cursor") {
      code = `// In Cursor or Cline AI Settings -> OpenAI-Compatible:
{
  "baseURL": "${baseUrl}",
  "apiKey": "llmproxy",
  "model": "${model}"
}`;
    }

    if (el.snippetCode) el.snippetCode.textContent = code;
  }

  document.querySelectorAll(".snippet-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".snippet-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.activeSnippetLang = tab.dataset.lang;
      updateSnippetCode();
    });
  });

  if (el.snippetModelSelect) {
    el.snippetModelSelect.addEventListener("change", (e) => {
      state.selectedSnippetModel = e.target.value;
      updateSnippetCode();
    });
  }

  if (el.copySnippetBtn) {
    el.copySnippetBtn.addEventListener("click", () => {
      if (el.snippetCode) {
        navigator.clipboard.writeText(el.snippetCode.textContent);
        showToast("Code snippet copied to clipboard");
      }
    });
  }

  if (el.copyEndpointBtn) {
    el.copyEndpointBtn.addEventListener("click", () => {
      const url = `${window.location.origin}/v1`;
      navigator.clipboard.writeText(url);
      showToast(`Copied endpoint: ${url}`);
    });
  }

  // =========================================================================
  // Providers & Plugins Tab
  // =========================================================================

  function renderProvidersGrid() {
    if (!el.providersGrid) return;
    el.providersGrid.innerHTML = "";

    if (state.plugins.length === 0) {
      el.providersGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 48px 24px; text-align: center; background: var(--bg-card); border: 1px dashed var(--border-solid); border-radius: var(--radius-card);">
          <h3 style="font-size: 16px; margin-bottom: 8px;">No Provider Plugins Loaded</h3>
          <p style="color: var(--ink-secondary); margin-bottom: 20px;">Add your first Python plugin or load an offline mock provider.</p>
          <button class="btn btn-primary" id="btn-quick-starter">+ Load Starter Plugin</button>
        </div>
      `;
      const quickBtn = document.getElementById("btn-quick-starter");
      if (quickBtn) {
        quickBtn.addEventListener("click", () => openNewProviderModal());
      }
      return;
    }

    state.plugins.forEach((p) => {
      const card = document.createElement("div");
      card.className = "provider-card";

      const modelsCount = p.models ? p.models.length : 0;
      const modelBadges = (p.models || [])
        .slice(0, 3)
        .map((m) => `<span class="badge">${m.id}</span>`)
        .join(" ");

      const connectedAccounts = (state.accounts || []).filter((a) => a.provider_id === p.id);
      const accountsBadge = `<span class="badge ${connectedAccounts.length > 0 ? 'badge-success' : 'badge-muted'} btn-jump-accounts" data-id="${p.id}" style="cursor: pointer;" title="View in Accounts tab">${connectedAccounts.length} accounts connected</span>`;

      card.innerHTML = `
        <div>
          <div class="provider-header">
            <div class="provider-title-group">
              <h3>
                ${escapeHtml(p.name)}
                <span class="badge" style="font-size: 10px;">v${escapeHtml(p.version || "1.0.0")}</span>
              </h3>
              <div class="provider-id">${escapeHtml(p.id)} • ${escapeHtml(p.filename || p.id + ".py")}</div>
            </div>
            <label class="switch" title="Toggle plugin enabled status">
              <input type="checkbox" class="provider-enable-toggle" data-id="${p.id}" ${p.enabled ? "checked" : ""}>
              <span class="slider"></span>
            </label>
          </div>
          <p class="provider-desc">${escapeHtml(p.description || "Provider plugin for upstream model routing.")}</p>
          <div class="provider-models-badge-group">
            ${accountsBadge}
            ${modelBadges}
            ${modelsCount > 3 ? `<span class="badge">+${modelsCount - 3} more</span>` : ""}
          </div>
        </div>
        <div class="provider-footer">
          <span style="font-size: 12px; color: var(--ink-muted); font-family: var(--font-mono);">${modelsCount} models</span>
          <div class="provider-actions">
            <button class="btn btn-secondary btn-sm btn-provider-account" data-id="${p.id}">+ Account</button>
            <button class="btn btn-secondary btn-sm btn-vars" data-id="${p.id}">Variables</button>
            <button class="btn btn-secondary btn-sm btn-edit-code" data-id="${p.id}">Edit Code</button>
            <button class="btn btn-secondary btn-sm btn-reload" data-id="${p.id}" title="Hot reload plugin">⟳</button>
            <button class="btn btn-danger btn-sm btn-delete-plugin" data-id="${p.id}" title="Delete plugin">✕</button>
          </div>
        </div>
      `;

      // Event Listeners for Card
      const toggle = card.querySelector(".provider-enable-toggle");
      toggle.addEventListener("change", async (e) => {
        try {
          await api(`/api/plugins/${p.id}/config`, {
            method: "PUT",
            body: JSON.stringify({ enabled: e.target.checked }),
          });
          p.enabled = e.target.checked;
          showToast(`Plugin '${p.name}' ${p.enabled ? "enabled" : "disabled"}`);
          fetchStats();
        } catch (err) {
          e.target.checked = !e.target.checked;
          showToast(`Failed to update status: ${err.message}`, "error");
        }
      });

      card.querySelector(".btn-jump-accounts").addEventListener("click", () => {
        state.accountProviderFilter = p.id;
        if (el.accountsProviderFilter) el.accountsProviderFilter.value = p.id;
        switchTab("accounts");
      });
      card.querySelector(".btn-provider-account").addEventListener("click", () => {
        openAccountModal(null, p.id);
      });
      card.querySelector(".btn-vars").addEventListener("click", () => openVariablesModal(p.id));
      card.querySelector(".btn-edit-code").addEventListener("click", () => {
        switchTab("editor");
        selectEditorPlugin(p.id);
      });
      card.querySelector(".btn-reload").addEventListener("click", async () => {
        try {
          await api(`/api/plugins/${p.id}/reload`, { method: "POST" });
          showToast(`Plugin '${p.name}' hot-reloaded successfully`);
          fetchPlugins();
        } catch (err) {
          showToast(`Reload failed: ${err.message}`, "error");
        }
      });
      card.querySelector(".btn-delete-plugin").addEventListener("click", async () => {
        if (confirm(`Are you sure you want to delete plugin '${p.name}' and remove its Python file?`)) {
          try {
            await api(`/api/plugins/${p.id}`, { method: "DELETE" });
            showToast(`Plugin '${p.id}' deleted`);
            fetchPlugins();
          } catch (err) {
            showToast(`Delete failed: ${err.message}`, "error");
          }
        }
      });

      el.providersGrid.appendChild(card);
    });
  }

  // =========================================================================
  // Accounts & Multi-Account Load Balancing Tab
  // =========================================================================

  function renderAccountsStats() {
    const total = state.accounts.length;
    const now = Date.now() / 1000;
    const cooldownCount = state.accounts.filter(
      (a) => a.rate_limited_until && a.rate_limited_until > now
    ).length;
    const activeCount = state.accounts.filter(
      (a) => a.enabled && (!a.rate_limited_until || a.rate_limited_until <= now)
    ).length;

    if (el.statTotalAccounts) el.statTotalAccounts.textContent = total;
    if (el.statActiveAccounts) el.statActiveAccounts.textContent = activeCount;
    if (el.statCooldownAccounts) el.statCooldownAccounts.textContent = cooldownCount;

    if (el.statRoutingStrategy) {
      const agPlugin = state.plugins.find((p) => p.id === "antigravity");
      const stratVar = agPlugin?.variables?.find((v) => v.name === "load_balancing_strategy");
      const stratVal = stratVar?.value || stratVar?.default || "round_robin";
      el.statRoutingStrategy.textContent = String(stratVal).replace("_", " ");
    }
  }

  function renderAccountsTable() {
    if (!el.accountsTableBody) return;
    el.accountsTableBody.innerHTML = "";

    const filterProvider = state.accountProviderFilter;
    const filtered = filterProvider
      ? state.accounts.filter((a) => a.provider_id === filterProvider)
      : state.accounts;

    if (filtered.length === 0) {
      el.accountsTableBody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align: center; padding: 48px 16px; color: var(--ink-secondary);">
            <div style="font-size: 14px; margin-bottom: 8px;">No Connected Accounts ${filterProvider ? `for ${filterProvider}` : ''}</div>
            <p style="font-size: 12px; color: var(--ink-muted); margin-bottom: 16px;">Connect accounts to enable automated load balancing and priority failovers.</p>
            <button class="btn btn-primary btn-sm" id="btn-empty-connect">+ Connect Account</button>
          </td>
        </tr>
      `;
      const emptyBtn = document.getElementById("btn-empty-connect");
      if (emptyBtn) emptyBtn.addEventListener("click", () => openAccountModal(null, filterProvider));
      return;
    }

    const now = Date.now() / 1000;

    filtered.forEach((acc) => {
      const plugin = state.plugins.find((p) => p.id === acc.provider_id);
      const providerLabel = plugin ? plugin.name : acc.provider_id;
      const isCooldown = Boolean(acc.rate_limited_until && acc.rate_limited_until > now);
      const remSeconds = isCooldown ? Math.ceil(acc.rate_limited_until - now) : 0;

      let statusBadge = "";
      if (!acc.enabled) {
        statusBadge = `<span class="badge badge-muted">Disabled</span>`;
      } else if (isCooldown) {
        statusBadge = `<span class="badge badge-warning" data-cooldown-until="${acc.rate_limited_until}">⏳ Cooldown (${remSeconds}s)</span>`;
      } else {
        statusBadge = `<span class="badge badge-success"><span class="status-dot"></span> Active</span>`;
      }

      const maskedKey = acc.api_key
        ? `${acc.api_key.slice(0, 4)}••••••••${acc.api_key.slice(-4)}`
        : `<span style="color: var(--ink-muted);">(No Key Required)</span>`;

      const lastUsed = acc.last_used_at
        ? new Date(acc.last_used_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
        : `<span style="color: var(--ink-muted);">Never</span>`;

      const row = document.createElement("tr");
      row.innerHTML = `
        <td>
          <span class="badge" style="font-weight: 600;">${escapeHtml(providerLabel)}</span>
        </td>
        <td>
          <div style="font-weight: 500; color: var(--ink-primary);">${escapeHtml(acc.name)}</div>
          <div class="mono-cell" style="font-size: 11px; color: var(--ink-muted);">${escapeHtml(acc.id)}</div>
        </td>
        <td>${statusBadge}</td>
        <td>
          <div style="display: flex; align-items: center; gap: 6px;">
            <span class="mono-cell" id="key-text-${acc.id}">${maskedKey}</span>
            ${acc.api_key ? `<button class="btn btn-secondary btn-sm btn-show-key" data-id="${acc.id}" style="padding: 2px 6px; font-size: 10px;">Reveal</button>` : ''}
          </div>
        </td>
        <td>
          <span class="mono-cell">Pri: ${acc.priority || 1} • Wt: ${acc.weight || 1}</span>
        </td>
        <td>
          <span class="mono-cell" style="color: #34d399;">✓ ${acc.success_count || 0}</span>
          <span class="mono-cell" style="color: #f43f5e; margin-left: 6px;">✕ ${acc.error_count || 0}</span>
        </td>
        <td>${lastUsed}</td>
        <td>
          <div style="display: flex; gap: 6px; align-items: center;">
            ${isCooldown ? `<button class="btn btn-secondary btn-sm btn-reset-cooldown" data-id="${acc.id}" title="Reset cooldown">Reset</button>` : ''}
            <button class="btn btn-secondary btn-sm btn-edit-account" data-id="${acc.id}">Edit</button>
            <button class="btn btn-secondary btn-sm btn-toggle-account" data-id="${acc.id}">${acc.enabled ? "Disable" : "Enable"}</button>
            <button class="btn btn-danger btn-sm btn-delete-account" data-id="${acc.id}" title="Delete account">✕</button>
          </div>
        </td>
      `;

      // Event handlers
      const revealBtn = row.querySelector(".btn-show-key");
      if (revealBtn) {
        revealBtn.addEventListener("click", () => {
          const keySpan = document.getElementById(`key-text-${acc.id}`);
          if (revealBtn.textContent === "Reveal") {
            keySpan.textContent = acc.api_key;
            revealBtn.textContent = "Hide";
          } else {
            keySpan.innerHTML = maskedKey;
            revealBtn.textContent = "Reveal";
          }
        });
      }

      const resetBtn = row.querySelector(".btn-reset-cooldown");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => resetAccountCooldown(acc.id));
      }

      row.querySelector(".btn-edit-account").addEventListener("click", () => openAccountModal(acc));
      row.querySelector(".btn-toggle-account").addEventListener("click", async () => {
        try {
          await api(`/api/accounts/${acc.id}`, {
            method: "PUT",
            body: JSON.stringify({ enabled: !acc.enabled }),
          });
          acc.enabled = !acc.enabled;
          showToast(`Account '${acc.name}' ${acc.enabled ? "enabled" : "disabled"}`);
          fetchAccounts();
        } catch (err) {
          showToast(`Failed: ${err.message}`, "error");
        }
      });

      row.querySelector(".btn-delete-account").addEventListener("click", async () => {
        if (confirm(`Delete account '${acc.name}' from ${providerLabel}?`)) {
          try {
            await api(`/api/accounts/${acc.id}`, { method: "DELETE" });
            showToast(`Account '${acc.name}' deleted`);
            fetchAccounts();
          } catch (err) {
            showToast(`Delete failed: ${err.message}`, "error");
          }
        }
      });

      el.accountsTableBody.appendChild(row);
    });
  }

  function openAccountModal(account = null, defaultProviderId = null) {
    syncAccountProviderSelects();
    if (account) {
      if (el.modalAccountTitle) el.modalAccountTitle.textContent = "Edit Provider Account";
      if (el.accountFormId) el.accountFormId.value = account.id;
      if (el.accountProviderSelect) el.accountProviderSelect.value = account.provider_id;
      if (el.accountName) el.accountName.value = account.name;
      if (el.accountApiKey) el.accountApiKey.value = account.api_key || "";
      if (el.accountBaseUrl) el.accountBaseUrl.value = account.base_url || "";
      if (el.accountPriority) el.accountPriority.value = account.priority || 1;
      if (el.accountWeight) el.accountWeight.value = account.weight || 1;
      if (el.accountEnabledToggle) el.accountEnabledToggle.checked = account.enabled !== false;
    } else {
      if (el.modalAccountTitle) el.modalAccountTitle.textContent = "Connect Provider Account";
      if (el.accountFormId) el.accountFormId.value = "";
      if (el.accountProviderSelect) {
        if (defaultProviderId) {
          el.accountProviderSelect.value = defaultProviderId;
        } else if (state.plugins.length > 0) {
          el.accountProviderSelect.value = state.plugins[0].id;
        }
      }
      if (el.accountName) el.accountName.value = "";
      if (el.accountApiKey) el.accountApiKey.value = "";
      if (el.accountBaseUrl) el.accountBaseUrl.value = "";
      if (el.accountPriority) el.accountPriority.value = 1;
      if (el.accountWeight) el.accountWeight.value = 1;
      if (el.accountEnabledToggle) el.accountEnabledToggle.checked = true;
    }

    if (el.btnToggleAccountKey && el.accountApiKey) {
      el.accountApiKey.type = "password";
      el.btnToggleAccountKey.textContent = "Show";
    }

    if (el.modalAccount) el.modalAccount.classList.add("active");
  }

  async function saveAccount() {
    const id = el.accountFormId ? el.accountFormId.value.trim() : "";
    const providerId = el.accountProviderSelect ? el.accountProviderSelect.value : "";
    const name = el.accountName ? el.accountName.value.trim() : "";
    const apiKey = el.accountApiKey ? el.accountApiKey.value.trim() : "";
    const baseUrl = el.accountBaseUrl ? el.accountBaseUrl.value.trim() : "";
    const priority = el.accountPriority ? parseInt(el.accountPriority.value, 10) || 1 : 1;
    const weight = el.accountWeight ? parseInt(el.accountWeight.value, 10) || 1 : 1;
    const enabled = el.accountEnabledToggle ? el.accountEnabledToggle.checked : true;

    if (!name) {
      showToast("Account name / label is required", "error");
      return;
    }
    if (!providerId) {
      showToast("Please select a provider", "error");
      return;
    }

    try {
      if (id) {
        await api(`/api/accounts/${id}`, {
          method: "PUT",
          body: JSON.stringify({
            name,
            api_key: apiKey,
            base_url: baseUrl || null,
            priority,
            weight,
            enabled,
          }),
        });
        showToast(`Account '${name}' updated`);
      } else {
        await api("/api/accounts", {
          method: "POST",
          body: JSON.stringify({
            provider_id: providerId,
            name,
            api_key: apiKey,
            base_url: baseUrl || null,
            priority,
            weight,
            enabled,
          }),
        });
        showToast(`Account '${name}' connected successfully`);
      }

      if (el.modalAccount) el.modalAccount.classList.remove("active");
      fetchAccounts();
    } catch (err) {
      showToast(`Save failed: ${err.message}`, "error");
    }
  }

  async function resetAccountCooldown(accountId) {
    try {
      await api(`/api/accounts/${accountId}/reset-cooldown`, { method: "POST" });
      showToast("Cooldown reset. Account is active.");
      fetchAccounts();
    } catch (err) {
      showToast(`Failed to reset cooldown: ${err.message}`, "error");
    }
  }

  // Periodic cooldown countdown updater
  setInterval(() => {
    if (state.currentTab !== "accounts") return;
    const now = Date.now() / 1000;
    const cooldownSpans = document.querySelectorAll("[data-cooldown-until]");
    cooldownSpans.forEach((span) => {
      const until = parseFloat(span.dataset.cooldownUntil);
      if (until > now) {
        const rem = Math.ceil(until - now);
        span.textContent = `⏳ Cooldown (${rem}s)`;
      } else {
        span.className = "badge badge-success";
        span.innerHTML = `<span class="status-dot"></span> Active`;
        span.removeAttribute("data-cooldown-until");
      }
    });
  }, 1000);

  // Wire Accounts Event Listeners
  if (el.btnConnectAccount) {
    el.btnConnectAccount.addEventListener("click", () => openAccountModal());
  }

  if (el.accountsProviderFilter) {
    el.accountsProviderFilter.addEventListener("change", (e) => {
      state.accountProviderFilter = e.target.value;
      renderAccountsTable();
    });
  }

  if (el.btnSaveAccount) {
    el.btnSaveAccount.addEventListener("click", saveAccount);
  }

  if (el.btnToggleAccountKey && el.accountApiKey) {
    el.btnToggleAccountKey.addEventListener("click", () => {
      if (el.accountApiKey.type === "password") {
        el.accountApiKey.type = "text";
        el.btnToggleAccountKey.textContent = "Hide";
      } else {
        el.accountApiKey.type = "password";
        el.btnToggleAccountKey.textContent = "Show";
      }
    });
  }

  // =========================================================================
  // Code Editor & Swap Tab
  // =========================================================================

  function syncEditorPluginSelect() {
    if (!el.editorPluginSelect) return;
    const currentVal = el.editorPluginSelect.value;
    el.editorPluginSelect.innerHTML = "";

    state.plugins.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.name} (${p.filename || p.id + ".py"})`;
      el.editorPluginSelect.appendChild(opt);
    });

    if (state.selectedEditorPluginId && state.plugins.some((p) => p.id === state.selectedEditorPluginId)) {
      el.editorPluginSelect.value = state.selectedEditorPluginId;
    } else if (state.plugins.length > 0) {
      el.editorPluginSelect.value = state.plugins[0].id;
      selectEditorPlugin(state.plugins[0].id);
    }
  }

  async function selectEditorPlugin(pluginId) {
    state.selectedEditorPluginId = pluginId;
    if (el.editorPluginSelect) el.editorPluginSelect.value = pluginId;
    hideEditorFeedback();

    const plugin = state.plugins.find((p) => p.id === pluginId);
    if (el.editorFilenameBadge && plugin) {
      el.editorFilenameBadge.textContent = plugin.filename || `${pluginId}.py`;
    }

    try {
      const data = await api(`/api/plugins/${pluginId}/code`);
      if (el.editorTextarea) {
        el.editorTextarea.value = data.code;
        updateLineNumbers();
      }
    } catch (err) {
      showToast(`Failed to load source code: ${err.message}`, "error");
    }
  }

  if (el.editorPluginSelect) {
    el.editorPluginSelect.addEventListener("change", (e) => {
      selectEditorPlugin(e.target.value);
    });
  }

  function updateLineNumbers() {
    if (!el.editorTextarea || !el.editorLineNumbers) return;
    const lines = el.editorTextarea.value.split("\n").length;
    let numbers = "";
    for (let i = 1; i <= lines; i++) {
      numbers += i + "\n";
    }
    el.editorLineNumbers.textContent = numbers;
  }

  if (el.editorTextarea) {
    el.editorTextarea.addEventListener("input", updateLineNumbers);
    el.editorTextarea.addEventListener("scroll", () => {
      if (el.editorLineNumbers) el.editorLineNumbers.scrollTop = el.editorTextarea.scrollTop;
    });

    // Handle Tab key for proper Python indentation (4 spaces)
    el.editorTextarea.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const start = el.editorTextarea.selectionStart;
        const end = el.editorTextarea.selectionEnd;
        el.editorTextarea.value =
          el.editorTextarea.value.substring(0, start) + "    " + el.editorTextarea.value.substring(end);
        el.editorTextarea.selectionStart = el.editorTextarea.selectionEnd = start + 4;
        updateLineNumbers();
      } else if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        saveEditorCode();
      }
    });
  }

  async function saveEditorCode() {
    const pluginId = state.selectedEditorPluginId;
    if (!pluginId) {
      showToast("No plugin selected to save", "warning");
      return;
    }

    const code = el.editorTextarea.value;
    hideEditorFeedback();

    try {
      const res = await api(`/api/plugins/${pluginId}/code`, {
        method: "PUT",
        body: JSON.stringify({ code }),
      });
      showEditorFeedback("success", `✓ Code saved & hot-reloaded successfully! (v${res.plugin.version})`);
      showToast(`Plugin '${pluginId}' updated and reloaded`);
      fetchPlugins();
    } catch (err) {
      showEditorFeedback("error", `✕ Hot-reload failed: ${err.message}`);
      showToast("Error updating plugin code", "error");
    }
  }

  if (el.editorSaveBtn) {
    el.editorSaveBtn.addEventListener("click", saveEditorCode);
  }

  function showEditorFeedback(type, message) {
    if (!el.editorFeedback) return;
    el.editorFeedback.className = `feedback-banner ${type}`;
    el.editorFeedback.textContent = message;
    el.editorFeedback.style.display = "flex";
  }

  function hideEditorFeedback() {
    if (!el.editorFeedback) return;
    el.editorFeedback.style.display = "none";
  }

  // Swap / Upload File
  if (el.editorSwapBtn && el.editorFileInput) {
    el.editorSwapBtn.addEventListener("click", () => el.editorFileInput.click());
    el.editorFileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        el.editorTextarea.value = evt.target.result;
        updateLineNumbers();
        showToast(`Loaded file '${file.name}' into editor. Click 'Save & Hot-Reload' to apply.`);
      };
      reader.readAsText(file);
      e.target.value = "";
    });
  }

  // Template Selector
  function syncTemplateSelects() {
    if (el.editorTemplateSelect) {
      el.editorTemplateSelect.innerHTML = `<option value="">-- Insert Template --</option>`;
      state.templates.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.id;
        opt.textContent = t.name;
        el.editorTemplateSelect.appendChild(opt);
      });
    }

    const newProviderTemplateSelect = document.getElementById("new-provider-template-select");
    if (newProviderTemplateSelect) {
      newProviderTemplateSelect.innerHTML = `<option value="">-- Choose Starter Template --</option>`;
      state.templates.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.id;
        opt.textContent = t.name;
        newProviderTemplateSelect.appendChild(opt);
      });
    }
  }

  if (el.editorTemplateSelect) {
    el.editorTemplateSelect.addEventListener("change", (e) => {
      const templateId = e.target.value;
      if (!templateId) return;
      const tmpl = state.templates.find((t) => t.id === templateId);
      if (tmpl && confirm(`Replace current editor content with template '${tmpl.name}'?`)) {
        el.editorTextarea.value = tmpl.code;
        updateLineNumbers();
        showToast(`Template '${tmpl.name}' inserted.`);
      }
      e.target.value = "";
    });
  }

  // =========================================================================
  // Models Management Tab
  // =========================================================================

  function renderModelsTable() {
    if (!el.modelsTableBody) return;
    el.modelsTableBody.innerHTML = "";

    const query = (el.modelsSearchInput ? el.modelsSearchInput.value : "").toLowerCase().trim();

    let allModels = [];
    state.plugins.forEach((p) => {
      (p.models || []).forEach((m) => {
        allModels.push({
          ...m,
          pluginId: p.id,
          pluginName: p.name,
          pluginEnabled: p.enabled,
        });
      });
    });

    if (query) {
      allModels = allModels.filter(
        (m) =>
          m.id.toLowerCase().includes(query) ||
          m.name.toLowerCase().includes(query) ||
          m.pluginName.toLowerCase().includes(query)
      );
    }

    if (allModels.length === 0) {
      el.modelsTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; padding: 32px; color: var(--ink-secondary);">
            No models found. Add a custom model to an enabled provider.
          </td>
        </tr>
      `;
      return;
    }

    allModels.forEach((m) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono-cell" style="cursor: pointer;" title="Click to copy ID">${escapeHtml(m.id)}</td>
        <td><strong>${escapeHtml(m.name)}</strong></td>
        <td><span class="badge">${escapeHtml(m.pluginName)}</span></td>
        <td class="mono-cell" style="color: var(--ink-muted);">${escapeHtml(m.upstream_model || m.id)}</td>
        <td>${(m.context_window || 8192).toLocaleString()} / ${(m.max_tokens || 4096).toLocaleString()}</td>
        <td>
          <label class="switch">
            <input type="checkbox" class="model-toggle" data-plugin="${m.pluginId}" data-model="${m.id}" ${m.enabled ? "checked" : ""}>
            <span class="slider"></span>
          </label>
        </td>
        <td>
          <div style="display: flex; gap: 6px;">
            <button class="btn btn-secondary btn-sm btn-test-model" data-model="${m.id}">Playground</button>
            <button class="btn btn-secondary btn-sm btn-edit-model" data-plugin="${m.pluginId}" data-model="${m.id}">Edit</button>
            <button class="btn btn-danger btn-sm btn-del-model" data-plugin="${m.pluginId}" data-model="${m.id}">✕</button>
          </div>
        </td>
      `;

      // Copy Model ID
      tr.querySelector(".mono-cell").addEventListener("click", () => {
        navigator.clipboard.writeText(m.id);
        showToast(`Copied model ID: ${m.id}`);
      });

      // Toggle Enable/Disable
      tr.querySelector(".model-toggle").addEventListener("change", async (e) => {
        try {
          await api(`/api/plugins/${m.pluginId}/models/${m.id}`, {
            method: "PUT",
            body: JSON.stringify({ enabled: e.target.checked }),
          });
          m.enabled = e.target.checked;
          showToast(`Model '${m.id}' ${m.enabled ? "enabled" : "disabled"}`);
          fetchStats();
        } catch (err) {
          e.target.checked = !e.target.checked;
          showToast(`Failed to update model: ${err.message}`, "error");
        }
      });

      // Jump to Playground
      tr.querySelector(".btn-test-model").addEventListener("click", () => {
        switchTab("playground");
        if (el.playgroundModelSelect) {
          el.playgroundModelSelect.value = m.id;
          state.selectedPlaygroundModel = m.id;
        }
      });

      // Edit Model
      tr.querySelector(".btn-edit-model").addEventListener("click", () => {
        openEditModelModal(m.pluginId, m);
      });

      // Delete Model
      tr.querySelector(".btn-del-model").addEventListener("click", async () => {
        if (confirm(`Remove model '${m.id}' from provider '${m.pluginName}'?`)) {
          try {
            await api(`/api/plugins/${m.pluginId}/models/${m.id}`, { method: "DELETE" });
            showToast(`Model '${m.id}' deleted`);
            fetchPlugins();
          } catch (err) {
            showToast(`Failed to delete model: ${err.message}`, "error");
          }
        }
      });

      el.modelsTableBody.appendChild(tr);
    });
  }

  if (el.modelsSearchInput) {
    el.modelsSearchInput.addEventListener("input", renderModelsTable);
  }

  // =========================================================================
  // Playground Chat Tab
  // =========================================================================

  function syncPlaygroundModels() {
    if (!el.playgroundModelSelect) return;
    const currentVal = el.playgroundModelSelect.value;
    el.playgroundModelSelect.innerHTML = "";

    const activeModels = [];
    state.plugins.forEach((p) => {
      if (p.enabled) {
        (p.models || []).forEach((m) => {
          if (m.enabled) {
            activeModels.push({ id: m.id, name: m.name, provider: p.name });
          }
        });
      }
    });

    if (activeModels.length === 0) {
      el.playgroundModelSelect.innerHTML = `<option value="">No active models available</option>`;
      return;
    }

    activeModels.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.name} (${m.id}) - ${m.provider}`;
      el.playgroundModelSelect.appendChild(opt);
    });

    if (activeModels.some((m) => m.id === currentVal)) {
      el.playgroundModelSelect.value = currentVal;
    } else {
      el.playgroundModelSelect.value = activeModels[0].id;
    }
    state.selectedPlaygroundModel = el.playgroundModelSelect.value;
  }

  if (el.playgroundModelSelect) {
    el.playgroundModelSelect.addEventListener("change", (e) => {
      state.selectedPlaygroundModel = e.target.value;
    });
  }

  if (el.playgroundTempInput && el.playgroundTempDisplay) {
    el.playgroundTempInput.addEventListener("input", (e) => {
      el.playgroundTempDisplay.textContent = parseFloat(e.target.value).toFixed(1);
    });
  }

  if (el.playgroundPrompt) {
    el.playgroundPrompt.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        sendPlaygroundMessage();
      }
    });
  }

  if (el.playgroundSendBtn) {
    el.playgroundSendBtn.addEventListener("click", sendPlaygroundMessage);
  }

  if (el.playgroundClearBtn) {
    el.playgroundClearBtn.addEventListener("click", () => {
      state.chatMessages = [];
      renderChatHistory();
      showToast("Playground chat cleared");
    });
  }

  async function sendPlaygroundMessage() {
    const promptText = el.playgroundPrompt ? el.playgroundPrompt.value.trim() : "";
    if (!promptText) return;

    const model = state.selectedPlaygroundModel || el.playgroundModelSelect.value;
    if (!model) {
      showToast("Please select an active model first", "warning");
      return;
    }

    // Add user message to history
    state.chatMessages.push({ role: "user", content: promptText });
    el.playgroundPrompt.value = "";
    renderChatHistory();

    // Prepare API request payload
    const systemPrompt = el.playgroundSystemInput ? el.playgroundSystemInput.value.trim() : "";
    const messagesPayload = [];
    if (systemPrompt) {
      messagesPayload.push({ role: "system", content: systemPrompt });
    }
    state.chatMessages.forEach((msg) => {
      messagesPayload.push({ role: msg.role, content: msg.content });
    });

    const temperature = el.playgroundTempInput ? parseFloat(el.playgroundTempInput.value) : 1.0;
    const maxTokens = el.playgroundMaxTokensInput && el.playgroundMaxTokensInput.value ? parseInt(el.playgroundMaxTokensInput.value, 10) : undefined;

    // Show loading assistant indicator
    const loadingMsgId = `loading-${Date.now()}`;
    const loadingElem = document.createElement("div");
    loadingElem.className = "chat-msg assistant";
    loadingElem.id = loadingMsgId;
    loadingElem.innerHTML = `
      <div class="msg-header"><span>Router Engine</span></div>
      <div class="msg-bubble" style="color: var(--ink-muted); font-style: italic;">Routing through ${escapeHtml(model)}...</div>
    `;
    el.playgroundHistory.appendChild(loadingElem);
    el.playgroundHistory.scrollTop = el.playgroundHistory.scrollHeight;

    if (el.playgroundSendBtn) {
      el.playgroundSendBtn.disabled = true;
      el.playgroundSendBtn.textContent = "Routing...";
    }

    const startTime = performance.now();
    let accumulatedText = "";

    try {
      const response = await fetch("/api/test-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: messagesPayload,
          temperature,
          max_tokens: maxTokens,
          stream: true,
        }),
      });

      if (!response.ok) {
        let errDetail = `HTTP ${response.status}`;
        try {
          const errData = await response.json();
          errDetail = errData.detail || errDetail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const contentType = response.headers.get("content-type") || "";

      if (contentType.includes("text/event-stream") && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        const bubble = loadingElem.querySelector(".msg-bubble");
        bubble.style.color = "#f4f4f5";
        bubble.style.fontStyle = "normal";
        bubble.textContent = "";

        let done = false;
        let buffer = "";

        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          if (value) {
            buffer += decoder.decode(value, { stream: !done });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || !trimmed.startsWith("data:")) continue;
              const payload = trimmed.substring(5).trim();
              if (payload === "[DONE]") continue;

              try {
                const chunk = JSON.parse(payload);
                const deltaContent =
                  chunk.choices && chunk.choices[0] && chunk.choices[0].delta
                    ? chunk.choices[0].delta.content
                    : "";
                if (deltaContent) {
                  accumulatedText += deltaContent;
                  bubble.textContent = accumulatedText;
                  el.playgroundHistory.scrollTop = el.playgroundHistory.scrollHeight;
                }
              } catch (_) {}
            }
          }
        }
      } else {
        const res = await response.json();
        accumulatedText =
          res.choices && res.choices[0] && res.choices[0].message
            ? res.choices[0].message.content
            : "(No content returned)";
      }

      const latencyMs = Math.round(performance.now() - startTime);

      state.chatMessages.push({
        role: "assistant",
        content: accumulatedText || "(Empty response)",
        telemetry: {
          latencyMs,
          promptTokens: Math.max(1, promptText.split(/\s+/).length),
          compTokens: Math.max(1, accumulatedText.split(/\s+/).length),
          model,
        },
      });

      fetchStats();
    } catch (err) {
      state.chatMessages.push({
        role: "assistant",
        content: `Error: ${err.message}`,
        isError: true,
      });
      showToast(`Playground error: ${err.message}`, "error");
    } finally {
      const loading = document.getElementById(loadingMsgId);
      if (loading) loading.remove();

      if (el.playgroundSendBtn) {
        el.playgroundSendBtn.disabled = false;
        el.playgroundSendBtn.textContent = "Send Message";
      }

      renderChatHistory();
    }
  }

  function renderChatHistory() {
    if (!el.playgroundHistory) return;
    el.playgroundHistory.innerHTML = "";

    if (state.chatMessages.length === 0) {
      el.playgroundHistory.innerHTML = `
        <div style="margin: auto; text-align: center; color: var(--ink-muted); padding: 48px;">
          <div style="font-size: 24px; margin-bottom: 8px;">◈</div>
          <div style="font-weight: 600; color: var(--ink-secondary); margin-bottom: 4px;">Playground Ready</div>
          <div style="font-size: 12px;">Type a message below or configure parameters in the left panel.</div>
        </div>
      `;
      return;
    }

    state.chatMessages.forEach((msg) => {
      const msgDiv = document.createElement("div");
      msgDiv.className = `chat-msg ${msg.role}`;

      let telemetryHtml = "";
      if (msg.telemetry) {
        telemetryHtml = `
          <div class="msg-telemetry">
            <span>⚡ ${msg.telemetry.latencyMs}ms</span>
            <span>▤ ${msg.telemetry.promptTokens} in / ${msg.telemetry.compTokens} out</span>
            <span>◈ ${escapeHtml(msg.telemetry.model)}</span>
          </div>
        `;
      }

      msgDiv.innerHTML = `
        <div class="msg-header">
          <span>${msg.role === "user" ? "User Prompt" : "Assistant Response"}</span>
        </div>
        <div class="msg-bubble" ${msg.isError ? 'style="border-color: var(--accent-red-border); color: var(--accent-red);"' : ""}>
          ${escapeHtml(msg.content)}
        </div>
        ${telemetryHtml}
      `;

      el.playgroundHistory.appendChild(msgDiv);
    });

    el.playgroundHistory.scrollTop = el.playgroundHistory.scrollHeight;
  }

  // =========================================================================
  // Request Logs Tab
  // =========================================================================

  function renderLogsTable() {
    if (!el.logsTableBody) return;
    el.logsTableBody.innerHTML = "";

    if (state.logs.length === 0) {
      el.logsTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; padding: 32px; color: var(--ink-secondary);">
            No request logs recorded yet. Send a query via the Playground or OpenAI-compatible endpoint.
          </td>
        </tr>
      `;
      return;
    }

    state.logs.forEach((log) => {
      const tr = document.createElement("tr");

      const dateStr = log.timestamp
        ? new Date(log.timestamp * 1000).toLocaleTimeString([], { hour12: false })
        : "-";

      const statusBadge =
        log.status === "success"
          ? `<span class="badge" style="background: var(--accent-green-bg); border-color: var(--accent-green-border); color: var(--accent-green);">200 OK</span>`
          : `<span class="badge" style="background: var(--accent-red-bg); border-color: var(--accent-red-border); color: var(--accent-red);">ERR</span>`;

      tr.innerHTML = `
        <td class="mono-cell" style="font-size: 11.5px; color: var(--ink-muted);">${dateStr}</td>
        <td>${statusBadge}</td>
        <td class="mono-cell"><strong>${escapeHtml(log.model)}</strong></td>
        <td><span class="badge">${escapeHtml(log.plugin_id || "router")}</span></td>
        <td class="mono-cell">${Math.round(log.latency_ms)} ms</td>
        <td class="mono-cell">${log.prompt_tokens} / ${log.completion_tokens}</td>
        <td style="color: var(--ink-muted); font-size: 12px;">${escapeHtml(log.error_message || "—")}</td>
      `;

      el.logsTableBody.appendChild(tr);
    });
  }

  if (el.logsRefreshBtn) {
    el.logsRefreshBtn.addEventListener("click", () => {
      fetchLogs();
      fetchStats();
      showToast("Logs refreshed");
    });
  }

  if (el.logsAutoToggle) {
    el.logsAutoToggle.addEventListener("change", (e) => {
      state.autoRefreshLogs = e.target.checked;
      if (state.autoRefreshLogs) {
        state.autoRefreshInterval = setInterval(() => {
          if (state.currentTab === "logs" || state.currentTab === "overview") {
            fetchLogs();
            fetchStats();
          }
        }, 5000);
        showToast("Auto-refresh enabled (5s)");
      } else {
        clearInterval(state.autoRefreshInterval);
        showToast("Auto-refresh paused");
      }
    });
  }

  // =========================================================================
  // Modals Management
  // =========================================================================

  function closeModal(modalElem) {
    if (modalElem) modalElem.classList.remove("active");
  }

  document.querySelectorAll(".modal-close").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const modal = e.target.closest(".modal-overlay");
      closeModal(modal);
    });
  });

  // Modal: New Provider
  function openNewProviderModal() {
    if (!el.modalNewProvider) return;
    document.getElementById("new-provider-filename").value = "";
    document.getElementById("new-provider-code").value = "";
    syncTemplateSelects();
    el.modalNewProvider.classList.add("active");
  }

  if (el.btnNewProvider) {
    el.btnNewProvider.addEventListener("click", openNewProviderModal);
  }

  const newProviderTemplateSelect = document.getElementById("new-provider-template-select");
  if (newProviderTemplateSelect) {
    newProviderTemplateSelect.addEventListener("change", (e) => {
      const tmplId = e.target.value;
      if (!tmplId) return;
      const tmpl = state.templates.find((t) => t.id === tmplId);
      if (tmpl) {
        document.getElementById("new-provider-filename").value = tmpl.filename;
        document.getElementById("new-provider-code").value = tmpl.code;
      }
    });
  }

  const btnSubmitNewProvider = document.getElementById("btn-submit-new-provider");
  if (btnSubmitNewProvider) {
    btnSubmitNewProvider.addEventListener("click", async () => {
      const filename = document.getElementById("new-provider-filename").value.trim();
      const code = document.getElementById("new-provider-code").value.trim();

      if (!filename || !code) {
        showToast("Filename and Python code are required", "warning");
        return;
      }

      try {
        const res = await api("/api/plugins", {
          method: "POST",
          body: JSON.stringify({ filename, code }),
        });
        showToast(`Plugin '${res.plugin.name}' created and loaded!`);
        closeModal(el.modalNewProvider);
        await fetchPlugins();
        switchTab("providers");
      } catch (err) {
        showToast(`Failed to create plugin: ${err.message}`, "error");
      }
    });
  }

  // Modal: Variables Configuration
  function openVariablesModal(pluginId) {
    const plugin = state.plugins.find((p) => p.id === pluginId);
    if (!plugin || !el.modalVariables) return;

    document.getElementById("vars-plugin-name").textContent = plugin.name;
    const body = document.getElementById("vars-modal-body");
    body.innerHTML = "";

    if (!plugin.variables || plugin.variables.length === 0) {
      body.innerHTML = `<p style="color: var(--ink-secondary); font-size: 13px;">This plugin does not declare any custom variables.</p>`;
    } else {
      plugin.variables.forEach((v) => {
        const group = document.createElement("div");
        group.className = "form-group";

        const isSecret = v.type === "secret";
        const val = v.value !== undefined && v.value !== null ? v.value : v.default || "";

        group.innerHTML = `
          <label class="form-label">
            ${escapeHtml(v.label || v.name)}
            ${v.required ? '<span style="color: var(--accent-red);">*</span>' : ""}
          </label>
          <div style="display: flex; gap: 8px;">
            <input
              type="${isSecret ? "password" : "text"}"
              class="form-input var-input"
              data-name="${escapeHtml(v.name)}"
              data-type="${escapeHtml(v.type || 'string')}"
              value="${escapeHtml(String(val))}"
              placeholder="${escapeHtml(v.default ? String(v.default) : '')}"
              style="flex: 1; font-family: var(--font-mono);"
            >
            ${isSecret ? '<button type="button" class="btn btn-secondary btn-sm toggle-mask-btn">👁</button>' : ""}
          </div>
          ${v.description ? `<span class="form-hint">${escapeHtml(v.description)}</span>` : ""}
        `;

        if (isSecret) {
          const input = group.querySelector(".var-input");
          const maskBtn = group.querySelector(".toggle-mask-btn");
          maskBtn.addEventListener("click", () => {
            if (input.type === "password") {
              input.type = "text";
              maskBtn.textContent = "🔒";
            } else {
              input.type = "password";
              maskBtn.textContent = "👁";
            }
          });
        }

        body.appendChild(group);
      });
    }

    const saveBtn = document.getElementById("btn-save-variables");
    saveBtn.onclick = async () => {
      const updatedVars = {};
      body.querySelectorAll(".var-input").forEach((inp) => {
        const type = inp.dataset.type;
        let value = inp.value;
        if (type === "number") {
          const num = Number(value);
          value = isNaN(num) ? value : num;
        } else if (type === "boolean") {
          value = value === "true" || value === "1" || value === true;
        }
        updatedVars[inp.dataset.name] = value;
      });

      try {
        await api(`/api/plugins/${pluginId}/config`, {
          method: "PUT",
          body: JSON.stringify({ variables: updatedVars }),
        });
        showToast("Variables saved successfully");
        closeModal(el.modalVariables);
        fetchPlugins();
      } catch (err) {
        showToast(`Failed to save variables: ${err.message}`, "error");
      }
    };

    el.modalVariables.classList.add("active");
  }

  // Modal: Add / Edit Custom Model
  function openAddModelModal() {
    if (!el.modalModel) return;
    document.getElementById("modal-model-title").textContent = "Add Custom Model";
    document.getElementById("model-form-id").value = "";
    document.getElementById("model-form-id").disabled = false;
    document.getElementById("model-form-name").value = "";
    document.getElementById("model-form-upstream").value = "";
    document.getElementById("model-form-context").value = 128000;
    document.getElementById("model-form-max-tokens").value = 4096;

    const pluginSelect = document.getElementById("model-form-plugin");
    pluginSelect.innerHTML = "";
    state.plugins.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.name} (${p.id})`;
      pluginSelect.appendChild(opt);
    });
    pluginSelect.disabled = false;

    const saveBtn = document.getElementById("btn-save-model");
    saveBtn.onclick = async () => {
      const pluginId = pluginSelect.value;
      const modelId = document.getElementById("model-form-id").value.trim();
      const name = document.getElementById("model-form-name").value.trim();
      const upstream = document.getElementById("model-form-upstream").value.trim();
      const context = parseInt(document.getElementById("model-form-context").value, 10) || 8192;
      const maxTokens = parseInt(document.getElementById("model-form-max-tokens").value, 10) || 4096;

      if (!modelId || !name) {
        showToast("Model ID and Display Name are required", "warning");
        return;
      }

      try {
        await api(`/api/plugins/${pluginId}/models`, {
          method: "POST",
          body: JSON.stringify({
            id: modelId,
            name,
            upstream_model: upstream || modelId,
            context_window: context,
            max_tokens: maxTokens,
            enabled: true,
          }),
        });
        showToast(`Model '${modelId}' added to ${pluginId}`);
        closeModal(el.modalModel);
        fetchPlugins();
      } catch (err) {
        showToast(`Failed to add model: ${err.message}`, "error");
      }
    };

    el.modalModel.classList.add("active");
  }

  function openEditModelModal(pluginId, model) {
    if (!el.modalModel) return;
    document.getElementById("modal-model-title").textContent = `Edit Model: ${model.id}`;
    document.getElementById("model-form-id").value = model.id;
    document.getElementById("model-form-id").disabled = true;
    document.getElementById("model-form-name").value = model.name;
    document.getElementById("model-form-upstream").value = model.upstream_model || "";
    document.getElementById("model-form-context").value = model.context_window || 8192;
    document.getElementById("model-form-max-tokens").value = model.max_tokens || 4096;

    const pluginSelect = document.getElementById("model-form-plugin");
    pluginSelect.innerHTML = `<option value="${pluginId}">${pluginId}</option>`;
    pluginSelect.disabled = true;

    const saveBtn = document.getElementById("btn-save-model");
    saveBtn.onclick = async () => {
      const name = document.getElementById("model-form-name").value.trim();
      const upstream = document.getElementById("model-form-upstream").value.trim();
      const context = parseInt(document.getElementById("model-form-context").value, 10) || 8192;
      const maxTokens = parseInt(document.getElementById("model-form-max-tokens").value, 10) || 4096;

      try {
        await api(`/api/plugins/${pluginId}/models/${model.id}`, {
          method: "PUT",
          body: JSON.stringify({
            name,
            upstream_model: upstream || model.id,
            context_window: context,
            max_tokens: maxTokens,
          }),
        });
        showToast(`Model '${model.id}' updated`);
        closeModal(el.modalModel);
        fetchPlugins();
      } catch (err) {
        showToast(`Failed to update model: ${err.message}`, "error");
      }
    };

    el.modalModel.classList.add("active");
  }

  if (el.btnAddModel) {
    el.btnAddModel.addEventListener("click", openAddModelModal);
  }

  // =========================================================================
  // Utility Functions
  // =========================================================================

  function escapeHtml(str) {
    if (typeof str !== "string") return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // =========================================================================
  // App Initialization
  // =========================================================================

  async function init() {
    await Promise.all([
      fetchStats(),
      fetchPlugins(),
      fetchAccounts(),
      fetchTemplates(),
      fetchLogs(),
    ]);

    renderChatHistory();
  }

  init();
})();
