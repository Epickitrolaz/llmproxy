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
    apiKeys: [],
    rules: [],
    stats: {},
    logs: [],
    statusProviderFilter: "",
    activeSnippetLang: "python",
    selectedSnippetModel: "test-model",
    temporaryApiKey: null,
    rulesSearchQuery: "",
    fullscreenOriginalCode: "",
    fullscreenCodeDirty: false,
    fullscreenOriginalStrategy: null,
    fullscreenOriginalVariables: null,
    currentFullscreenPluginId: null,
    currentFullscreenTab: "code",
    selectedPlaygroundModel: null,
    chatMessages: [],
    autoRefreshLogs: false,
    autoRefreshInterval: null,
    cachedQuotaData: null,
    hiddenLimits: {}, // { [accountId]: Set of bucketIds/limitNames }
  };

  // Load hidden limits from localStorage
  try {
    const savedHidden = localStorage.getItem("llmproxy_hidden_limits");
    if (savedHidden) {
      const parsed = JSON.parse(savedHidden);
      state.hiddenLimits = {};
      for (const [k, v] of Object.entries(parsed)) {
        state.hiddenLimits[k] = new Set(v);
      }
    }
  } catch (_) {
    state.hiddenLimits = {};
  }

  function saveHiddenLimits() {
    try {
      const serializable = {};
      for (const [k, v] of Object.entries(state.hiddenLimits)) {
        serializable[k] = Array.from(v);
      }
      localStorage.setItem("llmproxy_hidden_limits", JSON.stringify(serializable));
    } catch (_) {}
  }

  // DOM Elements Cache
  const el = {
    mainContent: document.querySelector(".main-content"),
    navTabs: document.querySelectorAll(".nav-tab"),
    tabPanels: document.querySelectorAll(".tab-panel"),
    // Stats
    statRequests: document.getElementById("stat-total-requests"),
    statSuccessRate: document.getElementById("stat-success-rate"),
    statActiveModels: document.getElementById("stat-active-models"),
    statAvgLatency: document.getElementById("stat-avg-latency"),
    // Overview
    apiKeysCountBadge: document.getElementById("api-keys-count-badge"),
    apiKeysTableBody: document.getElementById("api-keys-table-body"),
    btnCreateApiKey: document.getElementById("btn-create-api-key"),
    snippetCode: document.getElementById("snippet-code"),
    snippetModelSelect: document.getElementById("snippet-model-select"),
    headerEndpointDisplay: document.getElementById("header-endpoint-display"),
    copySnippetBtn: document.getElementById("copy-snippet-btn"),
    copyEndpointBtn: document.getElementById("copy-endpoint-btn"),
    // Providers
    statProvidersCount: document.getElementById("stat-providers-count"),
    statTotalAccounts: document.getElementById("stat-total-accounts"),
    statActiveAccounts: document.getElementById("stat-active-accounts"),
    statCooldownAccounts: document.getElementById("stat-cooldown-accounts"),
    providersGrid: document.getElementById("providers-grid"),
    btnNewProvider: document.getElementById("btn-new-provider"),
    btnTemplatesGallery: document.getElementById("btn-templates-gallery"),
    // Rules
    rulesTableBody: document.getElementById("rules-table-body"),
    btnCreateRule: document.getElementById("btn-create-rule"),
    // Playground
    playgroundModelSelect: document.getElementById("playground-model-select"),
    playgroundTempInput: document.getElementById("playground-temp-input"),
    playgroundTempDisplay: document.getElementById("playground-temp-display"),
    playgroundTokensInput: document.getElementById("playground-tokens-input"),
    playgroundSystemInput: document.getElementById("playground-system-input"),
    playgroundHistory: document.getElementById("playground-history"),
    playgroundPrompt: document.getElementById("playground-prompt"),
    playgroundSendBtn: document.getElementById("playground-send-btn"),
    playgroundClearBtn: document.getElementById("playground-clear-btn"),
    // Status
    statusProviderTabs: document.getElementById("status-provider-tabs"),
    statusAccountsGrid: document.getElementById("status-accounts-grid"),
    logsTableBody: document.getElementById("logs-table-body"),
    logsRefreshBtn: document.getElementById("logs-refresh-btn"),
    logsAutoToggle: document.getElementById("logs-auto-toggle"),
    // Modals
    modalNewProvider: document.getElementById("modal-new-provider"),
    modalTemplates: document.getElementById("modal-templates"),
    templatesGalleryList: document.getElementById("templates-gallery-list"),
    modalRule: document.getElementById("modal-rule"),
    modalApiKey: document.getElementById("modal-api-key"),
    modalAccount: document.getElementById("modal-account"),
    modalModel: document.getElementById("modal-model"),
    modalProviderFullscreen: document.getElementById("modal-provider-fullscreen"),
    modalConfirm: document.getElementById("modal-confirm"),
    toastContainer: document.getElementById("toast-container"),
    // Rules search
    rulesSearchInput: document.getElementById("rules-search-input"),
    ruleTargetsSearchInput: document.getElementById("rule-targets-search-input"),
    // Code editor layers
    fsEditorHighlight: document.getElementById("fs-editor-highlight"),
    fsEditorHighlightCode: document.getElementById("fs-editor-highlight-code"),
    // Config panel
    fsBtnSaveConfig: document.getElementById("fs-btn-save-config"),
    fsSwitchWaitInput: document.getElementById("fs-switch-wait-input"),
  };

  // Modal Animation Helpers (Opening and Closing Animations)
  function openModal(modalEl) {
    if (!modalEl) return;
    modalEl.classList.remove("closing");
    modalEl.classList.add("active");
  }

  function closeModal(modalEl) {
    if (!modalEl || !modalEl.classList.contains("active")) return;
    if (modalEl === el.modalAccount && typeof stopOAuthPolling === "function") {
      stopOAuthPolling();
      resetGoogleOAuthUI();
    }
    modalEl.classList.add("closing");
    setTimeout(() => {
      modalEl.classList.remove("active");
      modalEl.classList.remove("closing");
    }, 180);
  }

  // Themed Custom Confirmation Dialog System
  let confirmResolve = null;

  function customConfirm(message, { title = "Are you sure?", confirmText = "Confirm", cancelText = "Cancel", danger = true } = {}) {
    return new Promise((resolve) => {
      confirmResolve = resolve;
      const modal = document.getElementById("modal-confirm");
      const titleEl = document.getElementById("confirm-dialog-title");
      const msgEl = document.getElementById("confirm-dialog-message");
      const confirmBtn = document.getElementById("confirm-dialog-confirm");
      const cancelBtn = document.getElementById("confirm-dialog-cancel");
      const closeBtn = document.getElementById("confirm-dialog-close");

      if (titleEl) titleEl.textContent = title;
      if (msgEl) msgEl.textContent = message;
      if (confirmBtn) {
        confirmBtn.textContent = confirmText;
        confirmBtn.className = danger ? "btn btn-danger" : "btn btn-primary";
      }
      if (cancelBtn) cancelBtn.textContent = cancelText;

      openModal(modal);

      function handleAction(result) {
        closeModal(modal);
        if (confirmResolve) {
          const r = confirmResolve;
          confirmResolve = null;
          r(result);
        }
      }

      if (confirmBtn) {
        confirmBtn.onclick = () => handleAction(true);
      }
      if (cancelBtn) {
        cancelBtn.onclick = () => handleAction(false);
      }
      if (closeBtn) {
        closeBtn.onclick = () => handleAction(false);
      }
    });
  }

  // Python Code Color Coding Highlighter
  function highlightPython(code) {
    if (!code) return "";
    const tokenRegex = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(@\w+)|(\b(?:def|class|return|if|elif|else|while|for|in|try|except|finally|raise|import|from|as|with|async|await|yield|lambda|pass|break|continue|global|nonlocal|assert|del|is|not|and|or)\b)|(\b(?:True|False|None|self|cls|super|int|str|float|bool|list|dict|set|tuple|len|range|print|isinstance|type|Exception|BaseException)\b)|(\b\d+(?:\.\d+)?\b)|(\b[a-zA-Z_]\w*(?=\s*\())|([+\-*/%&|^~<>=!:]+)/gm;

    let result = "";
    let lastIndex = 0;
    let match;

    while ((match = tokenRegex.exec(code)) !== null) {
      if (match.index > lastIndex) {
        result += escapeHtml(code.slice(lastIndex, match.index));
      }

      const [full, comment, str, dec, kw, builtin, num, fn, op] = match;
      if (comment) {
        result += `<span class="tok-comment">${escapeHtml(comment)}</span>`;
      } else if (str) {
        result += `<span class="tok-str">${escapeHtml(str)}</span>`;
      } else if (dec) {
        result += `<span class="tok-dec">${escapeHtml(dec)}</span>`;
      } else if (kw) {
        result += `<span class="tok-kw">${escapeHtml(kw)}</span>`;
      } else if (builtin) {
        result += `<span class="tok-builtin">${escapeHtml(builtin)}</span>`;
      } else if (num) {
        result += `<span class="tok-num">${escapeHtml(num)}</span>`;
      } else if (fn) {
        result += `<span class="tok-fn">${escapeHtml(fn)}</span>`;
      } else if (op) {
        result += `<span class="tok-op">${escapeHtml(op)}</span>`;
      } else {
        result += escapeHtml(full);
      }

      lastIndex = tokenRegex.lastIndex;
    }

    if (lastIndex < code.length) {
      result += escapeHtml(code.slice(lastIndex));
    }

    return result;
  }

  // Toast Notifications
  function showToast(message, type = "info") {
    if (!el.toastContainer) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    let icon = "✓";
    if (type === "error") icon = "✕";
    if (type === "warning") icon = "⚠";

    toast.innerHTML = `<span style="color: ${type === 'error' ? '#f43f5e' : '#10b981'}; font-weight: bold;">${icon}</span> <span>${escapeHtml(message)}</span>`;
    el.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      toast.style.transition = "all 0.2s ease";
      setTimeout(() => toast.remove(), 200);
    }, 3200);
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Themed SVG Icon for Pin (minimalist monochrome)
  function renderPinSvg(pinned = false, size = 13) {
    if (pinned) {
      return `<svg class="pin-svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M16 3H8a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 16.24V17h14v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 11.76V7h1a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1z"></path>
        <line x1="12" y1="17" x2="12" y2="22" stroke-width="2"></line>
      </svg>`;
    }
    return `<svg class="pin-svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M16 3H8a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 16.24V17h14v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 11.76V7h1a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1z"></path>
      <line x1="12" y1="17" x2="12" y2="22"></line>
    </svg>`;
  }

  // Robust Cross-Browser Clipboard Helper
  // Fixes "navigator.clipboard is undefined" when accessed over HTTP IP (e.g. http://10.0.0.110:8080)
  async function copyToClipboard(text) {
    if (!text && text !== "") return false;
    if (navigator.clipboard && window.isSecureContext && typeof navigator.clipboard.writeText === "function") {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (err) {
        console.warn("navigator.clipboard.writeText failed, using fallback:", err);
      }
    }

    try {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.top = "-9999px";
      textarea.style.left = "-9999px";
      textarea.style.opacity = "0";
      textarea.setAttribute("readonly", "");
      document.body.appendChild(textarea);
      textarea.select();
      textarea.setSelectionRange(0, 999999);
      const successful = document.execCommand("copy");
      document.body.removeChild(textarea);
      return !!successful;
    } catch (err) {
      console.error("Clipboard execCommand fallback failed:", err);
      return false;
    }
  }

  // API Client
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
          if (errData.detail) {
            errorDetail = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
          } else if (errData.error && errData.error.message) {
            errorDetail = errData.error.message;
          }
        } catch (_) {}
        throw new Error(errorDetail);
      }

      const contentType = res.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        return await res.json();
      }
      return await res.text();
    } catch (err) {
      console.error(`API Request Failed: ${path}`, err);
      throw err;
    }
  }

  // =========================================================================
  // CUSTOM DROPDOWNS COMPONENT WITH SEARCH BAR
  // =========================================================================

  function enhanceSelect(selectEl) {
    if (!selectEl || selectEl.dataset.customEnhanced === "true") {
      updateCustomDropdown(selectEl);
      return;
    }

    selectEl.dataset.customEnhanced = "true";
    selectEl.style.display = "none";

    const wrapper = document.createElement("div");
    wrapper.className = "custom-dropdown";

    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "custom-dropdown-trigger";

    const labelSpan = document.createElement("span");
    labelSpan.className = "custom-dropdown-trigger-label";

    const arrowSpan = document.createElement("span");
    arrowSpan.className = "custom-dropdown-arrow";
    arrowSpan.innerHTML = "▼";

    trigger.appendChild(labelSpan);
    trigger.appendChild(arrowSpan);

    const menu = document.createElement("div");
    menu.className = "custom-dropdown-menu";

    const searchBox = document.createElement("div");
    searchBox.className = "custom-dropdown-search-box";

    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.className = "custom-dropdown-search-input";
    searchInput.placeholder = "Search options...";
    searchBox.appendChild(searchInput);

    const optionsList = document.createElement("div");
    optionsList.className = "custom-dropdown-options";

    const emptyMsg = document.createElement("div");
    emptyMsg.className = "custom-dropdown-empty";
    emptyMsg.textContent = "No matches found";
    emptyMsg.style.display = "none";

    menu.appendChild(searchBox);
    menu.appendChild(optionsList);
    menu.appendChild(emptyMsg);

    wrapper.appendChild(trigger);
    wrapper.appendChild(menu);
    selectEl.parentNode.insertBefore(wrapper, selectEl.nextSibling);

    wrapper._selectEl = selectEl;
    wrapper._trigger = trigger;
    wrapper._labelSpan = labelSpan;
    wrapper._menu = menu;
    wrapper._searchInput = searchInput;
    wrapper._optionsList = optionsList;
    wrapper._emptyMsg = emptyMsg;
    selectEl._customDropdown = wrapper;

    function populateOptions() {
      optionsList.innerHTML = "";
      const selectedIndex = selectEl.selectedIndex;
      let selectedText = "Select...";

      Array.from(selectEl.options).forEach((opt, idx) => {
        const item = document.createElement("div");
        item.className = "custom-dropdown-option";
        item.dataset.value = opt.value;
        item.textContent = opt.text || opt.value;
        if (idx === selectedIndex) {
          item.classList.add("selected");
          selectedText = opt.text || opt.value;
        }

        item.addEventListener("click", (e) => {
          e.stopPropagation();
          selectEl.value = opt.value;
          wrapper.querySelectorAll(".custom-dropdown-option").forEach((o) => o.classList.remove("selected"));
          item.classList.add("selected");
          labelSpan.textContent = opt.text || opt.value;
          closeDropdown();
          selectEl.dispatchEvent(new Event("change", { bubbles: true }));
        });

        optionsList.appendChild(item);
      });

      if (selectEl.options.length > 0 && selectedIndex >= 0) {
        selectedText = selectEl.options[selectedIndex].text || selectEl.options[selectedIndex].value;
      }
      labelSpan.textContent = selectedText;
    }

    populateOptions();

    function filterOptions(query) {
      const q = (query || "").toLowerCase().trim();
      let visibleCount = 0;
      wrapper.querySelectorAll(".custom-dropdown-option").forEach((opt) => {
        const text = (opt.textContent || "").toLowerCase();
        if (!q || text.includes(q)) {
          opt.classList.remove("hidden");
          visibleCount++;
        } else {
          opt.classList.add("hidden");
        }
      });
      emptyMsg.style.display = visibleCount === 0 ? "block" : "none";
    }

    searchInput.addEventListener("input", (e) => {
      filterOptions(e.target.value);
    });

    function openDropdown() {
      // Close other custom dropdowns
      document.querySelectorAll(".custom-dropdown.active").forEach((d) => {
        if (d !== wrapper) d.classList.remove("active");
      });
      wrapper.classList.add("active");
      searchInput.value = "";
      filterOptions("");
      setTimeout(() => searchInput.focus(), 50);
    }

    function closeDropdown() {
      wrapper.classList.remove("active");
    }

    trigger.addEventListener("click", (e) => {
      e.stopPropagation();
      if (wrapper.classList.contains("active")) {
        closeDropdown();
      } else {
        openDropdown();
      }
    });

    // Close on outside click
    document.addEventListener("click", (e) => {
      if (!wrapper.contains(e.target)) {
        closeDropdown();
      }
    });

    // Close on ESC
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && wrapper.classList.contains("active")) {
        closeDropdown();
      }
    });
  }

  function updateCustomDropdown(selectEl) {
    if (!selectEl || !selectEl._customDropdown) return;
    const wrapper = selectEl._customDropdown;
    const optionsList = wrapper._optionsList;
    const labelSpan = wrapper._labelSpan;
    if (!optionsList || !labelSpan) return;

    optionsList.innerHTML = "";
    const selectedIndex = selectEl.selectedIndex;
    let selectedText = "Select...";

    Array.from(selectEl.options).forEach((opt, idx) => {
      const item = document.createElement("div");
      item.className = "custom-dropdown-option";
      item.dataset.value = opt.value;
      item.textContent = opt.text || opt.value;
      if (idx === selectedIndex) {
        item.classList.add("selected");
        selectedText = opt.text || opt.value;
      }

      item.addEventListener("click", (e) => {
        e.stopPropagation();
        selectEl.value = opt.value;
        wrapper.querySelectorAll(".custom-dropdown-option").forEach((o) => o.classList.remove("selected"));
        item.classList.add("selected");
        labelSpan.textContent = opt.text || opt.value;
        wrapper.classList.remove("active");
        selectEl.dispatchEvent(new Event("change", { bubbles: true }));
      });

      optionsList.appendChild(item);
    });

    if (selectEl.options.length > 0 && selectedIndex >= 0) {
      selectedText = selectEl.options[selectedIndex].text || selectEl.options[selectedIndex].value;
    }
    labelSpan.textContent = selectedText;
  }

  function enhanceAllSelects(root = document) {
    root.querySelectorAll("select").forEach((s) => enhanceSelect(s));
  }

  // =========================================================================
  // TAB NAVIGATION
  // =========================================================================

  function switchTab(tabId) {
    state.currentTab = tabId;
    if (el.mainContent) {
      el.mainContent.classList.toggle("wide-layout", tabId === "playground" || tabId === "logs" || tabId === "status");
    }
    el.navTabs.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabId);
    });
    el.tabPanels.forEach((p) => {
      p.classList.toggle("active", p.id === `tab-${tabId}`);
    });

    if (tabId === "overview") {
      fetchStats();
      fetchApiKeys();
    } else if (tabId === "providers") {
      fetchPlugins();
      fetchAccounts();
    } else if (tabId === "rules") {
      fetchRules();
    } else if (tabId === "playground") {
      populatePlaygroundModels();
    } else if (tabId === "status" || tabId === "logs") {
      fetchStatusData(true);
    }
  }

  el.navTabs.forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // =========================================================================
  // TAB 1: OVERVIEW & LOCAL API KEYS
  // =========================================================================

  async function fetchStats() {
    try {
      const data = await api("/api/stats");
      state.stats = data;
      if (el.statRequests) el.statRequests.textContent = data.total_requests || 0;
      if (el.statSuccessRate) el.statSuccessRate.textContent = `${data.success_rate || 100}%`;
      if (el.statActiveModels) el.statActiveModels.textContent = data.active_models_count || 0;
      if (el.statAvgLatency) el.statAvgLatency.textContent = `${data.avg_latency_ms || 0} ms`;
      if (el.statProvidersCount) el.statProvidersCount.textContent = data.active_plugins_count || 0;
      if (el.statTotalAccounts) el.statTotalAccounts.textContent = data.total_accounts || 0;
      if (el.statActiveAccounts) el.statActiveAccounts.textContent = data.active_accounts || 0;
    } catch (_) {}
  }

  async function fetchApiKeys() {
    try {
      const data = await api("/api/keys");
      state.apiKeys = (data || []).filter((k) => !k.is_temporary);
      renderApiKeysTable();
    } catch (err) {
      console.error("Failed to load API keys", err);
    }
  }

  function renderApiKeysTable() {
    if (!el.apiKeysTableBody) return;
    if (el.apiKeysCountBadge) el.apiKeysCountBadge.textContent = `${state.apiKeys.length} keys`;
    el.apiKeysTableBody.innerHTML = "";

    if (state.apiKeys.length === 0) {
      el.apiKeysTableBody.innerHTML = `
        <tr>
          <td colspan="9" style="text-align: center; color: var(--ink-muted); padding: 24px;">
            No local API keys created yet. When created, incoming requests to <code>/v1</code> must supply <code>Authorization: Bearer &lt;key&gt;</code>.
          </td>
        </tr>
      `;
      return;
    }

    state.apiKeys.forEach((k) => {
      const tr = document.createElement("tr");
      const maskedSecret = k.key.length > 14
        ? `${k.key.slice(0, 7)}...${k.key.slice(-4)}`
        : k.key;
      const lastUsed = k.last_used_at
        ? new Date(k.last_used_at * 1000).toLocaleString()
        : "Never";

      tr.innerHTML = `
        <td><strong>${escapeHtml(k.name)}</strong></td>
        <td>
          <div style="display: flex; align-items: center; gap: 6px;">
            <code style="font-family: var(--font-mono); font-size: 11.5px; color: var(--ink-secondary);">${escapeHtml(maskedSecret)}</code>
            <button class="btn btn-secondary btn-sm btn-copy-key" data-key="${escapeHtml(k.key)}" style="padding: 2px 6px; font-size: 11px;">Copy</button>
          </div>
        </td>
        <td>
          <span class="badge ${k.enabled ? 'badge-active' : 'badge-muted'}">
            ${k.enabled ? "Active" : "Disabled"}
          </span>
        </td>
        <td style="font-family: var(--font-mono);">${k.total_requests || 0}</td>
        <td style="font-family: var(--font-mono);">${k.prompt_tokens || 0}</td>
        <td style="font-family: var(--font-mono);">${k.completion_tokens || 0}</td>
        <td style="font-family: var(--font-mono); font-weight: 600;">${k.total_tokens || 0}</td>
        <td style="font-size: 11.5px; color: var(--ink-muted);">${escapeHtml(lastUsed)}</td>
        <td>
          <div style="display: flex; gap: 6px;">
            <button class="btn btn-secondary btn-sm btn-edit-key" data-id="${k.id}">Edit</button>
            <button class="btn btn-danger btn-sm btn-delete-key" data-id="${k.id}">Delete</button>
          </div>
        </td>
      `;

      tr.querySelector(".btn-copy-key").addEventListener("click", async () => {
        const ok = await copyToClipboard(k.key);
        if (ok) {
          showToast("API key copied to clipboard");
        } else {
          showToast("Failed to copy API key", "error");
        }
      });

      tr.querySelector(".btn-edit-key").addEventListener("click", () => {
        openApiKeyModal(k);
      });

      tr.querySelector(".btn-delete-key").addEventListener("click", async () => {
        const ok = await customConfirm(`Delete local API key '${k.name}'?`, {
          title: "Delete API Key",
          confirmText: "Delete",
          danger: true,
        });
        if (ok) {
          try {
            await api(`/api/keys/${k.id}`, { method: "DELETE" });
            showToast(`API key '${k.name}' deleted`);
            fetchApiKeys();
          } catch (err) {
            showToast(`Failed to delete: ${err.message}`, "error");
          }
        }
      });

      el.apiKeysTableBody.appendChild(tr);
    });
  }

  function openApiKeyModal(key = null) {
    const idInput = document.getElementById("api-key-form-id");
    const nameInput = document.getElementById("api-key-form-name");
    const secretInput = document.getElementById("api-key-form-secret");
    const enabledInput = document.getElementById("api-key-form-enabled");
    const secretGroup = document.getElementById("api-key-secret-group");
    const title = document.getElementById("modal-api-key-title");

    if (key) {
      title.textContent = "Edit Local API Key";
      idInput.value = key.id;
      nameInput.value = key.name;
      enabledInput.checked = !!key.enabled;
      if (secretGroup) secretGroup.style.display = "none";
    } else {
      title.textContent = "Create Local API Key";
      idInput.value = "";
      nameInput.value = "";
      if (secretInput) secretInput.value = "";
      enabledInput.checked = true;
      if (secretGroup) secretGroup.style.display = "block";
    }

    openModal(el.modalApiKey);
  }

  if (el.btnCreateApiKey) {
    el.btnCreateApiKey.addEventListener("click", () => openApiKeyModal(null));
  }

  const btnSubmitApiKey = document.getElementById("btn-submit-api-key");
  if (btnSubmitApiKey) {
    btnSubmitApiKey.addEventListener("click", async () => {
      const idInput = document.getElementById("api-key-form-id");
      const nameInput = document.getElementById("api-key-form-name");
      const secretInput = document.getElementById("api-key-form-secret");
      const enabledInput = document.getElementById("api-key-form-enabled");

      const name = nameInput.value.trim();
      if (!name) {
        showToast("Key name is required", "error");
        return;
      }

      try {
        if (idInput.value) {
          // Edit
          await api(`/api/keys/${idInput.value}`, {
            method: "PUT",
            body: JSON.stringify({ name, enabled: enabledInput.checked }),
          });
          showToast(`API key '${name}' updated`);
        } else {
          // Create
          const keySecret = secretInput ? secretInput.value.trim() : "";
          await api("/api/keys", {
            method: "POST",
            body: JSON.stringify({ name, key: keySecret || undefined, enabled: enabledInput.checked }),
          });
          showToast(`API key '${name}' created`);
        }
        closeModal(el.modalApiKey);
        fetchApiKeys();
        updateSnippet();
      } catch (err) {
        showToast(`Failed to save key: ${err.message}`, "error");
      }
    });
  }

  // Snippets Generator (Default to temporary testing key)
  async function fetchTemporaryApiKey() {
    try {
      const data = await api("/api/keys/temporary");
      if (data && data.key) {
        state.temporaryApiKey = data.key;
      }
    } catch (err) {
      console.warn("Could not fetch temporary key:", err);
    }
  }

  function getDetectedBaseUrl() {
    if (typeof window !== "undefined" && window.location && window.location.origin) {
      return `${window.location.origin}/v1`;
    }
    return "http://localhost:8000/v1";
  }

  function updateSnippet() {
    if (!el.snippetCode) return;
    const baseUrl = getDetectedBaseUrl();
    const model = state.selectedSnippetModel || "test-model";
    const apiKey = state.temporaryApiKey || (state.apiKeys && state.apiKeys.length > 0 ? state.apiKeys[0].key : "sk-proxy-demo");

    if (state.activeSnippetLang === "python") {
      el.snippetCode.textContent = `# Note: Using temporary testing key (expires in 30 min or 10 requests)
from openai import OpenAI

client = OpenAI(
    base_url="${baseUrl}",
    api_key="${apiKey}"
)

response = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello LLMProxy!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()`;
    } else if (state.activeSnippetLang === "curl") {
      el.snippetCode.textContent = `# Note: Using temporary testing key (expires in 30 min or 10 requests)
curl -X POST ${baseUrl}/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKey}" \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Hello LLMProxy!"}],
    "stream": true
  }'`;
    } else if (state.activeSnippetLang === "cursor") {
      el.snippetCode.textContent = `// OpenCode / Cursor / Cline Settings (Testing key: valid for 30 min or 10 requests)
{
  "openai.baseURL": "${baseUrl}",
  "openai.apiKey": "${apiKey}",
  "model": "${model}"
}`;
    }
  }

  document.querySelectorAll(".snippet-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".snippet-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.activeSnippetLang = tab.dataset.lang;
      updateSnippet();
    });
  });

  // Initialize endpoint display with active origin
  if (el.headerEndpointDisplay && window.location.origin) {
    el.headerEndpointDisplay.textContent = `${window.location.origin}/v1`;
  }

  if (el.copySnippetBtn) {
    el.copySnippetBtn.addEventListener("click", async () => {
      if (el.snippetCode) {
        const ok = await copyToClipboard(el.snippetCode.textContent);
        if (ok) {
          showToast("Integration snippet copied to clipboard");
        } else {
          showToast("Failed to copy snippet", "error");
        }
      }
    });
  }

  if (el.copyEndpointBtn) {
    el.copyEndpointBtn.addEventListener("click", async () => {
      const url = (el.headerEndpointDisplay && el.headerEndpointDisplay.textContent) || `${window.location.origin}/v1`;
      const ok = await copyToClipboard(url);
      if (ok) {
        showToast("Base URL endpoint copied");
      } else {
        showToast("Failed to copy URL", "error");
      }
    });
  }

  const copyGoogleBtn = document.getElementById("btn-copy-google-url");
  if (copyGoogleBtn) {
    copyGoogleBtn.addEventListener("click", async () => {
      const authLink = document.getElementById("btn-open-google-auth");
      if (authLink && authLink.href) {
        const ok = await copyToClipboard(authLink.href);
        if (ok) {
          showToast("Sign-in link copied to clipboard");
        } else {
          showToast("Failed to copy link", "error");
        }
      }
    });
  }

  // =========================================================================
  // TAB 2: PROVIDERS & PLUGINS
  // =========================================================================

  async function fetchPlugins() {
    try {
      const data = await api("/api/plugins");
      state.plugins = data || [];
      if (el.statProvidersCount) el.statProvidersCount.textContent = state.plugins.length;
      renderProvidersGrid();
      populateSnippetModels();
    } catch (err) {
      console.error("Failed to fetch plugins", err);
    }
  }

  async function loadTemplates() {
    try {
      const data = await api("/api/templates");
      state.templates = data || [];
      renderTemplateDropdowns();
      renderTemplatesGallery();
    } catch (err) {
      console.error("Failed to load templates", err);
    }
  }

  function renderTemplateDropdowns() {
    const fsSelect = document.getElementById("fs-editor-template-select");
    const newSelect = document.getElementById("new-provider-template-select");

    [fsSelect, newSelect].forEach((sel) => {
      if (!sel) return;
      const currentVal = sel.value;
      sel.innerHTML = '<option value="">-- Choose Starter Template --</option>';
      state.templates.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.id;
        opt.textContent = `${t.name} (${t.filename})`;
        sel.appendChild(opt);
      });
      if (currentVal) sel.value = currentVal;
      enhanceSelect(sel);
    });
  }

  function renderTemplatesGallery() {
    const list = document.getElementById("templates-gallery-list");
    if (!list) return;
    list.innerHTML = "";

    if (state.templates.length === 0) {
      list.innerHTML = `<div style="text-align: center; color: var(--ink-muted); padding: 24px;">No templates available.</div>`;
      return;
    }

    state.templates.forEach((t) => {
      const card = document.createElement("div");
      card.className = "card";
      card.style.padding = "16px";
      card.style.display = "flex";
      card.style.justifyContent = "space-between";
      card.style.alignItems = "center";
      card.style.gap = "16px";

      card.innerHTML = `
        <div style="flex: 1;">
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <strong style="font-size: 14px; color: var(--ink-primary);">${escapeHtml(t.name)}</strong>
            <code style="font-family: var(--font-mono); font-size: 11px; color: var(--ink-muted);">${escapeHtml(t.filename)}</code>
          </div>
          <p style="font-size: 12px; color: var(--ink-secondary); margin: 0; line-height: 1.4;">${escapeHtml(t.description)}</p>
        </div>
        <button class="btn btn-primary btn-sm btn-install-template" data-id="${t.id}" style="white-space: nowrap;">
          + Add Provider
        </button>
      `;

      card.querySelector(".btn-install-template").addEventListener("click", async () => {
        try {
          await api(`/api/templates/${t.id}/add`, { method: "POST" });
          showToast(`Provider '${t.name}' added successfully`);
          closeModal(el.modalTemplates);
          fetchPlugins();
        } catch (err) {
          showToast(`Failed to add template: ${err.message}`, "error");
        }
      });

      list.appendChild(card);
    });
  }

  async function fetchAccounts() {
    try {
      const data = await api("/api/accounts");
      state.accounts = data || [];
      if (el.statTotalAccounts) el.statTotalAccounts.textContent = state.accounts.length;
      if (el.statActiveAccounts) {
        el.statActiveAccounts.textContent = state.accounts.filter((a) => a.enabled).length;
      }
      if (el.statCooldownAccounts) {
        const now = Date.now() / 1000;
        el.statCooldownAccounts.textContent = state.accounts.filter(
          (a) => a.rate_limited_until && a.rate_limited_until > now
        ).length;
      }
      populateSnippetModels();
    } catch (_) {}
  }

  function renderProvidersGrid() {
    if (!el.providersGrid) return;
    el.providersGrid.innerHTML = "";

    if (state.plugins.length === 0) {
      el.providersGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 48px 24px; text-align: center; background: var(--bg-card); border: 1px dashed var(--border-solid); border-radius: var(--radius-card);">
          <h3 style="font-size: 16px; margin-bottom: 8px;">No Provider Plugins Configured</h3>
          <p style="color: var(--ink-secondary); margin-bottom: 20px;">By default no providers are pre-configured. You can add one of the built-in templates below or create a custom one.</p>
          <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
            <button class="btn btn-primary" id="btn-empty-add-template">+ Add from Template</button>
            <button class="btn btn-secondary" id="btn-quick-starter">+ Create Custom Provider</button>
          </div>
        </div>
      `;
      const tplBtn = document.getElementById("btn-empty-add-template");
      if (tplBtn) tplBtn.addEventListener("click", () => openModal(el.modalTemplates));
      const quickBtn = document.getElementById("btn-quick-starter");
      if (quickBtn) quickBtn.addEventListener("click", () => openNewProviderModal());
      return;
    }

    state.plugins.forEach((p) => {
      const card = document.createElement("div");
      card.className = `provider-card ${p.pinned ? 'is-pinned' : ''}`;

      const modelsCount = p.models ? p.models.length : 0;
      const connectedAccounts = (state.accounts || []).filter((a) => a.provider_id === p.id);
      const isPinned = !!p.pinned;

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
            <div style="display: flex; align-items: center; gap: 8px;">
              <label class="switch" title="Toggle plugin enabled status">
                <input type="checkbox" class="provider-enable-toggle" data-id="${p.id}" ${p.enabled ? "checked" : ""}>
                <span class="slider"></span>
              </label>
            </div>
          </div>
          <p class="provider-desc">${escapeHtml(p.description || "Provider plugin for upstream model routing.")}</p>
          <div class="provider-models-badge-group">
            <span class="badge ${connectedAccounts.length > 0 ? 'badge-active' : 'badge-muted'}">${connectedAccounts.length} accounts connected</span>
            <span class="badge">${modelsCount} models</span>
          </div>
        </div>
        <div class="provider-footer">
          <span style="font-size: 12px; color: var(--ink-muted); font-family: var(--font-mono);">${escapeHtml(p.id)}</span>
          <div class="provider-actions">
            <button class="btn btn-primary btn-sm btn-provider-edit" data-id="${p.id}">Edit</button>
            <button class="btn btn-secondary btn-sm btn-provider-refresh" data-id="${p.id}" title="Hot reload plugin">Refresh</button>
            <button class="btn btn-danger btn-sm btn-provider-delete" data-id="${p.id}" title="Delete plugin">Delete</button>
          </div>
        </div>
      `;

      // Event: Enable toggle
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

      // Action 1: Edit -> Opens Fullscreen Configuration Window
      card.querySelector(".btn-provider-edit").addEventListener("click", () => {
        openFullscreenProviderModal(p.id);
      });

      // Action 2: Refresh -> Hot reloads plugin
      card.querySelector(".btn-provider-refresh").addEventListener("click", async () => {
        try {
          await api(`/api/plugins/${p.id}/reload`, { method: "POST" });
          showToast(`Plugin '${p.name}' hot-reloaded successfully`);
          fetchPlugins();
        } catch (err) {
          showToast(`Reload failed: ${err.message}`, "error");
        }
      });

      // Action 3: Delete -> Removes plugin
      card.querySelector(".btn-provider-delete").addEventListener("click", async () => {
        const ok = await customConfirm(`Are you sure you want to delete plugin '${p.name}' and remove its Python file?`, {
          title: "Delete Provider Plugin",
          confirmText: "Delete Plugin",
          danger: true,
        });
        if (ok) {
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

  let _snippetModelListenerAttached = false;

  function populateSnippetModels() {
    if (!el.snippetModelSelect) return;
    el.snippetModelSelect.innerHTML = "";

    const connectedAccounts = (state.accounts || []).filter((a) => a.enabled);
    const hasConnectedProvider = connectedAccounts.length > 0;

    // If no provider account is connected, default to test-model
    if (!hasConnectedProvider) {
      state.selectedSnippetModel = "test-model";
      const opt = document.createElement("option");
      opt.value = "test-model";
      opt.textContent = "test-model (Example)";
      opt.selected = true;
      el.snippetModelSelect.appendChild(opt);
      enhanceSelect(el.snippetModelSelect);
      updateSnippet();
      return;
    }

    // Once a provider is connected, populate actual models
    const connectedProvIds = new Set(connectedAccounts.map((a) => a.provider_id));
    const allModels = [];

    // Prioritize models from connected providers
    state.plugins.forEach((p) => {
      if (connectedProvIds.has(p.id)) {
        (p.models || []).forEach((m) => {
          if (m.enabled && !allModels.includes(m.id)) allModels.push(m.id);
        });
      }
    });

    // Other enabled models
    state.plugins.forEach((p) => {
      (p.models || []).forEach((m) => {
        if (m.enabled && !allModels.includes(m.id)) allModels.push(m.id);
      });
    });

    state.rules.forEach((r) => {
      if (r.enabled && !allModels.includes(r.model_id)) {
        allModels.unshift(r.model_id);
      }
    });

    if (allModels.length === 0) {
      allModels.push("test-model");
    }

    if (!state.selectedSnippetModel || state.selectedSnippetModel === "test-model" || !allModels.includes(state.selectedSnippetModel)) {
      state.selectedSnippetModel = allModels[0];
    }

    allModels.forEach((mid) => {
      const opt = document.createElement("option");
      opt.value = mid;
      opt.textContent = mid;
      if (mid === state.selectedSnippetModel) opt.selected = true;
      el.snippetModelSelect.appendChild(opt);
    });

    enhanceSelect(el.snippetModelSelect);
    updateSnippet();

    if (!_snippetModelListenerAttached) {
      el.snippetModelSelect.addEventListener("change", (e) => {
        state.selectedSnippetModel = e.target.value;
        updateSnippet();
      });
      _snippetModelListenerAttached = true;
    }
  }

  // =========================================================================
  // FULLSCREEN PROVIDER CONFIGURATION WINDOW
  // =========================================================================

  async function openFullscreenProviderModal(pluginId) {
    state.currentFullscreenPluginId = pluginId;
    state.fullscreenCodeDirty = false;
    state.fullscreenOriginalStrategy = null;
    state.fullscreenOriginalVariables = null;
    const plugin = state.plugins.find((p) => p.id === pluginId);
    if (!plugin) return;

    const titleEl = document.getElementById("fs-provider-title");
    const idBadge = document.getElementById("fs-provider-id-badge");
    const fnBadge = document.getElementById("fs-provider-filename-badge");
    const pinBtn = document.getElementById("fs-provider-pin-btn");
    const enableToggle = document.getElementById("fs-provider-enabled-toggle");

    if (titleEl) titleEl.textContent = `Configure ${plugin.name}`;
    if (idBadge) idBadge.textContent = plugin.id;
    if (fnBadge) fnBadge.textContent = plugin.filename || `${plugin.id}.py`;

    if (pinBtn) {
      pinBtn.remove();
    }

    if (enableToggle) {
      enableToggle.checked = !!plugin.enabled;
      enableToggle.onchange = async (e) => {
        try {
          await api(`/api/plugins/${plugin.id}/config`, {
            method: "PUT",
            body: JSON.stringify({ enabled: e.target.checked }),
          });
          plugin.enabled = e.target.checked;
          showToast(`Plugin '${plugin.name}' status updated`);
          fetchPlugins();
        } catch (err) {
          e.target.checked = !e.target.checked;
          showToast(`Update failed: ${err.message}`, "error");
        }
      };
    }

    // Load code
    await loadFullscreenPluginCode(pluginId);
    // Render sub-tabs content
    renderFullscreenAccounts(pluginId);
    renderFullscreenConfig(plugin);
    renderFullscreenModels(plugin);

    switchFullscreenTab("code");
    openModal(el.modalProviderFullscreen);
  }

  function getFullscreenVariablesSnapshot() {
    const vars = {};
    document.querySelectorAll(".fs-var-input").forEach((inp) => {
      const vName = inp.dataset.var;
      if (vName) {
        vars[vName] = inp.type === "checkbox" ? inp.checked : inp.value;
      }
    });
    return vars;
  }

  function isFullscreenStrategyDirty() {
    if (!state.fullscreenOriginalStrategy) return false;
    const select = document.getElementById("fs-strategy-select");
    const cdInput = document.getElementById("fs-cooldown-input");
    const swInput = document.getElementById("fs-switch-wait-input");
    const retriesInput = document.getElementById("fs-retries-input");
    if (!select || !cdInput || !retriesInput) return false;
    const orig = state.fullscreenOriginalStrategy;
    if (select.value !== orig.strategy) return true;
    if (String(cdInput.value) !== String(orig.cooldown)) return true;
    if (swInput && String(swInput.value) !== String(orig.switchWait)) return true;
    if (String(retriesInput.value) !== String(orig.retries)) return true;
    return false;
  }

  function isFullscreenVariablesDirty() {
    if (!state.fullscreenOriginalVariables) return false;
    const current = getFullscreenVariablesSnapshot();
    const orig = state.fullscreenOriginalVariables;
    const keys = Object.keys(orig);
    const currentKeys = Object.keys(current);
    if (keys.length !== currentKeys.length) return true;
    for (const k of keys) {
      if (orig[k] !== current[k]) return true;
    }
    return false;
  }

  function isFullscreenDirty() {
    const textarea = document.getElementById("fs-editor-code");
    const codeDirty = state.fullscreenCodeDirty ||
      (textarea && textarea.value !== (state.fullscreenOriginalCode || ""));
    return Boolean(codeDirty || isFullscreenStrategyDirty() || isFullscreenVariablesDirty());
  }

  async function requestCloseFullscreen() {
    if (isFullscreenDirty()) {
      const ok = await customConfirm(
        "You have unsaved changes in this provider configuration. Are you sure you want to discard your edits and exit?",
        {
          title: "Exit Configuration",
          confirmText: "Discard & Exit",
          cancelText: "Keep Editing",
          danger: true,
        }
      );
      if (!ok) return;
    }
    state.fullscreenCodeDirty = false;
    state.fullscreenOriginalStrategy = null;
    state.fullscreenOriginalVariables = null;
    closeModal(el.modalProviderFullscreen);
  }

  function switchFullscreenTab(tabKey) {
    state.currentFullscreenTab = tabKey;
    document.querySelectorAll(".modal-fullscreen-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.fsTab === tabKey);
    });
    document.querySelectorAll(".modal-fullscreen-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `fs-panel-${tabKey}`);
    });
  }

  document.querySelectorAll(".modal-fullscreen-tab").forEach((btn) => {
    btn.addEventListener("click", () => switchFullscreenTab(btn.dataset.fsTab));
  });

  async function loadFullscreenPluginCode(pluginId) {
    const textarea = document.getElementById("fs-editor-code");
    const feedback = document.getElementById("fs-editor-feedback");
    if (feedback) feedback.className = "feedback-banner";

    try {
      const data = await api(`/api/plugins/${pluginId}/code`);
      state.fullscreenOriginalCode = data.code || "";
      state.fullscreenCodeDirty = false;
      if (textarea) textarea.value = state.fullscreenOriginalCode;
      updateFsEditor(true);
    } catch (err) {
      if (feedback) {
        feedback.className = "feedback-banner error";
        feedback.textContent = `Failed to load source code: ${err.message}`;
      }
    }
  }

  let highlightTimer = null;
  let cachedLineCount = 0;

  function updateFsEditor(immediate = false) {
    const textarea = document.getElementById("fs-editor-code");
    const lineNumbers = document.getElementById("fs-editor-line-numbers");
    const highlightCode = document.getElementById("fs-editor-highlight-code");
    if (!textarea) return;

    const val = textarea.value || "";

    // Fast line-number update: only if line count changed, using pre-wrap string with textContent
    const lines = (val.match(/\n/g) || []).length + 1;
    if (lineNumbers && lines !== cachedLineCount) {
      cachedLineCount = lines;
      let numStr = "";
      for (let i = 1; i <= lines; i++) {
        numStr += i + "\n";
      }
      lineNumbers.textContent = numStr;
    }

    if (!highlightCode) return;

    if (immediate) {
      if (highlightTimer) {
        clearTimeout(highlightTimer);
        highlightTimer = null;
      }
      highlightCode.innerHTML = highlightPython(val) + "\n";
      return;
    }

    // Instantly update text content to prevent visual lag while typing
    highlightCode.textContent = val + "\n";

    // Debounce expensive syntax highlighting
    if (highlightTimer) clearTimeout(highlightTimer);
    highlightTimer = setTimeout(() => {
      highlightTimer = null;
      highlightCode.innerHTML = highlightPython(val) + "\n";
    }, 60);
  }

  const fsEditorTextarea = document.getElementById("fs-editor-code");
  let scrollRaf = null;
  if (fsEditorTextarea) {
    fsEditorTextarea.addEventListener("input", () => {
      state.fullscreenCodeDirty = (fsEditorTextarea.value !== (state.fullscreenOriginalCode || ""));
      updateFsEditor();
    });

    fsEditorTextarea.addEventListener("scroll", () => {
      if (scrollRaf) return;
      scrollRaf = requestAnimationFrame(() => {
        scrollRaf = null;
        const lineNumbers = document.getElementById("fs-editor-line-numbers");
        const highlight = document.getElementById("fs-editor-highlight");
        const top = fsEditorTextarea.scrollTop;
        const left = fsEditorTextarea.scrollLeft;
        if (lineNumbers) lineNumbers.scrollTop = top;
        if (highlight) {
          highlight.scrollTop = top;
          highlight.scrollLeft = left;
        }
      });
    }, { passive: true });

    fsEditorTextarea.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const start = fsEditorTextarea.selectionStart;
        const end = fsEditorTextarea.selectionEnd;
        fsEditorTextarea.value = fsEditorTextarea.value.substring(0, start) + "    " + fsEditorTextarea.value.substring(end);
        fsEditorTextarea.selectionStart = fsEditorTextarea.selectionEnd = start + 4;
        updateFsEditor(true);
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        saveFullscreenPluginCode();
      }
    });
  }

  async function saveFullscreenPluginCode() {
    const pluginId = state.currentFullscreenPluginId;
    if (!pluginId) return;
    const textarea = document.getElementById("fs-editor-code");
    const feedback = document.getElementById("fs-editor-feedback");
    if (!textarea) return;

    try {
      if (feedback) {
        feedback.className = "feedback-banner info";
        feedback.textContent = "Validating syntax and reloading plugin...";
      }
      await api(`/api/plugins/${pluginId}/code`, {
        method: "PUT",
        body: JSON.stringify({ code: textarea.value }),
      });
      state.fullscreenOriginalCode = textarea.value;
      state.fullscreenCodeDirty = false;
      if (feedback) {
        feedback.className = "feedback-banner success";
        feedback.textContent = "Code updated and hot-reloaded successfully!";
      }
      showToast("Plugin code hot-reloaded successfully");
      fetchPlugins();
    } catch (err) {
      if (feedback) {
        feedback.className = "feedback-banner error";
        feedback.textContent = `Save failed: ${err.message}`;
      }
      showToast(`Save failed: ${err.message}`, "error");
    }
  }

  const fsEditorSaveBtn = document.getElementById("fs-editor-save-btn");
  if (fsEditorSaveBtn) {
    fsEditorSaveBtn.addEventListener("click", saveFullscreenPluginCode);
  }

  const fsEditorSwapBtn = document.getElementById("fs-editor-swap-btn");
  const fsEditorFileInput = document.getElementById("fs-editor-file-input");
  if (fsEditorSwapBtn && fsEditorFileInput) {
    fsEditorSwapBtn.addEventListener("click", () => fsEditorFileInput.click());
    fsEditorFileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        if (fsEditorTextarea) {
          fsEditorTextarea.value = evt.target.result;
          state.fullscreenCodeDirty = true;
          updateFsEditor();
          showToast(`Loaded ${file.name}. Click Save & Hot-Reload to apply.`);
        }
      };
      reader.readAsText(file);
    });
  }

  // Fullscreen: Accounts
  function renderFullscreenAccounts(pluginId) {
    const tbody = document.getElementById("fs-accounts-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    const isAg = pluginId === "antigravity";
    const toolbar = document.getElementById("fs-accounts-toolbar");
    if (toolbar) {
      if (isAg) {
        toolbar.innerHTML = `
          <button class="btn btn-primary btn-sm btn-fs-google-oauth" style="display: inline-flex; align-items: center; gap: 6px;">
            ${GOOGLE_SVG}
            <span>Sign in with Google</span>
          </button>
        `;
        const gOauthBtn = toolbar.querySelector(".btn-fs-google-oauth");
        if (gOauthBtn) gOauthBtn.addEventListener("click", () => openAccountModal(null, "antigravity"));
      } else {
        toolbar.innerHTML = `<button class="btn btn-primary btn-sm" id="fs-btn-add-account">+ Connect Account</button>`;
        const btn = toolbar.querySelector("#fs-btn-add-account");
        if (btn) btn.addEventListener("click", () => openAccountModal(null, pluginId));
      }
    }

    const accounts = (state.accounts || []).filter((a) => a.provider_id === pluginId);
    if (accounts.length === 0) {
      const emptyMsg = isAg
        ? 'No Google accounts connected yet. Click "Sign in with Google" above to connect an account.'
        : 'No accounts connected for this provider. Add keys or tokens to enable load balancing.';
      tbody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align: center; color: var(--ink-muted); padding: 24px;">
            ${escapeHtml(emptyMsg)}
          </td>
        </tr>
      `;
      return;
    }

    accounts.forEach((acc) => {
      const tr = document.createElement("tr");
      const maskedKey = acc.api_key && acc.api_key.length > 12
        ? `${acc.api_key.slice(0, 6)}...${acc.api_key.slice(-4)}`
        : (acc.api_key || "OAuth / None");

      const inCooldown = acc.rate_limited_until && acc.rate_limited_until > Date.now() / 1000;
      const statusBadge = inCooldown
        ? `<span class="badge" style="color: var(--accent-red); border-color: var(--accent-red-border);">Cooldown (${Math.round(acc.rate_limited_until - Date.now() / 1000)}s)</span>`
        : acc.enabled
          ? `<span class="badge badge-active">Active</span>`
          : `<span class="badge badge-muted">Disabled</span>`;

      tr.innerHTML = `
        <td><strong>${escapeHtml(acc.name)}</strong></td>
        <td><code style="font-family: var(--font-mono); font-size: 11.5px;">${escapeHtml(maskedKey)}</code></td>
        <td>${acc.priority || 1}</td>
        <td>${acc.weight || 1}</td>
        <td>${statusBadge}</td>
        <td>${acc.success_count || 0} / ${acc.error_count || 0}</td>
        <td style="font-size: 11.5px; color: var(--ink-muted);">${acc.last_used_at ? new Date(acc.last_used_at * 1000).toLocaleTimeString() : 'Never'}</td>
        <td>
          <div style="display: flex; gap: 6px;">
            ${inCooldown ? `<button class="btn btn-secondary btn-sm btn-reset-cd" data-id="${acc.id}">Reset</button>` : ''}
            <button class="btn btn-secondary btn-sm btn-edit-acc" data-id="${acc.id}">Edit</button>
            <button class="btn btn-danger btn-sm btn-del-acc" data-id="${acc.id}">Delete</button>
          </div>
        </td>
      `;

      if (inCooldown) {
        tr.querySelector(".btn-reset-cd").addEventListener("click", async () => {
          try {
            await api(`/api/accounts/${acc.id}/reset-cooldown`, { method: "POST" });
            showToast("Cooldown cleared");
            await fetchAccounts();
            renderFullscreenAccounts(pluginId);
          } catch (err) {
            showToast(err.message, "error");
          }
        });
      }

      tr.querySelector(".btn-edit-acc").addEventListener("click", () => openAccountModal(acc, pluginId));
      tr.querySelector(".btn-del-acc").addEventListener("click", async () => {
        const ok = await customConfirm(`Delete account '${acc.name}'?`, {
          title: "Delete Account",
          confirmText: "Delete",
          danger: true,
        });
        if (ok) {
          try {
            await api(`/api/accounts/${acc.id}`, { method: "DELETE" });
            showToast("Account deleted");
            await fetchAccounts();
            renderFullscreenAccounts(pluginId);
          } catch (err) {
            showToast(err.message, "error");
          }
        }
      });

      tbody.appendChild(tr);
    });
  }

  const fsBtnAddAccount = document.getElementById("fs-btn-add-account");
  if (fsBtnAddAccount) {
    fsBtnAddAccount.addEventListener("click", () => {
      openAccountModal(null, state.currentFullscreenPluginId);
    });
  }

  // Combined Fullscreen Configuration (Load Balancing on Left, Variables on Right)
  const LOAD_BALANCING_VAR_NAMES = new Set([
    "load_balancing_strategy",
    "account_switch_cooldown",
    "account_switch_wait",
    "max_retries",
  ]);

  function renderFullscreenConfig(plugin) {
    const select = document.getElementById("fs-strategy-select");
    const cdInput = document.getElementById("fs-cooldown-input");
    const swInput = document.getElementById("fs-switch-wait-input");
    const retriesInput = document.getElementById("fs-retries-input");

    const vars = {};
    (plugin.variables || []).forEach((v) => { vars[v.name] = v.value !== undefined ? v.value : v.default; });

    const defaultStrat = (plugin.id === "antigravity") ? "fallback" : "round_robin";
    const currentStrat = vars.load_balancing_strategy || defaultStrat;

    if (select) {
      select.value = currentStrat;
      enhanceSelect(select);
    }
    if (cdInput) cdInput.value = vars.account_switch_cooldown !== undefined ? vars.account_switch_cooldown : 60;
    if (swInput) swInput.value = vars.account_switch_wait !== undefined ? vars.account_switch_wait : 60;
    if (retriesInput) retriesInput.value = vars.max_retries !== undefined ? vars.max_retries : 2;

    state.fullscreenOriginalStrategy = {
      strategy: select ? select.value : currentStrat,
      cooldown: cdInput ? String(cdInput.value) : "60",
      switchWait: swInput ? String(swInput.value) : "60",
      retries: retriesInput ? String(retriesInput.value) : "2",
    };

    // Render variables on the right, excluding load balancing settings so Failover Retries is not duplicated!
    const container = document.getElementById("fs-variables-container");
    if (!container) return;
    container.innerHTML = "";

    const declaredVars = (plugin.variables || []).filter((v) => !LOAD_BALANCING_VAR_NAMES.has(v.name));
    if (declaredVars.length === 0) {
      container.innerHTML = `<div style="color: var(--ink-muted); font-size: 13px;">This plugin does not declare custom variables beyond load balancing.</div>`;
      state.fullscreenOriginalVariables = {};
      return;
    }

    declaredVars.forEach((v) => {
      const grp = document.createElement("div");
      grp.className = "form-group";

      const val = (v.value !== undefined && v.value !== null && v.value !== "") ? v.value : (v.default !== undefined && v.default !== null ? v.default : "");
      if (v.type === "boolean") {
        grp.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <label class="form-label">${escapeHtml(v.label || v.name)}</label>
              ${v.description ? `<span class="form-hint">${escapeHtml(v.description)}</span>` : ""}
            </div>
            <label class="switch">
              <input type="checkbox" class="fs-var-input" data-var="${escapeHtml(v.name)}" ${val ? "checked" : ""}>
              <span class="slider"></span>
            </label>
          </div>
        `;
      } else if (v.type === "select" && v.options && v.options.length) {
        let optHtml = "";
        v.options.forEach((opt) => {
          optHtml += `<option value="${escapeHtml(opt)}" ${String(val) === String(opt) ? "selected" : ""}>${escapeHtml(opt)}</option>`;
        });
        grp.innerHTML = `
          <label class="form-label">${escapeHtml(v.label || v.name)} ${v.required ? '<span style="color: var(--accent-red);">*</span>' : ''}</label>
          <select class="select-input fs-var-input" data-var="${escapeHtml(v.name)}">${optHtml}</select>
          ${v.description ? `<span class="form-hint">${escapeHtml(v.description)}</span>` : ""}
        `;
      } else {
        grp.innerHTML = `
          <label class="form-label">${escapeHtml(v.label || v.name)} ${v.required ? '<span style="color: var(--accent-red);">*</span>' : ''}</label>
          <input type="${v.type === 'secret' ? 'password' : 'text'}" class="form-input fs-var-input" data-var="${escapeHtml(v.name)}" value="${escapeHtml(val)}" placeholder="${escapeHtml(v.default || '')}">
          ${v.description ? `<span class="form-hint">${escapeHtml(v.description)}</span>` : ""}
        `;
      }
      container.appendChild(grp);
    });

    state.fullscreenOriginalVariables = getFullscreenVariablesSnapshot();
  }

  const fsBtnSaveConfig = document.getElementById("fs-btn-save-config");
  if (fsBtnSaveConfig) {
    fsBtnSaveConfig.addEventListener("click", async () => {
      const pluginId = state.currentFullscreenPluginId;
      if (!pluginId) return;

      const strategy = document.getElementById("fs-strategy-select").value;
      const cd = parseFloat(document.getElementById("fs-cooldown-input").value) || 60;
      const sw = parseFloat(document.getElementById("fs-switch-wait-input").value) || 60;
      const retries = parseInt(document.getElementById("fs-retries-input").value, 10) || 2;

      const vars = {
        load_balancing_strategy: strategy,
        account_switch_cooldown: cd,
        account_switch_wait: sw,
        max_retries: retries,
      };

      document.querySelectorAll(".fs-var-input").forEach((inp) => {
        const vName = inp.dataset.var;
        vars[vName] = inp.type === "checkbox" ? inp.checked : inp.value;
      });

      try {
        await api(`/api/plugins/${pluginId}/config`, {
          method: "PUT",
          body: JSON.stringify({ variables: vars }),
        });
        state.fullscreenOriginalStrategy = {
          strategy: strategy,
          cooldown: String(cd),
          switchWait: String(sw),
          retries: String(retries),
        };
        state.fullscreenOriginalVariables = getFullscreenVariablesSnapshot();
        showToast("Configuration saved");
        fetchPlugins();
      } catch (err) {
        showToast(`Save failed: ${err.message}`, "error");
      }
    });
  }

  // Fullscreen: Models
  function renderFullscreenModels(plugin) {
    const tbody = document.getElementById("fs-models-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    const models = plugin.models || [];
    if (models.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--ink-muted); padding: 20px;">No models configured.</td></tr>`;
      return;
    }

    models.forEach((m) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${escapeHtml(m.id)}</strong></td>
        <td>${escapeHtml(m.name || m.id)}</td>
        <td><code style="font-family: var(--font-mono); font-size: 11.5px;">${escapeHtml(m.upstream_model || m.id)}</code></td>
        <td style="font-family: var(--font-mono);">${m.context_window || 8192} / ${m.max_tokens || 4096}</td>
        <td>
          <label class="switch">
            <input type="checkbox" class="fs-model-toggle" data-mid="${m.id}" ${m.enabled ? "checked" : ""}>
            <span class="slider"></span>
          </label>
        </td>
        <td>
          <div style="display: flex; gap: 6px; align-items: center;">
            <button class="btn btn-secondary btn-sm btn-test btn-fs-test-model" data-mid="${m.id}" title="Test model response">Test</button>
            <button class="btn btn-danger btn-sm btn-fs-del-model" data-mid="${m.id}">Delete</button>
          </div>
        </td>
      `;

      const testBtn = tr.querySelector(".btn-fs-test-model");
      if (testBtn) {
        testBtn.addEventListener("click", async () => {
          testBtn.disabled = true;
          testBtn.classList.remove("is-success", "is-error");
          testBtn.classList.add("is-testing");
          testBtn.textContent = "Testing...";

          const startT = performance.now();
          try {
            const res = await api("/api/test-chat", {
              method: "POST",
              body: JSON.stringify({
                model: m.id,
                prompt: "ping",
                max_tokens: 5,
              }),
            });
            const latency = Math.round(performance.now() - startT);
            testBtn.classList.remove("is-testing");
            testBtn.classList.add("is-success");
            testBtn.textContent = "✓ OK";
            testBtn.title = `Responded in ${res.latency_ms || latency}ms`;
            showToast(`Model '${m.id}' responded in ${res.latency_ms || latency}ms`);
          } catch (err) {
            testBtn.classList.remove("is-testing");
            testBtn.classList.add("is-error");
            testBtn.textContent = "✕ Fail";
            testBtn.title = `Error: ${err.message}`;
            showToast(`Model '${m.id}' test failed: ${err.message}`, "error");
          } finally {
            testBtn.disabled = false;
          }
        });
      }

      tr.querySelector(".fs-model-toggle").addEventListener("change", async (e) => {
        try {
          await api(`/api/plugins/${plugin.id}/models/${m.id}`, {
            method: "PUT",
            body: JSON.stringify({ enabled: e.target.checked }),
          });
          m.enabled = e.target.checked;
          showToast(`Model '${m.id}' ${m.enabled ? "enabled" : "disabled"}`);
          fetchPlugins();
        } catch (err) {
          e.target.checked = !e.target.checked;
          showToast(`Failed to update: ${err.message}`, "error");
        }
      });

      tr.querySelector(".btn-fs-del-model").addEventListener("click", async () => {
        const ok = await customConfirm(`Remove model '${m.id}' from plugin?`, {
          title: "Remove Model",
          confirmText: "Remove Model",
          danger: true,
        });
        if (ok) {
          try {
            await api(`/api/plugins/${plugin.id}/models/${m.id}`, { method: "DELETE" });
            showToast(`Model '${m.id}' removed`);
            await fetchPlugins();
            const updatedP = state.plugins.find((p) => p.id === plugin.id);
            if (updatedP) renderFullscreenModels(updatedP);
          } catch (err) {
            showToast(`Delete failed: ${err.message}`, "error");
          }
        }
      });

      tbody.appendChild(tr);
    });
  }

  const fsBtnAddModel = document.getElementById("fs-btn-add-model");
  if (fsBtnAddModel) {
    fsBtnAddModel.addEventListener("click", () => {
      document.getElementById("model-form-plugin-id").value = state.currentFullscreenPluginId;
      document.getElementById("model-form-id").value = "";
      document.getElementById("model-form-name").value = "";
      document.getElementById("model-form-upstream").value = "";
      openModal(el.modalModel);
    });
  }

  // Model creation submit
  const btnSaveModel = document.getElementById("btn-save-model");
  if (btnSaveModel) {
    btnSaveModel.addEventListener("click", async () => {
      const pId = document.getElementById("model-form-plugin-id").value;
      const mId = document.getElementById("model-form-id").value.trim();
      const mName = document.getElementById("model-form-name").value.trim() || mId;
      const mUpstream = document.getElementById("model-form-upstream").value.trim() || mId;
      const mContext = parseInt(document.getElementById("model-form-context").value, 10) || 128000;
      const mMaxTokens = parseInt(document.getElementById("model-form-max-tokens").value, 10) || 4096;

      if (!mId) {
        showToast("Model ID is required", "error");
        return;
      }

      try {
        await api(`/api/plugins/${pId}/models`, {
          method: "POST",
          body: JSON.stringify({
            id: mId,
            name: mName,
            upstream_model: mUpstream,
            context_window: mContext,
            max_tokens: mMaxTokens,
            enabled: true,
          }),
        });
        showToast(`Model '${mId}' added`);
        closeModal(el.modalModel);
        await fetchPlugins();
        const updatedP = state.plugins.find((p) => p.id === pId);
        if (updatedP) renderFullscreenModels(updatedP);
      } catch (err) {
        showToast(`Failed to add model: ${err.message}`, "error");
      }
    });
  }

  // =========================================================================
  // TAB 3: CUSTOM LOAD BALANCING RULES
  // =========================================================================

  async function fetchRules() {
    try {
      const data = await api("/api/load-balancing-rules");
      state.rules = data || [];
      renderRulesTable();
      populateSnippetModels();
    } catch (err) {
      console.error("Failed to load rules", err);
    }
  }

  function renderRulesTable() {
    if (!el.rulesTableBody) return;
    el.rulesTableBody.innerHTML = "";

    let rulesToDisplay = state.rules;
    if (state.rulesSearchQuery) {
      const q = state.rulesSearchQuery.toLowerCase().trim();
      rulesToDisplay = state.rules.filter((r) => {
        return (
          (r.model_id && r.model_id.toLowerCase().includes(q)) ||
          (r.name && r.name.toLowerCase().includes(q)) ||
          (r.strategy && r.strategy.toLowerCase().includes(q)) ||
          (r.targets && r.targets.some((t) => t.toLowerCase().includes(q))) ||
          (r.description && r.description.toLowerCase().includes(q))
        );
      });
    }

    if (rulesToDisplay.length === 0) {
      const msg = state.rulesSearchQuery
        ? `No custom rules matching '${escapeHtml(state.rulesSearchQuery)}'.`
        : `No custom load balancing rules created yet. Click "+ Create Custom Rule" to expose a virtual model routing across multiple providers.`;
      el.rulesTableBody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align: center; color: var(--ink-muted); padding: 32px;">
            ${msg}
          </td>
        </tr>
      `;
      return;
    }

    rulesToDisplay.forEach((r) => {
      const tr = document.createElement("tr");
      const targetsBadges = (r.targets || []).map((t) => `<span class="badge">${escapeHtml(t)}</span>`).join(" ");

      tr.innerHTML = `
        <td><strong style="color: var(--accent-white);">${escapeHtml(r.model_id)}</strong></td>
        <td>${escapeHtml(r.name || r.model_id)}</td>
        <td><code style="font-family: var(--font-mono); text-transform: uppercase; font-size: 11.5px;">${escapeHtml(r.strategy)}</code></td>
        <td><div style="display: flex; flex-wrap: wrap; gap: 4px;">${targetsBadges}</div></td>
        <td>
          <label class="switch">
            <input type="checkbox" class="rule-enable-toggle" data-id="${r.id}" ${r.enabled ? "checked" : ""}>
            <span class="slider"></span>
          </label>
        </td>
        <td>
          <div style="display: flex; gap: 6px;">
            <button class="btn btn-secondary btn-sm btn-edit-rule" data-id="${r.id}">Edit</button>
            <button class="btn btn-danger btn-sm btn-del-rule" data-id="${r.id}">Delete</button>
          </div>
        </td>
      `;

      tr.querySelector(".rule-enable-toggle").addEventListener("change", async (e) => {
        try {
          await api(`/api/load-balancing-rules/${r.id}`, {
            method: "PUT",
            body: JSON.stringify({ enabled: e.target.checked }),
          });
          r.enabled = e.target.checked;
          showToast(`Rule '${r.name}' ${r.enabled ? "enabled" : "disabled"}`);
          populateSnippetModels();
        } catch (err) {
          e.target.checked = !e.target.checked;
          showToast(`Update failed: ${err.message}`, "error");
        }
      });

      tr.querySelector(".btn-edit-rule").addEventListener("click", () => openRuleModal(r));
      tr.querySelector(".btn-del-rule").addEventListener("click", async () => {
        const ok = await customConfirm(`Delete load balancing rule for '${r.model_id}'?`, {
          title: "Delete Load Balancing Rule",
          confirmText: "Delete Rule",
          danger: true,
        });
        if (ok) {
          try {
            await api(`/api/load-balancing-rules/${r.id}`, { method: "DELETE" });
            showToast("Rule deleted");
            fetchRules();
          } catch (err) {
            showToast(`Delete failed: ${err.message}`, "error");
          }
        }
      });

      el.rulesTableBody.appendChild(tr);
    });
  }

  const rulesSearchInput = document.getElementById("rules-search-input");
  if (rulesSearchInput) {
    rulesSearchInput.addEventListener("input", (e) => {
      state.rulesSearchQuery = e.target.value;
      renderRulesTable();
    });
  }

  function openRuleModal(rule = null) {
    const idInput = document.getElementById("rule-form-id");
    const mIdInput = document.getElementById("rule-form-model-id");
    const nameInput = document.getElementById("rule-form-name");
    const stratSelect = document.getElementById("rule-form-strategy");
    const descInput = document.getElementById("rule-form-desc");
    const enabledInput = document.getElementById("rule-form-enabled");
    const targetsContainer = document.getElementById("rule-targets-container");
    const targetsSearchInput = document.getElementById("rule-targets-search-input");
    const title = document.getElementById("modal-rule-title");

    if (targetsSearchInput) {
      targetsSearchInput.value = "";
      targetsSearchInput.oninput = (e) => {
        const q = e.target.value.toLowerCase().trim();
        targetsContainer.querySelectorAll("label").forEach((lbl) => {
          const txt = lbl.textContent.toLowerCase();
          lbl.style.display = !q || txt.includes(q) ? "flex" : "none";
        });
      };
    }

    // Gather all active models across all providers
    const allModels = [];
    state.plugins.forEach((p) => {
      (p.models || []).forEach((m) => {
        allModels.push({ id: m.id, providerName: p.name, providerId: p.id });
      });
    });

    targetsContainer.innerHTML = "";
    if (allModels.length === 0) {
      targetsContainer.innerHTML = `<span style="color: var(--ink-muted); font-size: 12px;">No provider models available. Enable provider plugins first.</span>`;
    } else {
      allModels.forEach((m) => {
        const lbl = document.createElement("label");
        lbl.style.display = "flex";
        lbl.style.alignItems = "center";
        lbl.style.gap = "8px";
        lbl.style.fontSize = "12.5px";
        lbl.style.cursor = "pointer";

        const chk = document.createElement("input");
        chk.type = "checkbox";
        chk.value = m.id;
        chk.className = "rule-target-chk";
        if (rule && (rule.targets || []).includes(m.id)) {
          chk.checked = true;
        }

        lbl.appendChild(chk);
        lbl.appendChild(document.createTextNode(`${m.id} (${m.providerName})`));
        targetsContainer.appendChild(lbl);
      });
    }

    if (rule) {
      title.textContent = "Edit Custom Load Balancing Rule";
      idInput.value = rule.id;
      mIdInput.value = rule.model_id;
      mIdInput.disabled = true;
      nameInput.value = rule.name;
      stratSelect.value = rule.strategy;
      descInput.value = rule.description || "";
      enabledInput.checked = !!rule.enabled;
    } else {
      title.textContent = "Create Custom Load Balancing Rule";
      idInput.value = "";
      mIdInput.value = "";
      mIdInput.disabled = false;
      nameInput.value = "";
      stratSelect.value = "round_robin";
      descInput.value = "";
      enabledInput.checked = true;
    }

    enhanceSelect(stratSelect);
    openModal(el.modalRule);
  }

  if (el.btnCreateRule) {
    el.btnCreateRule.addEventListener("click", () => openRuleModal(null));
  }

  const btnSaveRule = document.getElementById("btn-save-rule");
  if (btnSaveRule) {
    btnSaveRule.addEventListener("click", async () => {
      const idInput = document.getElementById("rule-form-id");
      const mIdInput = document.getElementById("rule-form-model-id");
      const nameInput = document.getElementById("rule-form-name");
      const stratSelect = document.getElementById("rule-form-strategy");
      const descInput = document.getElementById("rule-form-desc");
      const enabledInput = document.getElementById("rule-form-enabled");

      const selectedTargets = Array.from(document.querySelectorAll(".rule-target-chk:checked")).map((c) => c.value);
      const modelId = mIdInput.value.trim();
      const name = nameInput.value.trim() || modelId;

      if (!modelId) {
        showToast("Exposed model ID is required", "error");
        return;
      }
      if (selectedTargets.length === 0) {
        showToast("Please select at least one target model", "error");
        return;
      }

      try {
        if (idInput.value) {
          // Edit
          await api(`/api/load-balancing-rules/${idInput.value}`, {
            method: "PUT",
            body: JSON.stringify({
              name,
              strategy: stratSelect.value,
              targets: selectedTargets,
              description: descInput.value,
              enabled: enabledInput.checked,
            }),
          });
          showToast(`Rule '${name}' updated`);
        } else {
          // Create
          await api("/api/load-balancing-rules", {
            method: "POST",
            body: JSON.stringify({
              model_id: modelId,
              name,
              strategy: stratSelect.value,
              targets: selectedTargets,
              description: descInput.value,
              enabled: enabledInput.checked,
            }),
          });
          showToast(`Rule '${modelId}' created and exposed to /v1`);
        }

        closeModal(el.modalRule);
        fetchRules();
      } catch (err) {
        showToast(`Failed to save rule: ${err.message}`, "error");
      }
    });
  }

  // =========================================================================
  // TAB 4: PLAYGROUND
  // =========================================================================

  function populatePlaygroundModels() {
    if (!el.playgroundModelSelect) return;
    el.playgroundModelSelect.innerHTML = "";

    const allModels = [];
    state.plugins.forEach((p) => {
      if (p.enabled) {
        (p.models || []).forEach((m) => {
          if (m.enabled) allModels.push(m.id);
        });
      }
    });

    state.rules.forEach((r) => {
      if (r.enabled && !allModels.includes(r.model_id)) {
        allModels.unshift(r.model_id);
      }
    });

    if (allModels.length === 0) allModels.push("test-model");

    allModels.forEach((mid) => {
      const opt = document.createElement("option");
      opt.value = mid;
      opt.textContent = mid;
      if (mid === state.selectedPlaygroundModel) opt.selected = true;
      el.playgroundModelSelect.appendChild(opt);
    });

    enhanceSelect(el.playgroundModelSelect);

    el.playgroundModelSelect.addEventListener("change", (e) => {
      state.selectedPlaygroundModel = e.target.value;
    });
  }

  if (el.playgroundTempInput) {
    el.playgroundTempInput.addEventListener("input", (e) => {
      if (el.playgroundTempDisplay) el.playgroundTempDisplay.textContent = e.target.value;
    });
  }

  if (el.playgroundClearBtn) {
    el.playgroundClearBtn.addEventListener("click", () => {
      state.chatMessages = [];
      if (el.playgroundHistory) el.playgroundHistory.innerHTML = "";
    });
  }

  if (el.playgroundSendBtn) {
    el.playgroundSendBtn.addEventListener("click", sendPlaygroundMessage);
  }

  if (el.playgroundPrompt) {
    el.playgroundPrompt.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        sendPlaygroundMessage();
      }
    });
  }

  async function sendPlaygroundMessage() {
    const prompt = (el.playgroundPrompt ? el.playgroundPrompt.value : "").trim();
    if (!prompt) return;

    const model = el.playgroundModelSelect ? el.playgroundModelSelect.value : (state.selectedSnippetModel || "test-model");
    const temp = parseFloat(el.playgroundTempInput.value) || 1.0;
    const maxTokens = parseInt(el.playgroundTokensInput.value, 10) || undefined;
    const systemPrompt = (el.playgroundSystemInput.value || "").trim();

    el.playgroundPrompt.value = "";

    // Append user message to UI
    appendChatBubble("user", prompt);
    state.chatMessages.push({ role: "user", content: prompt });

    const messagesPayload = [];
    if (systemPrompt) messagesPayload.push({ role: "system", content: systemPrompt });
    state.chatMessages.forEach((m) => messagesPayload.push(m));

    const assistantBubble = appendChatBubble("assistant", "Thinking...");

    try {
      const res = await api("/api/test-chat", {
        method: "POST",
        body: JSON.stringify({
          model,
          messages: messagesPayload,
          temperature: temp,
          max_tokens: maxTokens,
        }),
      });

      const replyContent = (res.choices && res.choices[0] && res.choices[0].message)
        ? res.choices[0].message.content
        : (res.response || JSON.stringify(res));

      assistantBubble.textContent = replyContent;
      state.chatMessages.push({ role: "assistant", content: replyContent });
    } catch (err) {
      assistantBubble.innerHTML = `<span style="color: var(--accent-red);">Error: ${escapeHtml(err.message)}</span>`;
    }
  }

  function appendChatBubble(role, text) {
    if (!el.playgroundHistory) return null;
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-msg ${role}`;
    msgDiv.innerHTML = `
      <div style="font-size: 11px; color: var(--ink-muted); text-transform: uppercase; font-weight: 600;">${role}</div>
      <div class="chat-bubble" style="background: ${role === 'user' ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.02)'}; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px 14px; font-size: 13px; white-space: pre-wrap; line-height: 1.5;">${escapeHtml(text)}</div>
    `;
    el.playgroundHistory.appendChild(msgDiv);
    el.playgroundHistory.scrollTop = el.playgroundHistory.scrollHeight;
    return msgDiv.querySelector(".chat-bubble");
  }

  // =========================================================================
  // TAB 5: STATUS (RENAMED FROM REQUEST LOGS)
  // =========================================================================

  async function fetchStatusData(fetchFreshQuota = true) {
    await Promise.allSettled([
      fetchPlugins(),
      fetchAccounts(),
      fetchLogs(),
    ]);
    renderStatusProviderTabs();
    await renderStatusAccountsGrid(fetchFreshQuota);
  }

  async function togglePluginPin(plugin) {
    try {
      const res = await api(`/api/plugins/${plugin.id}/pin`, { method: "POST" });
      plugin.pinned = res.pinned;
      const pInState = state.plugins.find((item) => item.id === plugin.id);
      if (pInState) pInState.pinned = res.pinned;
      showToast(`Provider '${plugin.name}' ${res.pinned ? "pinned to top" : "unpinned"}`);
      renderStatusProviderTabs();
      renderStatusAccountsGrid(false);
    } catch (err) {
      showToast(`Pin failed: ${err.message}`, "error");
    }
  }

  async function toggleAccountPin(account) {
    try {
      const res = await api(`/api/accounts/${account.id}/pin`, { method: "POST" });
      account.pinned = res.pinned;
      const accInState = state.accounts.find((a) => a.id === account.id);
      if (accInState) accInState.pinned = res.pinned;
      showToast(`Account '${account.name}' ${res.pinned ? "pinned to top" : "unpinned"}`);
      renderStatusAccountsGrid(false);
    } catch (err) {
      showToast(`Pin failed: ${err.message}`, "error");
    }
  }

  function renderStatusProviderTabs() {
    if (!el.statusProviderTabs) return;
    el.statusProviderTabs.innerHTML = "";

    // Tab: All Providers
    const allBtn = document.createElement("button");
    allBtn.className = `status-provider-tab ${state.statusProviderFilter === '' ? 'active' : ''}`;
    allBtn.textContent = "All Providers";
    allBtn.addEventListener("click", () => {
      state.statusProviderFilter = "";
      renderStatusProviderTabs();
      renderStatusAccountsGrid(false);
    });
    el.statusProviderTabs.appendChild(allBtn);

    // Sort providers by name with pinned first
    const sortedPlugins = [...state.plugins].sort((a, b) => {
      if (a.pinned && !b.pinned) return -1;
      if (!a.pinned && b.pinned) return 1;
      return a.name.localeCompare(b.name);
    });

    sortedPlugins.forEach((p) => {
      const btn = document.createElement("button");
      btn.className = `status-provider-tab ${state.statusProviderFilter === p.id ? 'active' : ''}`;
      btn.innerHTML = `
        <span>${escapeHtml(p.name)}</span>
        <span class="tab-pin-toggle ${p.pinned ? 'pinned' : ''}" data-pin-id="${p.id}" title="${p.pinned ? 'Unpin provider' : 'Pin provider to top'}">
          ${renderPinSvg(p.pinned, 11)}
        </span>
      `;
      btn.addEventListener("click", (e) => {
        const pinToggle = e.target.closest(".tab-pin-toggle");
        if (pinToggle) {
          e.stopPropagation();
          togglePluginPin(p);
          return;
        }
        state.statusProviderFilter = p.id;
        renderStatusProviderTabs();
        renderStatusAccountsGrid(false);
      });
      el.statusProviderTabs.appendChild(btn);
    });
  }

  function updateCardUnhideBar(card, accId) {
    const hiddenCount = state.hiddenLimits[accId] ? state.hiddenLimits[accId].size : 0;
    let bar = card.querySelector(".unhide-limits-bar");
    if (hiddenCount > 0) {
      if (!bar) {
        bar = document.createElement("div");
        bar.className = "unhide-limits-bar";
        card.appendChild(bar);
      }
      bar.innerHTML = `<button class="btn btn-secondary btn-sm btn-unhide-all" data-acc="${accId}" style="font-size: 10.5px; padding: 2px 8px;">Unhide all (${hiddenCount})</button>`;
      bar.querySelector(".btn-unhide-all").addEventListener("click", (e) => {
        e.stopPropagation();
        unhideAllForCard(card, accId);
      });
    } else if (bar) {
      bar.remove();
    }
  }

  function unhideAllForCard(card, accId) {
    if (state.hiddenLimits[accId]) {
      delete state.hiddenLimits[accId];
      saveHiddenLimits();
    }
    card.querySelectorAll(".limit-item").forEach((item) => {
      item.classList.remove("hidden-limit");
    });
    const bar = card.querySelector(".unhide-limits-bar");
    if (bar) bar.remove();
  }

  function formatTimeRemaining(resetTime) {
    if (!resetTime) return "Rolling";
    const target = new Date(resetTime).getTime();
    if (isNaN(target)) return "Rolling";
    const now = Date.now();
    const diffMs = target - now;
    if (diffMs <= 0) return "Ready";
    const diffSec = Math.floor(diffMs / 1000);
    const days = Math.floor(diffSec / 86400);
    const hours = Math.floor((diffSec % 86400) / 3600);
    const minutes = Math.floor((diffSec % 3600) / 60);

    if (days > 0) {
      return hours > 0 ? `${days}d ${hours}h left` : `${days}d left`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes}m left`;
    }
    return `${Math.max(1, minutes)}m left`;
  }

  async function renderStatusAccountsGrid(fetchFreshQuota = false) {
    if (!el.statusAccountsGrid) return;
    
    // Only show loading placeholder if grid is completely empty
    if (el.statusAccountsGrid.children.length === 0) {
      el.statusAccountsGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--ink-muted);">
          Loading accounts status & quota limits...
        </div>
      `;
    }

    // Filter accounts by current filter
    let accountsToDisplay = state.statusProviderFilter
      ? state.accounts.filter((a) => a.provider_id === state.statusProviderFilter)
      : [...state.accounts];

    if (accountsToDisplay.length === 0) {
      el.statusAccountsGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--ink-muted);">
          No accounts configured for this provider.
        </div>
      `;
      return;
    }

    // Sort accounts by provider name (pinned providers first)
    const providerMap = {};
    state.plugins.forEach((p) => { providerMap[p.id] = p; });

    accountsToDisplay.sort((a, b) => {
      const pA = providerMap[a.provider_id];
      const pB = providerMap[b.provider_id];
      const pinA = !!a.pinned;
      const pinB = !!b.pinned;
      const provPinA = pA ? !!pA.pinned : false;
      const provPinB = pB ? !!pB.pinned : false;

      // 1. Specifically pinned account comes first
      if (pinA && !pinB) return -1;
      if (!pinA && pinB) return 1;

      // 2. Accounts from specifically pinned providers next
      if (provPinA && !provPinB) return -1;
      if (!provPinA && provPinB) return 1;

      // 3. Provider name alphabetical
      const nameA = pA ? pA.name.toLowerCase() : "";
      const nameB = pB ? pB.name.toLowerCase() : "";
      if (nameA !== nameB) return nameA.localeCompare(nameB);

      // 4. Account name alphabetical
      return a.name.localeCompare(b.name);
    });

    // Fetch live quota for Antigravity and Codex accounts if requested fresh or not cached yet
    if (fetchFreshQuota || !state.cachedQuotaData) {
      if (!state.cachedQuotaData) state.cachedQuotaData = {};

      const antigravityAccounts = accountsToDisplay.filter((a) => a.provider_id === "antigravity");
      if (antigravityAccounts.length > 0) {
        try {
          const agStatusRes = await api("/api/antigravity/status");
          if (agStatusRes && agStatusRes.accounts) {
            agStatusRes.accounts.forEach((item) => {
              if (item.quota) {
                state.cachedQuotaData[item.account_id] = item.quota;
              }
            });
          }
        } catch (_) {}
      }

      const codexAccounts = accountsToDisplay.filter((a) => a.provider_id === "codex");
      if (codexAccounts.length > 0) {
        try {
          const codexStatusRes = await api("/api/codex/status");
          if (codexStatusRes && codexStatusRes.accounts) {
            codexStatusRes.accounts.forEach((item) => {
              if (item.quota) {
                state.cachedQuotaData[item.account_id] = item.quota;
              }
            });
          }
        } catch (_) {}
      }
    }
    const quotaDataMap = state.cachedQuotaData || {};

    el.statusAccountsGrid.innerHTML = "";

    accountsToDisplay.forEach((acc) => {
      const provider = providerMap[acc.provider_id] || { id: acc.provider_id, name: acc.provider_id, pinned: false };
      const card = document.createElement("div");
      card.className = "account-square-card";

      const hiddenSet = state.hiddenLimits[acc.id] || new Set();

      const inCooldown = acc.rate_limited_until && acc.rate_limited_until > Date.now() / 1000;
      const statusPill = inCooldown
        ? `<span class="badge" style="color: var(--accent-red); font-size: 11px;">Cooldown (${Math.round(acc.rate_limited_until - Date.now() / 1000)}s)</span>`
        : acc.enabled
          ? `<span class="badge badge-active" style="font-size: 11px;">Active</span>`
          : `<span class="badge badge-muted" style="font-size: 11px;">Disabled</span>`;

      // Render limits cleanly without verbose boilerplate
      let limitsHtml = "";
      const quota = quotaDataMap[acc.id];

      if (quota && quota.groups && quota.groups.length > 0) {
        quota.groups.forEach((grp) => {
          (grp.buckets || []).forEach((b) => {
            const bucketKey = b.bucketId || b.displayName || "limit";
            const isHidden = hiddenSet.has(bucketKey);

            let pct = 0;
            let color = "#34d399";
            let label = "";

            if (b.usedPercent !== undefined && b.usedPercent !== null) {
              // Usage meter (e.g. Codex) - 0% to 100% used
              pct = Math.min(100, Math.max(0, Math.round(b.usedPercent)));
              color = pct < 60 ? "#34d399" : pct < 85 ? "#fbbf24" : "#f43f5e";
              label = `${pct}%`;
            } else {
              // Remaining quota meter (e.g. Antigravity) - 100% to 0% remaining
              const frac = b.remainingFraction !== undefined ? b.remainingFraction : 1.0;
              pct = Math.min(100, Math.max(0, Math.round(frac * 100)));
              color = pct > 40 ? "#34d399" : pct > 10 ? "#fbbf24" : "#f43f5e";
              label = `${pct}%`;
            }

            const resetStr = formatTimeRemaining(b.resetTime);
            const groupName = grp.displayName || (b.usedPercent !== undefined ? "Usage" : "Quota");

            limitsHtml += `
              <div class="limit-item ${isHidden ? 'hidden-limit' : ''}" data-key="${escapeHtml(bucketKey)}">
                <div class="limit-header">
                  <span style="font-size: 12px; font-weight: 600; color: var(--ink-primary);">${escapeHtml(b.displayName || bucketKey)}</span>
                  <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-family: var(--font-mono); font-size: 11.5px; font-weight: 600; color: ${color};">${label}</span>
                    <button class="btn-hide-limit" data-acc="${acc.id}" data-key="${escapeHtml(bucketKey)}" title="Hide this limit">✕</button>
                  </div>
                </div>
                <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 5px; overflow: hidden;">
                  <div style="background: ${color}; height: 100%; width: ${pct}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 10.5px; color: var(--ink-muted);">
                  <span>${escapeHtml(groupName)}</span>
                  <span>Resets: ${escapeHtml(resetStr)}</span>
                </div>
              </div>
            `;
          });
        });
      } else {
        // Generic account limits indicator
        const limitKey = "standard_quota";
        const isHidden = hiddenSet.has(limitKey);
        limitsHtml += `
          <div class="limit-item ${isHidden ? 'hidden-limit' : ''}" data-key="${escapeHtml(limitKey)}">
            <div class="limit-header">
              <span style="font-size: 12px; font-weight: 600; color: var(--ink-primary);">Standard Quota</span>
              <button class="btn-hide-limit" data-acc="${acc.id}" data-key="${escapeHtml(limitKey)}" title="Hide this limit">✕</button>
            </div>
            <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 5px; overflow: hidden;">
              <div style="background: #34d399; height: 100%; width: 100%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 10.5px; color: var(--ink-muted);">
              <span>Requests: ${acc.success_count || 0}</span>
              <span>Errors: ${acc.error_count || 0}</span>
            </div>
          </div>
        `;
      }

      const hasHidden = hiddenSet.size > 0;

      const isAccountPinned = !!acc.pinned;
      const isProviderPinned = !!provider.pinned;
      const isCardPinned = isAccountPinned || isProviderPinned;
      const pinTitle = isAccountPinned
        ? "Unpin account"
        : isProviderPinned
          ? "Provider is pinned (Click to pin this account specifically)"
          : "Pin account to top";

      card.className = `account-square-card ${isCardPinned ? 'is-pinned' : ''}`;

      card.innerHTML = `
        <div class="account-square-header">
          <div style="min-width: 0; flex: 1;">
            <div class="account-square-title">${escapeHtml(acc.name)}</div>
            <span style="font-size: 11px; color: var(--ink-muted);">${escapeHtml(provider.name)}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px; flex-shrink: 0;">
            <button class="btn-pin ${isCardPinned ? 'pinned' : ''} btn-toggle-card-pin" data-acc-id="${acc.id}" title="${pinTitle}">
              ${renderPinSvg(isCardPinned, 12)}
            </button>
            ${statusPill}
          </div>
        </div>
        <div style="display: flex; flex-direction: column; gap: 8px; flex: 1;">
          ${limitsHtml}
        </div>
        ${hasHidden ? `
          <div class="unhide-limits-bar">
            <button class="btn btn-secondary btn-sm btn-unhide-all" data-acc="${acc.id}" style="font-size: 10.5px; padding: 2px 8px;">Unhide all (${hiddenSet.size})</button>
          </div>
        ` : ''}
      `;

      // Event: Pin toggle on account card (pins only this account)
      const pinBtn = card.querySelector(".btn-toggle-card-pin");
      if (pinBtn) {
        pinBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          toggleAccountPin(acc);
        });
      }

      // Events to hide limit (instantaneous DOM update without network roundtrip)
      card.querySelectorAll(".btn-hide-limit").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          const accId = btn.dataset.acc;
          const k = btn.dataset.key;
          if (!state.hiddenLimits[accId]) state.hiddenLimits[accId] = new Set();
          state.hiddenLimits[accId].add(k);
          saveHiddenLimits();

          const limitItem = btn.closest(".limit-item");
          if (limitItem) limitItem.classList.add("hidden-limit");
          updateCardUnhideBar(card, accId);
        });
      });

      // Event to unhide all (instantaneous DOM update)
      const unhideBtn = card.querySelector(".btn-unhide-all");
      if (unhideBtn) {
        unhideBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          unhideAllForCard(card, acc.id);
        });
      }

      el.statusAccountsGrid.appendChild(card);
    });
  }

  // Request Audit Logs
  async function fetchLogs() {
    try {
      const data = await api("/api/logs?limit=100");
      state.logs = data || [];
      renderLogsTable();
    } catch (err) {
      console.error("Failed to load logs", err);
    }
  }

  function renderLogsTable() {
    if (!el.logsTableBody) return;
    el.logsTableBody.innerHTML = "";

    if (state.logs.length === 0) {
      el.logsTableBody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align: center; color: var(--ink-muted); padding: 24px;">
            No requests logged yet. Make requests to <code>/v1/chat/completions</code> to see real-time persistent audit records.
          </td>
        </tr>
      `;
      return;
    }

    state.logs.forEach((log) => {
      const tr = document.createElement("tr");
      const timeStr = log.timestamp ? new Date(log.timestamp * 1000).toLocaleTimeString() : "";
      const isSuccess = log.status === "success" && (log.status_code >= 200 && log.status_code < 300);

      const statusBadge = isSuccess
        ? `<span class="badge badge-active">${log.status_code || 200} OK</span>`
        : `<span class="badge" style="color: var(--accent-red); border-color: var(--accent-red-border);">${log.status_code || 500} ERR</span>`;

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-size: 11.5px; color: var(--ink-muted);">${escapeHtml(timeStr)}</td>
        <td>${statusBadge}</td>
        <td><strong style="color: var(--ink-primary);">${escapeHtml(log.model)}</strong></td>
        <td><span class="badge">${escapeHtml(log.plugin_id || "load_balancer")}</span></td>
        <td><code style="font-family: var(--font-mono); font-size: 11px; color: var(--ink-secondary);">${escapeHtml(log.api_key_id || "direct")}</code></td>
        <td style="font-family: var(--font-mono);">${log.latency_ms || 0} ms</td>
        <td style="font-family: var(--font-mono);">${log.prompt_tokens || 0} / ${log.completion_tokens || 0}</td>
        <td style="font-size: 11.5px; color: var(--ink-secondary); max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${escapeHtml(log.error_message || "Completed")}
        </td>
      `;
      el.logsTableBody.appendChild(tr);
    });
  }

  if (el.logsRefreshBtn) {
    el.logsRefreshBtn.addEventListener("click", () => fetchStatusData(true));
  }

  if (el.logsAutoToggle) {
    el.logsAutoToggle.addEventListener("change", (e) => {
      state.autoRefreshLogs = e.target.checked;
      if (state.autoRefreshLogs) {
        if (state.currentTab === "status" || state.currentTab === "logs") {
          fetchStatusData(true);
        }
        if (state.autoRefreshInterval) clearInterval(state.autoRefreshInterval);
        state.autoRefreshInterval = setInterval(() => {
          if (state.currentTab === "status" || state.currentTab === "logs") {
            fetchStatusData(true);
          }
        }, 5000);
      } else if (state.autoRefreshInterval) {
        clearInterval(state.autoRefreshInterval);
        state.autoRefreshInterval = null;
      }
    });
  }

  // =========================================================================
  // MODALS & HELPERS
  // =========================================================================

  function openNewProviderModal() {
    const fnInput = document.getElementById("new-provider-filename");
    const codeInput = document.getElementById("new-provider-code");
    const tplSelect = document.getElementById("new-provider-template-select");
    if (fnInput) fnInput.value = "";
    if (codeInput) codeInput.value = "";
    if (tplSelect) {
      tplSelect.value = "";
      enhanceSelect(tplSelect);
    }
    openModal(el.modalNewProvider);
  }

  if (el.btnNewProvider) {
    el.btnNewProvider.addEventListener("click", openNewProviderModal);
  }

  if (el.btnTemplatesGallery) {
    el.btnTemplatesGallery.addEventListener("click", () => {
      openModal(el.modalTemplates);
    });
  }

  const newProviderTemplateSelect = document.getElementById("new-provider-template-select");
  if (newProviderTemplateSelect) {
    newProviderTemplateSelect.addEventListener("change", (e) => {
      const selectedId = e.target.value;
      const t = state.templates.find((tpl) => tpl.id === selectedId);
      if (t) {
        const fnInput = document.getElementById("new-provider-filename");
        const codeInput = document.getElementById("new-provider-code");
        if (fnInput) fnInput.value = t.filename;
        if (codeInput) codeInput.value = t.code;
      }
    });
  }

  const fsEditorTemplateSelect = document.getElementById("fs-editor-template-select");
  if (fsEditorTemplateSelect) {
    fsEditorTemplateSelect.addEventListener("change", (e) => {
      const selectedId = e.target.value;
      const t = state.templates.find((tpl) => tpl.id === selectedId);
      if (t) {
        const textarea = document.getElementById("fs-editor-code");
        if (textarea) {
          textarea.value = t.code;
          state.fullscreenCodeDirty = true;
          updateFsEditor(true);
        }
      }
    });
  }

  const btnSubmitNewProvider = document.getElementById("btn-submit-new-provider");
  if (btnSubmitNewProvider) {
    btnSubmitNewProvider.addEventListener("click", async () => {
      const fnInput = document.getElementById("new-provider-filename");
      const codeInput = document.getElementById("new-provider-code");
      const fn = fnInput.value.trim();
      const code = codeInput.value.trim();

      if (!fn) {
        showToast("Filename is required", "error");
        return;
      }
      if (!code) {
        showToast("Python source code cannot be empty", "error");
        return;
      }

      try {
        await api("/api/plugins", {
          method: "POST",
          body: JSON.stringify({ filename: fn, code }),
        });
        showToast(`Plugin '${fn}' created and loaded`);
        closeModal(el.modalNewProvider);
        fetchPlugins();
      } catch (err) {
        showToast(`Failed to create plugin: ${err.message}`, "error");
      }
    });
  }

  // =========================================================================
  // Google Antigravity OAuth & Account Helpers
  // =========================================================================

  const GOOGLE_SVG = `<svg width="16" height="16" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/></svg>`;

  let currentOAuthPollInterval = null;

  function stopOAuthPolling() {
    if (currentOAuthPollInterval) {
      clearInterval(currentOAuthPollInterval);
      currentOAuthPollInterval = null;
    }
  }

  function resetGoogleOAuthUI() {
    stopOAuthPolling();
    const btnStart = document.getElementById("btn-start-google-oauth");
    const flowBox = document.getElementById("google-oauth-flow-box");
    const pasteInput = document.getElementById("google-callback-url-paste");
    const statusText = document.getElementById("google-oauth-status-text");
    if (btnStart) {
      btnStart.disabled = false;
      btnStart.innerHTML = `${GOOGLE_SVG} <span>Sign in with Google (OAuth)</span>`;
    }
    if (flowBox) flowBox.style.display = "none";
    if (pasteInput) pasteInput.value = "";
    if (statusText) statusText.textContent = "Waiting for Google authorization...";
  }

  function updateAccountModalVisibility() {
    const provSelect = document.getElementById("account-provider-select");
    const idInput = document.getElementById("account-form-id");
    const isEditing = Boolean(idInput && idInput.value);
    const selectedProvider = provSelect ? provSelect.value : "";
    const isAg = selectedProvider === "antigravity";
    const isCodex = selectedProvider === "codex";

    const agSection = document.getElementById("antigravity-oauth-section");
    const codexSection = document.getElementById("codex-oauth-section");
    const keyGroup = document.getElementById("account-key-group");

    if (agSection) {
      agSection.style.display = (isAg && !isEditing) ? "block" : "none";
    }

    if (codexSection) {
      codexSection.style.display = (isCodex && !isEditing) ? "block" : "none";
    }

    if (keyGroup) {
      // Hide API Key / Token selection entirely when adding or managing a Google Antigravity or Codex account
      if (isCodex && !isEditing) {
        keyGroup.style.display = "none";
      } else {
        keyGroup.style.display = isAg ? "none" : "block";
      }
    }
  }

  const accountProvSelect = document.getElementById("account-provider-select");
  if (accountProvSelect) {
    accountProvSelect.addEventListener("change", updateAccountModalVisibility);
  }

  const btnStartGoogleOAuth = document.getElementById("btn-start-google-oauth");
  if (btnStartGoogleOAuth) {
    btnStartGoogleOAuth.addEventListener("click", async () => {
      const flowBox = document.getElementById("google-oauth-flow-box");
      const authLink = document.getElementById("btn-open-google-auth");
      const statusText = document.getElementById("google-oauth-status-text");

      try {
        btnStartGoogleOAuth.disabled = true;
        btnStartGoogleOAuth.innerHTML = `${GOOGLE_SVG} <span>Starting OAuth Flow...</span>`;
        const flow = await api("/api/antigravity/oauth/start", { method: "POST" });
        btnStartGoogleOAuth.disabled = false;
        btnStartGoogleOAuth.innerHTML = `${GOOGLE_SVG} <span>Restart Google Sign-In</span>`;

        if (flowBox) flowBox.style.display = "block";
        if (authLink) {
          authLink.href = flow.auth_url;
        }
        if (statusText) {
          statusText.innerHTML = `
            <span class="status-dot"></span>
            Loopback listening on <code>${escapeHtml(flow.redirect_uri)}</code>. Sign in with your browser:
          `;
        }

        // Attempt to open Google OAuth sign-in tab
        try {
          window.open(flow.auth_url, "_blank");
        } catch (_) {}

        stopOAuthPolling();
        currentOAuthPollInterval = setInterval(async () => {
          try {
            const check = await api(`/api/antigravity/oauth/status/${flow.flow_id}`);
            if (check.status === "complete") {
              stopOAuthPolling();
              showToast(`Google account '${check.account.name}' connected!`);
              closeModal(el.modalAccount);
              await fetchAccounts();
              if (state.currentFullscreenPluginId) {
                renderFullscreenAccounts(state.currentFullscreenPluginId);
              }
            } else if (check.status === "error") {
              stopOAuthPolling();
              showToast(`OAuth error: ${check.error}`, "error");
              if (statusText) statusText.textContent = `Error: ${check.error}`;
            }
          } catch (_) {}
        }, 2000);
      } catch (err) {
        btnStartGoogleOAuth.disabled = false;
        btnStartGoogleOAuth.innerHTML = `${GOOGLE_SVG} <span>Sign in with Google (OAuth)</span>`;
        showToast(`Failed to start OAuth: ${err.message}`, "error");
      }
    });
  }

  const btnImportLocalGoogle = document.getElementById("btn-import-local-google");
  if (btnImportLocalGoogle) {
    btnImportLocalGoogle.addEventListener("click", async () => {
      try {
        btnImportLocalGoogle.disabled = true;
        btnImportLocalGoogle.textContent = "Importing...";
        const res = await api("/api/antigravity/oauth/import-local", { method: "POST" });
        showToast(res.message || "Google Antigravity account imported!");
        closeModal(el.modalAccount);
        await fetchAccounts();
        if (state.currentFullscreenPluginId) {
          renderFullscreenAccounts(state.currentFullscreenPluginId);
        }
      } catch (err) {
        showToast(`Import failed: ${err.message}`, "error");
      } finally {
        btnImportLocalGoogle.disabled = false;
        btnImportLocalGoogle.textContent = "⟳ Auto-Import";
      }
    });
  }

  const btnSubmitCallbackPaste = document.getElementById("btn-submit-callback-paste");
  if (btnSubmitCallbackPaste) {
    btnSubmitCallbackPaste.addEventListener("click", async () => {
      const pasteInput = document.getElementById("google-callback-url-paste");
      const val = pasteInput ? pasteInput.value.trim() : "";
      if (!val) {
        showToast("Please paste the callback URL or code first", "error");
        return;
      }
      try {
        btnSubmitCallbackPaste.disabled = true;
        btnSubmitCallbackPaste.textContent = "Connecting...";
        const res = await api("/api/antigravity/oauth/callback", {
          method: "POST",
          body: JSON.stringify({ callback_url: val }),
        });
        showToast(`Connected Google account '${res.account.name}'!`);
        stopOAuthPolling();
        closeModal(el.modalAccount);
        await fetchAccounts();
        if (state.currentFullscreenPluginId) {
          renderFullscreenAccounts(state.currentFullscreenPluginId);
        }
      } catch (err) {
        showToast(`Connection failed: ${err.message}`, "error");
      } finally {
        btnSubmitCallbackPaste.disabled = false;
        btnSubmitCallbackPaste.textContent = "Connect";
      }
    });
  }

  function resetCodexOAuthUI() {
    const flowBox = document.getElementById("codex-oauth-flow-box");
    const statusText = document.getElementById("codex-oauth-status-text");
    const pasteInput = document.getElementById("codex-callback-url-paste");
    const startBtn = document.getElementById("btn-start-codex-oauth");

    if (flowBox) flowBox.style.display = "none";
    if (statusText) statusText.textContent = "Waiting for OpenAI / ChatGPT authorization...";
    if (pasteInput) pasteInput.value = "";
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/></svg>
        <span>Sign in with ChatGPT (OAuth)</span>
      `;
    }
  }

  const btnStartCodexOAuth = document.getElementById("btn-start-codex-oauth");
  if (btnStartCodexOAuth) {
    btnStartCodexOAuth.addEventListener("click", async () => {
      const flowBox = document.getElementById("codex-oauth-flow-box");
      const authLink = document.getElementById("btn-open-codex-auth");
      const statusText = document.getElementById("codex-oauth-status-text");

      try {
        btnStartCodexOAuth.disabled = true;
        btnStartCodexOAuth.innerHTML = `<span>Starting OAuth Flow...</span>`;
        const flow = await api("/api/codex/oauth/start", { method: "POST" });
        btnStartCodexOAuth.disabled = false;
        btnStartCodexOAuth.innerHTML = `<span>Restart ChatGPT Sign-In</span>`;

        if (flowBox) flowBox.style.display = "block";
        if (authLink) authLink.href = flow.auth_url;
        if (statusText) {
          statusText.innerHTML = `
            <span class="status-dot"></span>
            Loopback listening on <code>${escapeHtml(flow.redirect_uri)}</code>. Sign in with your browser:
          `;
        }

        try {
          window.open(flow.auth_url, "_blank");
        } catch (_) {}

        stopOAuthPolling();
        currentOAuthPollInterval = setInterval(async () => {
          try {
            const check = await api(`/api/codex/oauth/status/${flow.flow_id}`);
            if (check.status === "complete") {
              stopOAuthPolling();
              showToast(`Codex account '${check.account.name}' connected!`);
              closeModal(el.modalAccount);
              await fetchAccounts();
              if (state.currentFullscreenPluginId) {
                renderFullscreenAccounts(state.currentFullscreenPluginId);
              }
            } else if (check.status === "error") {
              stopOAuthPolling();
              showToast(`OAuth error: ${check.error}`, "error");
              if (statusText) statusText.textContent = `Error: ${check.error}`;
            }
          } catch (_) {}
        }, 2000);
      } catch (err) {
        btnStartCodexOAuth.disabled = false;
        btnStartCodexOAuth.innerHTML = `<span>Sign in with ChatGPT (OAuth)</span>`;
        showToast(`Failed to start Codex OAuth: ${err.message}`, "error");
      }
    });
  }

  const btnImportLocalCodex = document.getElementById("btn-import-local-codex");
  if (btnImportLocalCodex) {
    btnImportLocalCodex.addEventListener("click", async () => {
      try {
        btnImportLocalCodex.disabled = true;
        btnImportLocalCodex.textContent = "Importing...";
        const res = await api("/api/codex/oauth/import-local", { method: "POST" });
        showToast(res.message || "Codex account imported!");
        closeModal(el.modalAccount);
        await fetchAccounts();
        if (state.currentFullscreenPluginId) {
          renderFullscreenAccounts(state.currentFullscreenPluginId);
        }
      } catch (err) {
        showToast(`Import failed: ${err.message}`, "error");
      } finally {
        btnImportLocalCodex.disabled = false;
        btnImportLocalCodex.textContent = "⟳ Auto-Import";
      }
    });
  }

  const btnCopyCodexUrl = document.getElementById("btn-copy-codex-url");
  if (btnCopyCodexUrl) {
    btnCopyCodexUrl.addEventListener("click", () => {
      const authLink = document.getElementById("btn-open-codex-auth");
      if (authLink && authLink.href) {
        navigator.clipboard.writeText(authLink.href);
        showToast("ChatGPT sign-in URL copied to clipboard!");
      }
    });
  }

  const btnSubmitCodexCallbackPaste = document.getElementById("btn-submit-codex-callback-paste");
  if (btnSubmitCodexCallbackPaste) {
    btnSubmitCodexCallbackPaste.addEventListener("click", async () => {
      const pasteInput = document.getElementById("codex-callback-url-paste");
      const val = pasteInput ? pasteInput.value.trim() : "";
      if (!val) {
        showToast("Please paste the callback URL or code first", "error");
        return;
      }
      try {
        btnSubmitCodexCallbackPaste.disabled = true;
        btnSubmitCodexCallbackPaste.textContent = "Connecting...";
        const res = await api("/api/codex/oauth/callback", {
          method: "POST",
          body: JSON.stringify({ callback_url: val }),
        });
        showToast(`Connected Codex account '${res.account.name}'!`);
        stopOAuthPolling();
        closeModal(el.modalAccount);
        await fetchAccounts();
        if (state.currentFullscreenPluginId) {
          renderFullscreenAccounts(state.currentFullscreenPluginId);
        }
      } catch (err) {
        showToast(`Connection failed: ${err.message}`, "error");
      } finally {
        btnSubmitCodexCallbackPaste.disabled = false;
        btnSubmitCodexCallbackPaste.textContent = "Connect";
      }
    });
  }

  // Account Modal
  // Returns the lowest unused positive priority for a provider's accounts.
  // `excludeId` lets an edited account keep its own current priority when re-checking.
  function getNextAvailablePriority(providerId, excludeId = null) {
    const used = new Set(
      (state.accounts || [])
        .filter((a) => a.provider_id === providerId && a.id !== excludeId)
        .map((a) => parseInt(a.priority, 10) || 1)
    );
    let next = 1;
    while (used.has(next)) next += 1;
    return next;
  }

  function openAccountModal(account = null, defaultProviderId = null) {
    const idInput = document.getElementById("account-form-id");
    const provSelect = document.getElementById("account-provider-select");
    const nameInput = document.getElementById("account-name");
    const keyInput = document.getElementById("account-api-key");
    const baseInput = document.getElementById("account-base-url");
    const priorityInput = document.getElementById("account-priority");
    const weightInput = document.getElementById("account-weight");
    const enabledInput = document.getElementById("account-enabled-toggle");
    const title = document.getElementById("modal-account-title");

    provSelect.innerHTML = "";
    state.plugins.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name;
      if (defaultProviderId && p.id === defaultProviderId) opt.selected = true;
      if (account && p.id === account.provider_id) opt.selected = true;
      provSelect.appendChild(opt);
    });
    enhanceSelect(provSelect);

    resetGoogleOAuthUI();
    resetCodexOAuthUI();

    if (account) {
      title.textContent = "Edit Provider Account";
      idInput.value = account.id;
      nameInput.value = account.name;
      keyInput.value = account.api_key || "";
      baseInput.value = account.base_url || "";
      priorityInput.value = account.priority || 1;
      weightInput.value = account.weight || 1;
      enabledInput.checked = !!account.enabled;
    } else {
      title.textContent = "Connect Provider Account";
      idInput.value = "";
      nameInput.value = "";
      keyInput.value = "";
      baseInput.value = "";
      const initialProvId = defaultProviderId || (provSelect.value || (state.plugins[0] ? state.plugins[0].id : ""));
      priorityInput.value = getNextAvailablePriority(initialProvId);
      weightInput.value = 1;
      enabledInput.checked = true;
    }

    provSelect.onchange = () => {
      if (!idInput.value) {
        priorityInput.value = getNextAvailablePriority(provSelect.value);
      }
    };

    updateAccountModalVisibility();
    openModal(el.modalAccount);
  }

  const btnSaveAccount = document.getElementById("btn-save-account");
  if (btnSaveAccount) {
    btnSaveAccount.addEventListener("click", async () => {
      const idInput = document.getElementById("account-form-id");
      const provSelect = document.getElementById("account-provider-select");
      const nameInput = document.getElementById("account-name");
      const keyInput = document.getElementById("account-api-key");
      const baseInput = document.getElementById("account-base-url");
      const priorityInput = document.getElementById("account-priority");
      const weightInput = document.getElementById("account-weight");
      const enabledInput = document.getElementById("account-enabled-toggle");

      const name = nameInput.value.trim();
      const provId = provSelect.value;
      const isEditing = Boolean(idInput.value);

      if (!isEditing && provId === "antigravity") {
        showToast("Please use 'Sign in with Google (OAuth)' or 'Auto-Import' to connect a Google Antigravity account", "warning");
        return;
      }

      if (!isEditing && provId === "codex" && !keyInput.value.trim()) {
        showToast("Please use 'Sign in with ChatGPT (OAuth)' or 'Auto-Import' to connect an OpenAI Codex account", "warning");
        return;
      }

      if (!name) {
        showToast("Account name is required", "error");
        return;
      }

      if (!isEditing && provId !== "antigravity" && provId !== "codex" && !keyInput.value.trim()) {
        showToast("API Key / Token is required", "error");
        return;
      }

      let chosenPriority = parseInt(priorityInput.value, 10) || 1;
      const conflicting = (state.accounts || []).find(
        (a) => a.provider_id === provId && a.id !== idInput.value && parseInt(a.priority, 10) === chosenPriority
      );
      if (conflicting) {
        const nextP = getNextAvailablePriority(provId, idInput.value);
        showToast(`Priority ${chosenPriority} is already in use by '${conflicting.name}'. Automatically changed to ${nextP}`, "warning");
        chosenPriority = nextP;
        priorityInput.value = nextP;
      }

      try {
        if (isEditing) {
          // Edit
          const updatePayload = {
            name,
            base_url: baseInput.value || undefined,
            priority: chosenPriority,
            weight: parseInt(weightInput.value, 10) || 1,
            enabled: enabledInput.checked,
          };
          if (provId !== "antigravity" || keyInput.value) {
            updatePayload.api_key = keyInput.value;
          }

          await api(`/api/accounts/${idInput.value}`, {
            method: "PUT",
            body: JSON.stringify(updatePayload),
          });
          showToast(`Account '${name}' updated`);
        } else {
          // Create
          await api("/api/accounts", {
            method: "POST",
            body: JSON.stringify({
              provider_id: provId,
              name,
              api_key: keyInput.value,
              base_url: baseInput.value || undefined,
              priority: chosenPriority,
              weight: parseInt(weightInput.value, 10) || 1,
              enabled: enabledInput.checked,
            }),
          });
          showToast(`Account '${name}' connected`);
        }

        closeModal(el.modalAccount);
        await fetchAccounts();
        if (state.currentFullscreenPluginId) {
          renderFullscreenAccounts(state.currentFullscreenPluginId);
        }
      } catch (err) {
        showToast(`Failed to save account: ${err.message}`, "error");
      }
    });
  }

  // Toggle account key visibility
  const btnToggleKey = document.getElementById("btn-toggle-account-key");
  if (btnToggleKey) {
    btnToggleKey.addEventListener("click", () => {
      const keyInput = document.getElementById("account-api-key");
      if (keyInput) {
        const isPass = keyInput.type === "password";
        keyInput.type = isPass ? "text" : "password";
        btnToggleKey.textContent = isPass ? "Hide" : "Show";
      }
    });
  }

  // Close modals on click of .modal-close or clicking backdrop with animations & exit confirmation
  document.querySelectorAll(".modal-close").forEach((btn) => {
    btn.addEventListener("click", () => {
      const modal = btn.closest(".modal-overlay");
      if (modal === el.modalProviderFullscreen) {
        requestCloseFullscreen();
      } else if (modal) {
        closeModal(modal);
      }
    });
  });

  document.querySelectorAll(".modal-overlay").forEach((m) => {
    m.addEventListener("click", (e) => {
      if (e.target === m) {
        if (m === el.modalProviderFullscreen) {
          requestCloseFullscreen();
        } else {
          closeModal(m);
        }
      }
    });
  });

  // ESC key handler for closing modals with confirmation when editing
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const confirmModal = document.getElementById("modal-confirm");
      if (confirmModal && confirmModal.classList.contains("active")) {
        const cancelBtn = document.getElementById("confirm-dialog-cancel");
        if (cancelBtn) cancelBtn.click();
        return;
      }
      if (el.modalProviderFullscreen && el.modalProviderFullscreen.classList.contains("active")) {
        requestCloseFullscreen();
        return;
      }
      const activeModals = Array.from(document.querySelectorAll(".modal-overlay.active"));
      if (activeModals.length > 0) {
        closeModal(activeModals[activeModals.length - 1]);
      }
    }
  });

  // Global initial boot
  async function init() {
    enhanceAllSelects();
    await fetchStats();
    await loadTemplates();
    await fetchPlugins();
    await fetchAccounts();
    await fetchApiKeys();
    await fetchTemporaryApiKey();
    await fetchRules();
    populateSnippetModels();
    updateSnippet();
  }

  window.addEventListener("DOMContentLoaded", init);
})();
