/**
 * LLMProxy Management Dashboard Application
 * Modern Vanilla JavaScript (ES6+) • Zero External Dependencies • Offline-First
 */

(function () {
  "use strict";

  const ADMIN_TOKEN_STORAGE_KEY = "llmproxy_admin_token";
  let adminToken = "";

  try {
    adminToken = localStorage.getItem(ADMIN_TOKEN_STORAGE_KEY) || "";
  } catch (_) {
    adminToken = "";
  }

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
    managementKey: "",
    statusView: "overview", // "overview" | "graph"
    statusProviderFilter: "",
    expandedStatusGroups: new Set(),
    graphProviders: new Set(), // empty => All Providers
    graphMetric: "tokens", // "tokens" | "requests" | "cache" | "latency"
    graphRange: "all", // "1h" | "6h" | "24h" | "7d" | "all"
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
    attachedImages: [],
    autoRefreshLogs: false,
    autoRefreshInterval: null,
    cachedQuotaData: null,
    usageProviderIds: null,
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
    statCacheHitRate: document.getElementById("stat-cache-hit-rate"),
    // Overview
    apiKeysCountBadge: document.getElementById("api-keys-count-badge"),
    apiKeysTableBody: document.getElementById("api-keys-table-body"),
    btnCreateApiKey: document.getElementById("btn-create-api-key"),
    managementHubUrl: document.getElementById("management-hub-url"),
    managementHubKey: document.getElementById("management-hub-key"),
    copyManagementKeyBtn: document.getElementById("copy-management-key-btn"),
    snippetCode: document.getElementById("snippet-code"),
    snippetModelSelect: document.getElementById("snippet-model-select"),
    headerEndpointDisplay: document.getElementById("header-endpoint-display"),
    copySnippetBtn: document.getElementById("copy-snippet-btn"),
    copyEndpointBtn: document.getElementById("copy-endpoint-btn"),
    btnAdminLock: document.getElementById("btn-admin-lock"),
    authGate: document.getElementById("auth-gate"),
    authTokenInput: document.getElementById("auth-token-input"),
    authSubmitBtn: document.getElementById("auth-submit-btn"),
    authError: document.getElementById("auth-error"),
    // Providers
    statProvidersCount: document.getElementById("stat-providers-count"),
    statTotalAccounts: document.getElementById("stat-total-accounts"),
    statActiveAccounts: document.getElementById("stat-active-accounts"),
    statCooldownAccounts: document.getElementById("stat-cooldown-accounts"),
    statProviderModels: document.getElementById("stat-provider-models"),
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
    playgroundAttachBtn: document.getElementById("playground-attach-btn"),
    playgroundImageInput: document.getElementById("playground-image-input"),
    playgroundAttachments: document.getElementById("playground-attachments"),
    playgroundSidebar: document.getElementById("playground-sidebar"),
    playgroundParamsToggle: document.getElementById("playground-params-toggle"),
    // Status
    statusViewBtns: document.querySelectorAll(".status-view-btn"),
    statusViewOverview: document.getElementById("status-view-overview"),
    statusViewGraph: document.getElementById("status-view-graph"),
    statusPoolSummaryBadge: document.getElementById("status-pool-summary-badge"),
    statusPoolSubtitle: document.getElementById("status-pool-subtitle"),
    statusProviderTabs: document.getElementById("status-provider-tabs"),
    statusAccountsGrid: document.getElementById("status-accounts-grid"),
    logsFilterBadge: document.getElementById("logs-filter-badge"),
    logsTableBody: document.getElementById("logs-table-body"),
    logsRefreshBtn: document.getElementById("logs-refresh-btn"),
    logsAutoToggle: document.getElementById("logs-auto-toggle"),
    // Status Graph View
    graphProviderFilters: document.getElementById("graph-provider-filters"),
    graphMetricTabs: document.getElementById("graph-metric-tabs"),
    graphRangeTabs: document.getElementById("graph-range-tabs"),
    graphKpiRequests: document.getElementById("graph-kpi-requests"),
    graphKpiSuccess: document.getElementById("graph-kpi-success"),
    graphKpiTokens: document.getElementById("graph-kpi-tokens"),
    graphKpiTokensSub: document.getElementById("graph-kpi-tokens-sub"),
    graphKpiCache: document.getElementById("graph-kpi-cache"),
    graphKpiCacheSub: document.getElementById("graph-kpi-cache-sub"),
    graphKpiLatency: document.getElementById("graph-kpi-latency"),
    graphKpiErrors: document.getElementById("graph-kpi-errors"),
    graphChartTitle: document.getElementById("graph-chart-title"),
    graphChartSubtitle: document.getElementById("graph-chart-subtitle"),
    graphTimeseriesLegend: document.getElementById("graph-timeseries-legend"),
    graphTimeseriesContainer: document.getElementById("graph-timeseries-container"),
    graphPoolCountBadge: document.getElementById("graph-pool-count-badge"),
    graphPoolCapacityContainer: document.getElementById("graph-pool-capacity-container"),
    graphBreakdownTitle: document.getElementById("graph-breakdown-title"),
    graphBreakdownBadge: document.getElementById("graph-breakdown-badge"),
    graphProviderBreakdownContainer: document.getElementById("graph-provider-breakdown-container"),
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
    if (modalEl === el.modalAccount && typeof resetPluginOAuthUI === "function") {
      resetPluginOAuthUI();
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

  // =========================================================================
  // Admin Authentication
  // =========================================================================

  function setAdminToken(token) {
    adminToken = token || "";
    try {
      if (adminToken) {
        localStorage.setItem(ADMIN_TOKEN_STORAGE_KEY, adminToken);
      } else {
        localStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
      }
    } catch (_) {}
  }

  function showAuthGate(message = "") {
    if (el.authGate) el.authGate.hidden = false;
    if (el.authError) el.authError.textContent = message;
    if (el.authTokenInput) {
      el.authTokenInput.value = adminToken;
      el.authTokenInput.focus();
    }
  }

  function hideAuthGate() {
    if (el.authGate) el.authGate.hidden = true;
    if (el.authError) el.authError.textContent = "";
  }

  async function verifyAdminToken(token) {
    if (!token) return false;
    try {
      const res = await fetch("/api/auth/status", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return false;
      const data = await res.json();
      return Boolean(data && data.authenticated);
    } catch (_) {
      return false;
    }
  }

  // API Client
  async function api(path, options = {}) {
    try {
      const headers = {
        "Content-Type": "application/json",
        ...options.headers,
      };
      if (adminToken) {
        headers.Authorization = `Bearer ${adminToken}`;
      }

      const res = await fetch(path, { ...options, headers });

      if (res.status === 401) {
        showAuthGate("Your admin token is missing or no longer valid.");
        throw new Error("Admin authentication required");
      }

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
  // RESPONSIVE LAYOUT METRICS
  // =========================================================================

  let playgroundParamsTouched = false;

  function setPlaygroundParamsCollapsed(collapsed) {
    if (!el.playgroundSidebar || !el.playgroundParamsToggle) return;
    el.playgroundSidebar.classList.toggle("is-collapsed", collapsed);
    el.playgroundParamsToggle.setAttribute("aria-expanded", String(!collapsed));
    el.playgroundParamsToggle.textContent = collapsed ? "Parameters" : "Hide Parameters";
  }

  function syncResponsiveLayout() {
    const header = document.querySelector(".app-header");
    if (header) {
      document.documentElement.style.setProperty("--app-header-h", `${header.offsetHeight}px`);
    }
    if (!playgroundParamsTouched) {
      // Match the CSS breakpoint that stacks the playground into one column.
      setPlaygroundParamsCollapsed(window.matchMedia("(max-width: 1024px)").matches);
    }
  }

  if (el.playgroundParamsToggle) {
    el.playgroundParamsToggle.addEventListener("click", () => {
      if (!el.playgroundSidebar) return;
      playgroundParamsTouched = true;
      const collapsed = !el.playgroundSidebar.classList.contains("is-collapsed");
      setPlaygroundParamsCollapsed(collapsed);
    });
  }

  window.addEventListener("resize", syncResponsiveLayout, { passive: true });
  if (typeof ResizeObserver !== "undefined") {
    const headerEl = document.querySelector(".app-header");
    if (headerEl) new ResizeObserver(syncResponsiveLayout).observe(headerEl);
  }

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
      if (el.statCacheHitRate) el.statCacheHitRate.textContent = `${data.cache_hit_rate || 0}%`;
      if (el.statProvidersCount) el.statProvidersCount.textContent = data.active_plugins_count || 0;
      if (el.statTotalAccounts) el.statTotalAccounts.textContent = data.total_accounts || 0;
      if (el.statActiveAccounts) el.statActiveAccounts.textContent = data.active_accounts || 0;
    } catch (_) {}
  }

  async function fetchManagementInfo() {
    try {
      const data = await api("/api/management");
      if (el.managementHubUrl) el.managementHubUrl.textContent = data.base_url || "";
      if (el.managementHubKey) el.managementHubKey.textContent = data.management_key || "";
      state.managementKey = data.management_key || "";
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
      // The API returns a masked secret; copy actions fetch the literal value
      // through the admin-only reveal endpoint.
      const maskedSecret = k.key || "";
      const lastUsed = k.last_used_at
        ? new Date(k.last_used_at * 1000).toLocaleString()
        : "Never";

      tr.innerHTML = `
        <td data-label="Key Name"><strong>${escapeHtml(k.name)}</strong></td>
        <td data-label="Secret">
          <div class="cell-inline">
            <code class="cell-mono">${escapeHtml(maskedSecret)}</code>
            <button class="btn btn-secondary btn-sm btn-copy-key" data-id="${k.id}" style="padding: 2px 6px; font-size: 11px;">Copy</button>
          </div>
        </td>
        <td data-label="Status">
          <span class="badge ${k.enabled ? 'badge-active' : 'badge-muted'}">
            ${k.enabled ? "Active" : "Disabled"}
          </span>
        </td>
        <td data-label="Requests" class="cell-num">${k.total_requests || 0}</td>
        <td data-label="Prompt Tokens" class="cell-num">${k.prompt_tokens || 0}</td>
        <td data-label="Completion Tokens" class="cell-num">${k.completion_tokens || 0}</td>
        <td data-label="Total Tokens" class="cell-num cell-strong">${k.total_tokens || 0}</td>
        <td data-label="Last Used" class="cell-muted">${escapeHtml(lastUsed)}</td>
        <td data-label="Actions">
          <div class="cell-inline">
            <button class="btn btn-secondary btn-sm btn-edit-key" data-id="${k.id}">Edit</button>
            <button class="btn btn-danger btn-sm btn-delete-key" data-id="${k.id}">Delete</button>
          </div>
        </td>
      `;

      tr.querySelector(".btn-copy-key").addEventListener("click", async () => {
        try {
          const revealed = await api(`/api/keys/${k.id}/reveal`);
          const ok = await copyToClipboard(revealed.key || "");
          if (ok) {
            showToast("API key copied to clipboard");
          } else {
            showToast("Failed to copy API key", "error");
          }
        } catch (err) {
          showToast(`Failed to reveal API key: ${err.message}`, "error");
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
    const apiKey = state.temporaryApiKey || "sk-proxy-demo";

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

  if (el.copyManagementKeyBtn) {
    el.copyManagementKeyBtn.addEventListener("click", async () => {
      const key = (el.managementHubKey && el.managementHubKey.textContent) || state.managementKey || "";
      const ok = await copyToClipboard(key);
      if (ok) {
        showToast("Management key copied");
      } else {
        showToast("Failed to copy management key", "error");
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
      if (el.statProviderModels) {
        el.statProviderModels.textContent = state.plugins.reduce(
          (total, p) => total + (p.models || []).filter((m) => m.enabled).length,
          0
        );
      }
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
            <div class="provider-toggle-wrap">
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

    const editIdentityBtn = document.getElementById("btn-edit-provider-identity");
    if (editIdentityBtn) {
      editIdentityBtn.onclick = () => {
        const current = state.plugins.find(
          (p) => p.id === state.currentFullscreenPluginId
        );
        if (!current) return;
        if (isFullscreenDirty()) {
          showToast(
            "Save or discard your unsaved changes before editing the provider identity",
            "error"
          );
          return;
        }
        openProviderIdentityModal(current);
      };
    }

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

    const plugin = (state.plugins || []).find((item) => item.id === pluginId);
    const oauthSupported = Boolean(plugin && plugin.oauth_login);
    const toolbar = document.getElementById("fs-accounts-toolbar");
    if (toolbar) {
      const label = oauthSupported ? "Sign in / Connect Account" : "+ Connect Account";
      toolbar.innerHTML = `<button class="btn btn-primary btn-sm" id="fs-btn-add-account">${label}</button>`;
      const btn = toolbar.querySelector("#fs-btn-add-account");
      if (btn) btn.addEventListener("click", () => openAccountModal(null, pluginId));
    }

    const accounts = (state.accounts || []).filter((a) => a.provider_id === pluginId);
    if (accounts.length === 0) {
      const emptyMsg = oauthSupported
        ? 'No accounts connected yet. Click "Sign in / Connect Account" above to connect one.'
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
      // The API returns a masked or empty credential; keep the display honest
      // without re-masking an already masked value.
      const maskedKey = acc.api_key || "OAuth / None";

      const inCooldown = acc.rate_limited_until && acc.rate_limited_until > Date.now() / 1000;
      const statusBadge = inCooldown
        ? `<span class="badge" style="color: var(--accent-red); border-color: var(--accent-red-border);">Cooldown (${Math.round(acc.rate_limited_until - Date.now() / 1000)}s)</span>`
        : acc.enabled
          ? `<span class="badge badge-active">Active</span>`
          : `<span class="badge badge-muted">Disabled</span>`;

      tr.innerHTML = `
        <td data-label="Account Name"><strong>${escapeHtml(acc.name)}</strong></td>
        <td data-label="API Key / Token"><code class="cell-mono">${escapeHtml(maskedKey)}</code></td>
        <td data-label="Priority" class="cell-num">${acc.priority || 1}</td>
        <td data-label="Weight" class="cell-num">${acc.weight || 1}</td>
        <td data-label="Status">${statusBadge}</td>
        <td data-label="Success / Errors" class="cell-num">${acc.success_count || 0} / ${acc.error_count || 0}</td>
        <td data-label="Last Used" class="cell-muted">${acc.last_used_at ? new Date(acc.last_used_at * 1000).toLocaleTimeString() : 'Never'}</td>
        <td data-label="Actions">
          <div class="cell-inline">
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
      const visionBadge = m.supports_vision
        ? ' <span style="font-size: 9px; letter-spacing: 0.04em; text-transform: uppercase; border: 1px solid var(--border-solid); border-radius: 4px; padding: 1px 5px; color: var(--ink-secondary);">Vision</span>'
        : "";
      tr.innerHTML = `
        <td data-label="Model ID"><div class="cell-inline wrap"><strong>${escapeHtml(m.id)}</strong>${visionBadge}</div></td>
        <td data-label="Display Name">${escapeHtml(m.name || m.id)}</td>
        <td data-label="Upstream ID"><code class="cell-mono">${escapeHtml(m.upstream_model || m.id)}</code></td>
        <td data-label="Context / Max" class="cell-num">${m.context_window || 8192} / ${m.max_tokens || 4096}</td>
        <td data-label="Enabled">
          <label class="switch">
            <input type="checkbox" class="fs-model-toggle" data-mid="${m.id}" ${m.enabled ? "checked" : ""}>
            <span class="slider"></span>
          </label>
        </td>
        <td data-label="Actions">
          <div class="cell-inline">
            <button class="btn btn-secondary btn-sm btn-test btn-fs-test-model" data-mid="${m.id}" title="Test model response">Test</button>
            <button class="btn btn-secondary btn-sm btn-fs-edit-model" data-mid="${m.id}">Edit</button>
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
          await api(`/api/plugins/${plugin.id}/models/${encodeURIComponent(m.id)}`, {
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

      tr.querySelector(".btn-fs-edit-model").addEventListener("click", () => {
        openModelForm(plugin.id, m);
      });

      tr.querySelector(".btn-fs-del-model").addEventListener("click", async () => {
        const ok = await customConfirm(`Remove model '${m.id}' from plugin?`, {
          title: "Remove Model",
          confirmText: "Remove Model",
          danger: true,
        });
        if (ok) {
          try {
            await api(`/api/plugins/${plugin.id}/models/${encodeURIComponent(m.id)}`, { method: "DELETE" });
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

  function openModelForm(pluginId, model) {
    const isEdit = Boolean(model);
    document.getElementById("model-form-mode").value = isEdit ? "edit" : "add";
    document.getElementById("model-form-plugin-id").value = pluginId;
    const idInput = document.getElementById("model-form-id");
    idInput.value = isEdit ? model.id : "";
    idInput.dataset.originalId = isEdit ? model.id : "";
    document.getElementById("model-form-name").value = isEdit ? (model.name || "") : "";
    document.getElementById("model-form-upstream").value = isEdit ? (model.upstream_model || "") : "";
    document.getElementById("model-form-context").value = isEdit ? (model.context_window || 8192) : 128000;
    document.getElementById("model-form-max-tokens").value = isEdit ? (model.max_tokens || 4096) : 4096;
    document.getElementById("model-form-parameters").value =
      isEdit && model.parameters && Object.keys(model.parameters).length > 0
        ? JSON.stringify(model.parameters, null, 2)
        : "";
    document.getElementById("model-form-reasoning-levels").value =
      isEdit && Array.isArray(model.reasoning_levels)
        ? model.reasoning_levels.join(", ")
        : "";
    document.getElementById("model-form-vision").checked = isEdit ? Boolean(model.supports_vision) : false;
    document.getElementById("model-form-enabled").checked = isEdit ? model.enabled !== false : true;
    document.getElementById("modal-model-title").textContent = isEdit ? "Edit Model" : "Add Custom Model";
    document.getElementById("btn-save-model").textContent = isEdit ? "Save Changes" : "Save Model";
    openModal(el.modalModel);
  }

  const fsBtnAddModel = document.getElementById("fs-btn-add-model");
  if (fsBtnAddModel) {
    fsBtnAddModel.addEventListener("click", () => {
      openModelForm(state.currentFullscreenPluginId, null);
    });
  }

  // Model creation / edit submit
  const btnSaveModel = document.getElementById("btn-save-model");
  if (btnSaveModel) {
    btnSaveModel.addEventListener("click", async () => {
      const mode = document.getElementById("model-form-mode").value;
      const pId = document.getElementById("model-form-plugin-id").value;
      const originalId =
        document.getElementById("model-form-id").dataset.originalId || "";
      const mId = document.getElementById("model-form-id").value.trim();
      const mName = document.getElementById("model-form-name").value.trim() || mId;
      const mUpstream = document.getElementById("model-form-upstream").value.trim() || mId;
      const mContext = parseInt(document.getElementById("model-form-context").value, 10) || 8192;
      const mMaxTokens = parseInt(document.getElementById("model-form-max-tokens").value, 10) || 4096;
      const mVision = document.getElementById("model-form-vision").checked;
      const mEnabled = document.getElementById("model-form-enabled").checked;
      const paramsText = document.getElementById("model-form-parameters").value.trim();
      const mReasoningLevels = document
        .getElementById("model-form-reasoning-levels")
        .value.split(",")
        .map((level) => level.trim().toLowerCase())
        .filter(
          (level, index, all) =>
            ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"].includes(level) &&
            all.indexOf(level) === index
        );

      if (!mId) {
        showToast("Model ID is required", "error");
        return;
      }

      let mParameters = {};
      if (paramsText) {
        try {
          mParameters = JSON.parse(paramsText);
        } catch (err) {
          showToast("Parameters must be valid JSON", "error");
          return;
        }
        if (!mParameters || typeof mParameters !== "object" || Array.isArray(mParameters)) {
          showToast("Parameters must be a JSON object", "error");
          return;
        }
      }

      const payload = {
        name: mName,
        upstream_model: mUpstream,
        context_window: mContext,
        max_tokens: mMaxTokens,
        supports_vision: mVision,
        reasoning_levels: mReasoningLevels,
        enabled: mEnabled,
        parameters: mParameters,
      };

      try {
        if (mode === "edit") {
          await api(
            `/api/plugins/${encodeURIComponent(pId)}/models/${encodeURIComponent(originalId || mId)}`,
            {
              method: "PUT",
              body: JSON.stringify({ id: mId, ...payload }),
            }
          );
          showToast(`Model '${mId}' updated`);
        } else {
          await api(`/api/plugins/${encodeURIComponent(pId)}/models`, {
            method: "POST",
            body: JSON.stringify({ id: mId, ...payload }),
          });
          showToast(`Model '${mId}' added`);
        }
        closeModal(el.modalModel);
        await fetchPlugins();
        const updatedP = state.plugins.find((p) => p.id === pId);
        if (updatedP) renderFullscreenModels(updatedP);
      } catch (err) {
        showToast(`Failed to save model: ${err.message}`, "error");
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

    const activeTargetIds = new Set(buildRuleTargetCatalog().map((model) => model.id));

    let rulesToDisplay = state.rules;
    if (state.rulesSearchQuery) {
      const q = state.rulesSearchQuery.toLowerCase().trim();
      rulesToDisplay = state.rules.filter((r) => {
        return (
          (r.model_id && r.model_id.toLowerCase().includes(q)) ||
          (r.name && r.name.toLowerCase().includes(q)) ||
          (r.strategy && r.strategy.toLowerCase().includes(q)) ||
          (r.targets && r.targets.some((t) => t.toLowerCase().includes(q))) ||
          (r.exposed_reasoning_levels && r.exposed_reasoning_levels.some((l) => String(l).toLowerCase().includes(q))) ||
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
          <td colspan="7" style="text-align: center; color: var(--ink-muted); padding: 32px;">
            ${msg}
          </td>
        </tr>
      `;
      return;
    }

    rulesToDisplay.forEach((r) => {
      const tr = document.createElement("tr");
      const targetsBadges = (r.targets || [])
        .map((target) => {
          const parsed = splitReasoningSuffix(target);
          const baseId = parsed ? parsed.baseId : target;
          if (!activeTargetIds.has(baseId)) return "";
          const label = parsed ? `${baseId} · ${parsed.level}` : target;
          return `<span class="badge">${escapeHtml(label)}</span>`;
        })
        .filter(Boolean)
        .join(" ");
      const targetCell = targetsBadges ||
        `<span class="badge badge-muted">No active targets</span>`;

      const manualReasoning = r.manual_reasoning !== false;
      const exposedLevels = Array.isArray(r.exposed_reasoning_levels)
        ? r.exposed_reasoning_levels
        : [];
      const reasoningCell = manualReasoning
        ? `<span class="badge badge-muted">Manual</span>`
        : `<div style="display: flex; flex-wrap: wrap; gap: 4px;">
             <span class="badge badge-muted">Client</span>
             ${exposedLevels.map((level) => `<span class="badge">${escapeHtml(level)}</span>`).join(" ")}
           </div>`;

      tr.innerHTML = `
        <td data-label="Model ID"><strong class="cell-strong">${escapeHtml(r.model_id)}</strong></td>
        <td data-label="Name">${escapeHtml(r.name || r.model_id)}</td>
        <td data-label="Strategy"><code class="cell-mono cell-upper">${escapeHtml(r.strategy)}</code></td>
        <td data-label="Targets"><div class="cell-inline wrap">${targetCell}</div></td>
        <td data-label="Reasoning">${reasoningCell}</td>
        <td data-label="Status">
          <label class="switch">
            <input type="checkbox" class="rule-enable-toggle" data-id="${r.id}" ${r.enabled ? "checked" : ""}>
            <span class="slider"></span>
          </label>
        </td>
        <td data-label="Actions">
          <div class="cell-inline">
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

  const RULE_REASONING_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"];
  const RULE_DEFAULT_REASONING = "__default__";

  /** Split a `<model>:<level>` ID into its base model and reasoning level. */
  function splitReasoningSuffix(modelId) {
    if (!modelId || typeof modelId !== "string") return null;
    const separator = modelId.lastIndexOf(":");
    if (separator <= 0) return null;
    const level = modelId.slice(separator + 1).toLowerCase();
    if (!RULE_REASONING_LEVELS.includes(level)) return null;
    return { baseId: modelId.slice(0, separator), level };
  }

  /**
   * Build a de-duplicated list of selectable rule targets. Explicit reasoning
   * model IDs (`model:high`) are folded into the base model instead of being
   * shown as separate near-duplicate entries, and every provider-level
   * `reasoning_effort` option is attached to that provider's models.
   */
  function buildRuleTargetCatalog() {
    const catalog = [];
    const byId = new Map();

    function ensureEntry(id, provider) {
      let entry = byId.get(id);
      if (!entry) {
        entry = {
          id,
          name: id,
          providerName: provider ? provider.name : "",
          providerId: provider ? provider.id : "",
          levels: [],
        };
        byId.set(id, entry);
        catalog.push(entry);
      }
      return entry;
    }

    function addLevel(entry, level) {
      const normalized = String(level || "").toLowerCase();
      if (!normalized || normalized === "default" || normalized === "auto") return;
      if (!RULE_REASONING_LEVELS.includes(normalized)) return;
      if (!entry.levels.includes(normalized)) entry.levels.push(normalized);
    }

    state.plugins.forEach((p) => {
      if (p.enabled === false) return;
      const reasoningVar = (p.variables || []).find((v) => v.name === "reasoning_effort");
      const providerLevels = reasoningVar && Array.isArray(reasoningVar.options)
        ? reasoningVar.options
        : [];

      (p.models || []).forEach((m) => {
        if (m.enabled === false) return;
        const parsed = splitReasoningSuffix(m.id);
        const entry = ensureEntry(parsed ? parsed.baseId : m.id, p);
        if (!parsed) {
          entry.name = m.name || m.id;
        } else {
          addLevel(entry, parsed.level);
        }
        // Model-level capability metadata is authoritative. Fall back to the
        // provider's reasoning_effort options only when the model does not
        // advertise its own levels.
        const modelLevels = Array.isArray(m.reasoning_levels) ? m.reasoning_levels : [];
        if (modelLevels.length) {
          modelLevels.forEach((level) => addLevel(entry, level));
        } else {
          providerLevels.forEach((level) => addLevel(entry, level));
        }
      });
    });

    return catalog;
  }

  function openRuleModal(rule = null) {
    const idInput = document.getElementById("rule-form-id");
    const mIdInput = document.getElementById("rule-form-model-id");
    const nameInput = document.getElementById("rule-form-name");
    const stratSelect = document.getElementById("rule-form-strategy");
    const contextInput = document.getElementById("rule-form-context");
    const maxTokensInput = document.getElementById("rule-form-max-tokens");
    const descInput = document.getElementById("rule-form-desc");
    const enabledInput = document.getElementById("rule-form-enabled");
    const targetsContainer = document.getElementById("rule-targets-container");
    const targetsSearchInput = document.getElementById("rule-targets-search-input");
    const selectedContainer = document.getElementById("rule-selected-targets");
    const priorityHint = document.getElementById("rule-priority-hint");
    const title = document.getElementById("modal-rule-title");
    const manualReasoningInput = document.getElementById("rule-form-manual-reasoning");
    const exposedLevelsContainer = document.getElementById("rule-exposed-reasoning-levels");
    const exposedReasoningGroup = document.getElementById("rule-exposed-reasoning-group");
    const exposedReasoningHint = document.getElementById("rule-exposed-reasoning-hint");

    let manualReasoning = rule ? rule.manual_reasoning !== false : true;
    let exposedLevels = Array.isArray(rule && rule.exposed_reasoning_levels)
      ? rule.exposed_reasoning_levels
          .map((level) => String(level || "").toLowerCase())
          .filter((level) => RULE_REASONING_LEVELS.includes(level))
      : [];
    exposedLevels = exposedLevels.filter((level, index, all) => all.indexOf(level) === index);
    const reasoningOverrides = {};
    Object.entries((rule && rule.reasoning_overrides) || {}).forEach(([targetId, mapping]) => {
      if (!mapping || typeof mapping !== "object") return;
      const parsed = splitReasoningSuffix(targetId);
      const baseId = parsed ? parsed.baseId : targetId;
      const clean = {};
      Object.entries(mapping).forEach(([source, destination]) => {
        const from = String(source || "").toLowerCase();
        const to = String(destination || "").toLowerCase();
        if (
          RULE_REASONING_LEVELS.includes(from) &&
          (RULE_REASONING_LEVELS.includes(to) || to === RULE_DEFAULT_REASONING)
        ) {
          clean[from] = to;
        }
      });
      if (Object.keys(clean).length) {
        reasoningOverrides[baseId] = { ...(reasoningOverrides[baseId] || {}), ...clean };
      }
    });

    const catalog = buildRuleTargetCatalog();
    const catalogById = new Map(catalog.map((m) => [m.id, m]));
    const selectedTargets = [];

    (rule && rule.targets ? rule.targets : []).forEach((target) => {
      const parsed = splitReasoningSuffix(target);
      const targetId = parsed ? parsed.baseId : target;
      // Ignore targets from removed providers or disabled models. They remain
      // in the stored rule until it is saved again, but the editor should not
      // present them as selectable.
      if (!catalogById.has(targetId)) return;
      selectedTargets.push(
        parsed ? { id: targetId, level: parsed.level } : { id: target, level: "" }
      );
    });

    // Collapse exact duplicate targets while preserving the saved priority order.
    const seenTargets = new Set();
    for (let i = 0; i < selectedTargets.length; i++) {
      const key = `${selectedTargets[i].id}:${selectedTargets[i].level}`;
      if (seenTargets.has(key)) {
        selectedTargets.splice(i, 1);
        i--;
      } else {
        seenTargets.add(key);
      }
    }

    el.modalRule._ruleTargets = selectedTargets;

    function hasSelectedBase(baseId) {
      return selectedTargets.some((t) => t.id === baseId);
    }

    function levelsForTarget(target) {
      const entry = catalogById.get(target.id);
      const levels = entry ? entry.levels.slice() : [];
      if (target.level && !levels.includes(target.level)) levels.push(target.level);
      return levels;
    }

    function nativeLevelsForTarget(targetId) {
      const entry = catalogById.get(targetId);
      return entry ? entry.levels.slice() : [];
    }

    function effectiveTargetLevels(targetId) {
      const native = nativeLevelsForTarget(targetId);
      const valid = new Set(native);
      Object.entries(reasoningOverrides[targetId] || {}).forEach(([source, destination]) => {
        if (destination === RULE_DEFAULT_REASONING || valid.has(destination)) {
          valid.add(source);
        }
      });
      return valid;
    }

    function validExposedLevels() {
      const targetIds = selectedTargets.map((target) => target.id);
      if (!targetIds.length) return [];
      let intersection = null;
      targetIds.forEach((targetId) => {
        const valid = effectiveTargetLevels(targetId);
        intersection = intersection === null
          ? new Set(valid)
          : new Set([...intersection].filter((level) => valid.has(level)));
      });
      return RULE_REASONING_LEVELS.filter((level) => intersection && intersection.has(level));
    }

    function pruneExposedLevels() {
      const valid = new Set(validExposedLevels());
      exposedLevels = exposedLevels.filter((level) => valid.has(level));
    }

    function renderExposedReasoning() {
      if (manualReasoningInput) manualReasoningInput.checked = manualReasoning;
      if (exposedReasoningGroup) {
        exposedReasoningGroup.style.display = manualReasoning ? "none" : "";
      }
      if (!exposedLevelsContainer || manualReasoning) return;

      pruneExposedLevels();
      const valid = new Set(validExposedLevels());
      exposedLevelsContainer.innerHTML = "";

      if (!selectedTargets.length) {
        exposedLevelsContainer.innerHTML =
          `<span class="rule-reasoning-empty">Select at least one target model to choose exposed reasoning levels.</span>`;
        return;
      }

      RULE_REASONING_LEVELS.forEach((level) => {
        const supported = valid.has(level);
        const label = document.createElement("label");
        label.className = "rule-exposed-level" + (supported ? "" : " is-disabled");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = level;
        checkbox.checked = supported && exposedLevels.includes(level);
        checkbox.disabled = !supported;
        checkbox.addEventListener("change", () => {
          if (checkbox.checked) {
            if (!exposedLevels.includes(level)) exposedLevels.push(level);
          } else {
            exposedLevels = exposedLevels.filter((value) => value !== level);
          }
        });
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(level));
        exposedLevelsContainer.appendChild(label);
      });

      if (exposedReasoningHint) {
        exposedReasoningHint.textContent = valid.size
          ? "Only levels supported by every selected target (or remapped below) can be advertised."
          : "No level is supported by every selected target yet. Add a per-target mapping, or map it to Default / unspecified for models without reasoning support.";
      }
    }

    function mergeAutoTargetDuplicates() {
      if (manualReasoning) return;
      const seenIds = new Set();
      for (let i = 0; i < selectedTargets.length; i++) {
        if (seenIds.has(selectedTargets[i].id)) {
          selectedTargets.splice(i, 1);
          i--;
        } else {
          seenIds.add(selectedTargets[i].id);
        }
      }
    }

    function updatePriorityHint() {
      if (!priorityHint) return;
      const strategy = stratSelect.value;
      if (strategy === "fallback" || strategy === "priority") {
        priorityHint.textContent =
          "Fallback mode: the first selected model is the primary, and each lower entry is the next fallback.";
      } else {
        priorityHint.textContent =
          "The first selected model is tried first. Use the arrows to change priority.";
      }
    }

    function renderAvailableTargets() {
      const query = (targetsSearchInput ? targetsSearchInput.value : "").toLowerCase().trim();
      targetsContainer.innerHTML = "";

      if (catalog.length === 0) {
        targetsContainer.innerHTML = `<span style="color: var(--ink-muted); font-size: 12px;">No provider models available. Enable provider plugins first.</span>`;
        return;
      }

      let visible = 0;
      catalog.forEach((m) => {
        const labelText = `${m.id} (${m.providerName})`;
        if (query && !labelText.toLowerCase().includes(query)) return;
        visible++;

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
        chk.checked = hasSelectedBase(m.id);
        chk.addEventListener("change", () => {
          if (chk.checked) {
            if (!hasSelectedBase(m.id)) selectedTargets.push({ id: m.id, level: "" });
          } else {
            for (let i = selectedTargets.length - 1; i >= 0; i--) {
              if (selectedTargets[i].id === m.id) selectedTargets.splice(i, 1);
            }
          }
          renderSelectedTargets();
        });

        lbl.appendChild(chk);
        lbl.appendChild(document.createTextNode(labelText));
        targetsContainer.appendChild(lbl);
      });

      if (visible === 0) {
        targetsContainer.innerHTML = `<span style="color: var(--ink-muted); font-size: 12px;">No models match this search.</span>`;
      }
    }

    function renderSelectedTargets() {
      selectedContainer.innerHTML = "";

      if (selectedTargets.length === 0) {
        selectedContainer.innerHTML = `<div class="rule-selected-empty">No targets selected yet. Pick models above.</div>`;
        updatePriorityHint();
        renderExposedReasoning();
        return;
      }

      mergeAutoTargetDuplicates();

      selectedTargets.forEach((target, index) => {
        const entry = catalogById.get(target.id);
        const providerName = entry ? entry.providerName : "";
        const levels = levelsForTarget(target);
        const nativeLevels = nativeLevelsForTarget(target.id);
        const row = document.createElement("div");
        row.className = "rule-selected-row";
        row.dataset.targetId = target.id;

        let reasoningHtml = "";
        if (manualReasoning) {
          reasoningHtml = levels.length
            ? `<div class="rule-selected-controls">
                 <label for="rule-target-level-${index}">Reasoning</label>
                 <select class="select-input rule-reasoning-select" id="rule-target-level-${index}">
                   <option value="">Default</option>
                   ${levels
                     .map(
                       (level) =>
                         `<option value="${escapeHtml(level)}"${level === target.level ? " selected" : ""}>${escapeHtml(level)}</option>`
                     )
                     .join("")}
                 </select>
               </div>`
            : "";
        } else {
          const mapping = reasoningOverrides[target.id] || {};
          const mappingRows = Object.entries(mapping)
            .map(([source, destination]) => {
              const sourceOptions = RULE_REASONING_LEVELS
                .map(
                  (level) =>
                    `<option value="${escapeHtml(level)}"${level === source ? " selected" : ""}>${escapeHtml(level)}</option>`
                )
                .join("");
              const destinationLevels = nativeLevels.slice();
              if (!destinationLevels.includes(RULE_DEFAULT_REASONING)) {
                destinationLevels.push(RULE_DEFAULT_REASONING);
              }
              const destinationOptions = destinationLevels
                .map((level) => {
                  const label = level === RULE_DEFAULT_REASONING
                    ? "Default / unspecified"
                    : level;
                  return `<option value="${escapeHtml(level)}"${level === destination ? " selected" : ""}>${escapeHtml(label)}</option>`;
                })
                .join("");
              return `<div class="rule-reasoning-map-row">
                <span class="rule-reasoning-map-label">client asks</span>
                <select class="select-input rule-reasoning-map-source">${sourceOptions}</select>
                <span class="rule-reasoning-map-arrow">→</span>
                <select class="select-input rule-reasoning-map-target">${destinationOptions}</select>
                <button type="button" class="rule-order-btn rule-remove-reasoning-map" title="Remove mapping" aria-label="Remove mapping">✕</button>
              </div>`;
            })
            .join("");

          reasoningHtml = `<div class="rule-reasoning-overrides">
            <div class="rule-selected-controls">
              <span class="rule-reasoning-label">Reasoning mapping</span>
              <button type="button" class="btn btn-secondary btn-sm rule-add-reasoning-map">+ Add mapping</button>
            </div>
            ${mappingRows || `<span class="rule-reasoning-empty">Uses the client's requested level.</span>`}
          </div>`;
        }

        row.innerHTML = `
          <span class="rule-selected-index">${index + 1}</span>
          <div class="rule-selected-info">
            <div class="rule-selected-title">
              <span>${escapeHtml(target.id)}</span>
              ${providerName ? `<span class="badge badge-muted">${escapeHtml(providerName)}</span>` : ""}
            </div>
            ${reasoningHtml}
          </div>
          <div class="rule-selected-actions">
            <button type="button" class="rule-order-btn" data-action="up" title="Move up" aria-label="Move up" ${index === 0 ? "disabled" : ""}>↑</button>
            <button type="button" class="rule-order-btn" data-action="down" title="Move down" aria-label="Move down" ${index === selectedTargets.length - 1 ? "disabled" : ""}>↓</button>
            <button type="button" class="rule-order-btn rule-remove-btn" data-action="remove" title="Remove target" aria-label="Remove target">✕</button>
          </div>
        `;

        const levelSelect = row.querySelector(".rule-reasoning-select");
        if (levelSelect) {
          levelSelect.addEventListener("change", () => {
            selectedTargets[index].level = levelSelect.value;
          });
        }

        const syncReasoningOverrides = () => {
          const next = {};
          row.querySelectorAll(".rule-reasoning-map-row").forEach((mappingRow) => {
            const source = mappingRow.querySelector(".rule-reasoning-map-source").value;
            const destination = mappingRow.querySelector(".rule-reasoning-map-target").value;
            if (source && destination) next[source] = destination;
          });
          reasoningOverrides[target.id] = next;
        };

        row.querySelectorAll(".rule-reasoning-map-row").forEach((mappingRow) => {
          mappingRow.querySelectorAll("select").forEach((select) => {
            select.addEventListener("change", () => {
              syncReasoningOverrides();
              renderSelectedTargets();
            });
          });
          const removeMappingBtn = mappingRow.querySelector(".rule-remove-reasoning-map");
          if (removeMappingBtn) {
            removeMappingBtn.addEventListener("click", () => {
              mappingRow.remove();
              syncReasoningOverrides();
              renderSelectedTargets();
            });
          }
        });

        const addMappingBtn = row.querySelector(".rule-add-reasoning-map");
        if (addMappingBtn) {
          addMappingBtn.addEventListener("click", () => {
            const mapping = reasoningOverrides[target.id] || {};
            const source = RULE_REASONING_LEVELS.find((level) => !mapping[level]);
            const destination = nativeLevels[0] || RULE_DEFAULT_REASONING;
            if (!source) return;
            reasoningOverrides[target.id] = { ...mapping, [source]: destination };
            renderSelectedTargets();
          });
        }

        row.querySelectorAll(".rule-order-btn[data-action]").forEach((btn) => {
          btn.addEventListener("click", () => {
            const action = btn.dataset.action;
            if (action === "remove") {
              selectedTargets.splice(index, 1);
            } else if (action === "up" && index > 0) {
              const previous = selectedTargets[index - 1];
              selectedTargets[index - 1] = selectedTargets[index];
              selectedTargets[index] = previous;
            } else if (action === "down" && index < selectedTargets.length - 1) {
              const next = selectedTargets[index + 1];
              selectedTargets[index + 1] = selectedTargets[index];
              selectedTargets[index] = next;
            }
            renderSelectedTargets();
            if (action === "remove") renderAvailableTargets();
          });
        });

        selectedContainer.appendChild(row);
      });

      updatePriorityHint();
      renderExposedReasoning();
    }

    if (targetsSearchInput) {
      targetsSearchInput.value = "";
      targetsSearchInput.oninput = () => renderAvailableTargets();
    }
    stratSelect.onchange = updatePriorityHint;
    if (manualReasoningInput) {
      manualReasoningInput.checked = manualReasoning;
      manualReasoningInput.onchange = () => {
        manualReasoning = manualReasoningInput.checked;
        if (!manualReasoning) {
          mergeAutoTargetDuplicates();
          if (!exposedLevels.length) exposedLevels = validExposedLevels();
        }
        renderSelectedTargets();
      };
    }

    if (rule) {
      title.textContent = "Edit Custom Load Balancing Rule";
      idInput.value = rule.id;
      mIdInput.value = rule.model_id;
      mIdInput.disabled = false;
      nameInput.value = rule.name;
      stratSelect.value = rule.strategy;
      contextInput.value = rule.context_window || "";
      maxTokensInput.value = rule.max_tokens || "";
      descInput.value = rule.description || "";
      enabledInput.checked = !!rule.enabled;
    } else {
      title.textContent = "Create Custom Load Balancing Rule";
      idInput.value = "";
      mIdInput.value = "";
      mIdInput.disabled = false;
      nameInput.value = "";
      stratSelect.value = "round_robin";
      contextInput.value = "";
      maxTokensInput.value = "";
      descInput.value = "";
      enabledInput.checked = true;
    }

    renderAvailableTargets();
    renderSelectedTargets();
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
      const contextInput = document.getElementById("rule-form-context");
      const maxTokensInput = document.getElementById("rule-form-max-tokens");
      const descInput = document.getElementById("rule-form-desc");
      const enabledInput = document.getElementById("rule-form-enabled");
      const manualReasoningInput = document.getElementById("rule-form-manual-reasoning");

      const manualReasoning = manualReasoningInput ? manualReasoningInput.checked : true;
      const parseLimit = (value) => {
        const trimmed = (value || "").trim();
        if (!trimmed) return null;
        const parsed = parseInt(trimmed, 10);
        return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
      };
      const contextWindow = parseLimit(contextInput.value);
      const maxTokens = parseLimit(maxTokensInput.value);

      const draftTargets = (el.modalRule && el.modalRule._ruleTargets) || [];
      const selectedTargets = draftTargets
        .map((target) => (manualReasoning && target.level ? `${target.id}:${target.level}` : target.id))
        .filter((target, index, all) => target && all.indexOf(target) === index);
      const modelId = mIdInput.value.trim();
      const name = nameInput.value.trim() || modelId;

      let exposedLevels = [];
      const reasoningOverrides = {};
      if (!manualReasoning) {
        exposedLevels = Array.from(
          document.querySelectorAll("#rule-exposed-reasoning-levels input:checked")
        ).map((input) => input.value);
        document.querySelectorAll("#rule-selected-targets .rule-selected-row").forEach((row) => {
          const targetId = row.dataset.targetId;
          if (!targetId) return;
          const mapping = {};
          row.querySelectorAll(".rule-reasoning-map-row").forEach((mappingRow) => {
            const source = mappingRow.querySelector(".rule-reasoning-map-source").value;
            const destination = mappingRow.querySelector(".rule-reasoning-map-target").value;
            if (source && destination) mapping[source] = destination;
          });
          if (Object.keys(mapping).length) reasoningOverrides[targetId] = mapping;
        });
      }

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
              model_id: modelId,
              name,
              strategy: stratSelect.value,
              targets: selectedTargets,
              manual_reasoning: manualReasoning,
              exposed_reasoning_levels: exposedLevels,
              reasoning_overrides: reasoningOverrides,
              context_window: contextWindow,
              max_tokens: maxTokens,
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
              manual_reasoning: manualReasoning,
              exposed_reasoning_levels: exposedLevels,
              reasoning_overrides: reasoningOverrides,
              context_window: contextWindow,
              max_tokens: maxTokens,
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
      state.attachedImages = [];
      renderPlaygroundAttachments();
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

  const MAX_PLAYGROUND_IMAGE_BYTES = 20 * 1024 * 1024;

  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(reader.error || new Error("Could not read image"));
      reader.readAsDataURL(file);
    });
  }

  async function addPlaygroundImages(fileList) {
    const files = Array.from(fileList || []);
    for (const file of files) {
      if (!file.type || !file.type.startsWith("image/")) {
        showToast(`'${file.name}' is not an image`, "error");
        continue;
      }
      if (file.size > MAX_PLAYGROUND_IMAGE_BYTES) {
        showToast(`'${file.name}' is larger than 20 MB`, "error");
        continue;
      }
      try {
        const dataUrl = await fileToDataUrl(file);
        state.attachedImages.push({ name: file.name, dataUrl });
      } catch (err) {
        showToast(`Could not read '${file.name}': ${err.message}`, "error");
      }
    }
    renderPlaygroundAttachments();
  }

  function renderPlaygroundAttachments() {
    if (!el.playgroundAttachments) return;
    el.playgroundAttachments.innerHTML = "";
    if (!state.attachedImages.length) {
      el.playgroundAttachments.hidden = true;
      return;
    }
    el.playgroundAttachments.hidden = false;
    state.attachedImages.forEach((image, index) => {
      const wrapper = document.createElement("div");
      wrapper.className = "chat-attachment";
      const img = document.createElement("img");
      img.src = image.dataUrl;
      img.alt = image.name || `Image ${index + 1}`;
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "chat-attachment-remove";
      removeBtn.textContent = "✕";
      removeBtn.title = "Remove image";
      removeBtn.addEventListener("click", () => {
        state.attachedImages.splice(index, 1);
        renderPlaygroundAttachments();
      });
      wrapper.appendChild(img);
      wrapper.appendChild(removeBtn);
      el.playgroundAttachments.appendChild(wrapper);
    });
  }

  if (el.playgroundAttachBtn && el.playgroundImageInput) {
    el.playgroundAttachBtn.addEventListener("click", () => el.playgroundImageInput.click());
    el.playgroundImageInput.addEventListener("change", async (e) => {
      await addPlaygroundImages(e.target.files);
      e.target.value = "";
    });
  }

  function buildUserContent(prompt, images) {
    if (!images.length) return prompt;
    const parts = [];
    if (prompt) parts.push({ type: "text", text: prompt });
    images.forEach((image) => {
      parts.push({ type: "image_url", image_url: { url: image.dataUrl } });
    });
    return parts;
  }

  async function sendPlaygroundMessage() {
    const prompt = (el.playgroundPrompt ? el.playgroundPrompt.value : "").trim();
    const images = state.attachedImages.slice();
    if (!prompt && !images.length) return;

    const model = el.playgroundModelSelect ? el.playgroundModelSelect.value : (state.selectedSnippetModel || "test-model");
    const temp = parseFloat(el.playgroundTempInput.value) || 1.0;
    const maxTokens = parseInt(el.playgroundTokensInput.value, 10) || undefined;
    const systemPrompt = (el.playgroundSystemInput.value || "").trim();
    const userContent = buildUserContent(prompt, images);

    el.playgroundPrompt.value = "";
    state.attachedImages = [];
    renderPlaygroundAttachments();

    // Append user message to UI
    appendChatBubble("user", prompt || "(image)", images.map((image) => image.dataUrl));
    state.chatMessages.push({ role: "user", content: userContent });

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

      const replyMessage = (res.choices && res.choices[0] && res.choices[0].message)
        ? res.choices[0].message
        : null;
      const replyContent = replyMessage
        ? replyMessage.content
        : (res.response || JSON.stringify(res));
      const reasoningContent = replyMessage
        ? (replyMessage.reasoning_content || replyMessage.reasoning || "")
        : "";

      if (reasoningContent) {
        assistantBubble.innerHTML = `
          <div class="chat-reasoning">
            <div class="chat-reasoning-label">Reasoning</div>
            <div class="chat-reasoning-text">${escapeHtml(reasoningContent)}</div>
          </div>
          <div class="chat-answer">${escapeHtml(replyContent)}</div>
        `;
      } else {
        assistantBubble.textContent = replyContent;
      }
      state.chatMessages.push({ role: "assistant", content: replyContent });
    } catch (err) {
      assistantBubble.innerHTML = `<span style="color: var(--accent-red);">Error: ${escapeHtml(err.message)}</span>`;
    }
  }

  function appendChatBubble(role, text, imageUrls = []) {
    if (!el.playgroundHistory) return null;
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-msg ${role}`;
    msgDiv.innerHTML = `
      <div style="font-size: 11px; color: var(--ink-muted); text-transform: uppercase; font-weight: 600;">${role}</div>
      <div class="chat-bubble" style="background: ${role === 'user' ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.02)'}; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px 14px; font-size: 13px; white-space: pre-wrap; line-height: 1.5;">${escapeHtml(text)}</div>
    `;
    if (imageUrls.length) {
      const imagesDiv = document.createElement("div");
      imagesDiv.className = "chat-message-images";
      imageUrls.forEach((url, index) => {
        const img = document.createElement("img");
        img.src = url;
        img.alt = `Attached image ${index + 1}`;
        imagesDiv.appendChild(img);
      });
      msgDiv.appendChild(imagesDiv);
    }
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
    await renderStatusAccountsGrid(fetchFreshQuota);
    renderStatusProviderTabs();
    renderLogsTable();
    if (state.statusView === "graph") renderGraphView();
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
      renderLogsTable();
      if (state.statusView === "graph") renderGraphView();
    });
    el.statusProviderTabs.appendChild(allBtn);

    // Sort providers by name with pinned first
    const usageProviders = state.usageProviderIds instanceof Set ? state.usageProviderIds : null;
    const sortedPlugins = [...state.plugins]
      .filter((plugin) => !usageProviders || usageProviders.has(plugin.id))
      .sort((a, b) => {
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
        renderLogsTable();
        if (state.statusView === "graph") renderGraphView();
      });
      el.statusProviderTabs.appendChild(btn);
    });
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

  // --- Provider tier / plan labelling -------------------------------------

  const TIER_LABELS = {
    "free-tier": "Free",
    "g1-pro-tier": "Google AI Pro",
    "g1-ultra-tier": "Google AI Ultra",
    free: "Free",
    plus: "Plus",
    pro: "Pro",
    team: "Team",
    business: "Business",
    enterprise: "Enterprise",
    standard: "Standard",
  };

  function resolveAccountTier(acc, quota) {
    const providerId = acc.provider_id;
    const q = quota || {};
    let raw = "";
    if (providerId === "antigravity") {
      raw = acc.tier || q.tier || "free-tier";
      const label = q.tier_name || acc.tier_name || TIER_LABELS[raw] || raw;
      return { key: String(raw || "free-tier").toLowerCase(), label, pro: /pro|ultra/i.test(raw) };
    }
    if (providerId === "codex") {
      raw = acc.plan_type || q.plan_type || "standard";
      return { key: String(raw).toLowerCase(), label: TIER_LABELS[raw] || raw, pro: /plus|pro|team|business|enterprise/i.test(raw) };
    }
    if (providerId === "openrouter") {
      const isFree = q.is_free_tier === true;
      return { key: isFree ? "free" : "paid", label: isFree ? "Free" : "Paid", pro: !isFree };
    }
    if (providerId === "opencode_go") {
      return { key: "go", label: "Go", pro: true };
    }
    raw = acc.tier || q.tier || acc.plan_type || q.plan_type || "standard";
    return { key: String(raw).toLowerCase(), label: TIER_LABELS[raw] || raw, pro: false };
  }

  function quotaHasRenderableUsage(quota) {
    if (!quota || quota.error) return false;
    if (typeof quota.available_credits === "number") return true;
    if (quota.balance && typeof quota.balance === "object") return true;
    if (Array.isArray(quota.summary) && quota.summary.length > 0) return true;
    if (Array.isArray(quota.groups) && quota.groups.some((g) => g && Array.isArray(g.buckets) && g.buckets.length > 0)) {
      return true;
    }
    return false;
  }

  function bucketRemainingFraction(bucket) {
    if (bucket.remainingFraction !== undefined && bucket.remainingFraction !== null) {
      return Math.min(1, Math.max(0, Number(bucket.remainingFraction)));
    }
    if (bucket.usedPercent !== undefined && bucket.usedPercent !== null) {
      return Math.min(1, Math.max(0, 1 - Number(bucket.usedPercent) / 100));
    }
    return 1;
  }

  // --- Pool grouping -------------------------------------------------------

  function buildUsagePools(accountsToDisplay, quotaDataMap) {
    const providerMap = {};
    state.plugins.forEach((p) => { providerMap[p.id] = p; });

    const pools = new Map();
    accountsToDisplay.forEach((acc) => {
      if (acc.disable_usage_tracking) return;
      const quota = quotaDataMap[acc.id];
      if (!quotaHasRenderableUsage(quota)) return;

      const provider = providerMap[acc.provider_id] || { id: acc.provider_id, name: acc.provider_id, pinned: false };
      const tier = resolveAccountTier(acc, quota);
      const key = `${acc.provider_id}::${tier.key}`;

      let pool = pools.get(key);
      if (!pool) {
        pool = {
          key,
          providerId: acc.provider_id,
          providerName: provider.name,
          providerPinned: !!provider.pinned || !!acc.pinned,
          tierKey: tier.key,
          tierLabel: tier.label,
          tierPro: !!tier.pro,
          accounts: [],
          groups: new Map(),
          creditSum: 0,
          hasCredits: false,
          balances: new Map(),
          summaries: new Map(),
        };
        pools.set(key, pool);
      }
      if (acc.pinned) pool.providerPinned = true;
      pool.accounts.push({ acc, quota, tier });

      if (quota.balance && typeof quota.balance === "object" && typeof quota.balance.value === "number") {
        const label = quota.balance.label || "available balance";
        const prev = pool.balances.get(label);
        pool.balances.set(label, (prev || 0) + quota.balance.value);
      }
      if (typeof quota.available_credits === "number") {
        pool.hasCredits = true;
        pool.creditSum += quota.available_credits;
      }
      if (Array.isArray(quota.summary)) {
        quota.summary.forEach((item) => {
          if (!item || item.label === undefined) return;
          const label = String(item.label);
          const numeric = typeof item.value === "number" ? item.value : null;
          if (numeric === null) return;
          pool.summaries.set(label, (pool.summaries.get(label) || 0) + numeric);
        });
      }

      (quota.groups || []).forEach((grp) => {
        if (!grp || !Array.isArray(grp.buckets)) return;
        const groupName = grp.displayName || "Usage";
        let group = pool.groups.get(groupName);
        if (!group) {
          group = { displayName: groupName, buckets: new Map() };
          pool.groups.set(groupName, group);
        }
        grp.buckets.forEach((bucket) => {
          const bucketKey = bucket.bucketId || bucket.displayName || "limit";
          let entry = group.buckets.get(bucketKey);
          if (!entry) {
            entry = {
              bucketId: bucketKey,
              displayName: bucket.displayName || bucketKey,
              fractions: [],
              resets: [],
              accountCount: 0,
            };
            group.buckets.set(bucketKey, entry);
          }
          entry.fractions.push(bucketRemainingFraction(bucket));
          if (bucket.resetTime) entry.resets.push(bucket.resetTime);
          entry.accountCount += 1;
        });
      });
    });

    const result = Array.from(pools.values()).map((pool) => {
      const activeAccounts = pool.accounts.filter(({ acc }) => acc.enabled);
      const cooldownCount = pool.accounts.filter(({ acc }) => acc.rate_limited_until && acc.rate_limited_until > Date.now() / 1000).length;
      const errorCount = pool.accounts.filter(({ quota }) => quota && quota.error).length;
      const groups = Array.from(pool.groups.values()).map((group) => ({
        displayName: group.displayName,
        buckets: Array.from(group.buckets.values()).map((entry) => {
          const avg = entry.fractions.length
            ? entry.fractions.reduce((sum, value) => sum + value, 0) / entry.fractions.length
            : 1;
          const resets = entry.resets
            .map((value) => new Date(value).getTime())
            .filter((value) => !isNaN(value))
            .sort((a, b) => a - b);
          return {
            bucketId: entry.bucketId,
            displayName: entry.displayName,
            remainingFraction: avg,
            resetTime: resets.length ? new Date(resets[0]).toISOString() : null,
            accountCount: entry.accountCount,
          };
        }),
      }));
      const totalBuckets = groups.reduce((sum, group) => sum + group.buckets.length, 0);
      const poolRemaining = totalBuckets
        ? groups.reduce((sum, group) => sum + group.buckets.reduce((s, b) => s + b.remainingFraction, 0), 0) / totalBuckets
        : null;
      return {
        ...pool,
        groups,
        accountTotal: pool.accounts.length,
        activeCount: activeAccounts.length,
        cooldownCount,
        errorCount,
        poolRemaining,
      };
    });

    result.sort((a, b) => {
      if (a.providerPinned !== b.providerPinned) return a.providerPinned ? -1 : 1;
      const nameCmp = a.providerName.localeCompare(b.providerName);
      if (nameCmp !== 0) return nameCmp;
      if (a.tierPro !== b.tierPro) return a.tierPro ? -1 : 1;
      return a.tierLabel.localeCompare(b.tierLabel);
    });

    return { pools: result, providerMap };
  }

  function currentStatusAccounts() {
    return state.statusProviderFilter
      ? state.accounts.filter((a) => a.provider_id === state.statusProviderFilter)
      : [...state.accounts];
  }

  function getQuotaDataMap() {
    return state.cachedQuotaData || {};
  }

  // --- Pooled quota rendering ---------------------------------------------

  async function loadQuotaData(accountsToDisplay, fetchFreshQuota) {
    if (!(fetchFreshQuota || !state.cachedQuotaData)) return;
    if (!state.cachedQuotaData) state.cachedQuotaData = {};
    if (fetchFreshQuota) {
      accountsToDisplay.forEach((acc) => {
        delete state.cachedQuotaData[acc.id];
      });
    }

    const has = (providerId) => accountsToDisplay.some((a) => a.provider_id === providerId);

    const jobs = [];
    const collect = (path, key) => api(path)
      .then((res) => {
        if (res && res.accounts) {
          res.accounts.forEach((item) => {
            if (item && item[key]) state.cachedQuotaData[item.account_id] = item[key];
          });
        }
      })
      .catch(() => {});

    if (has("antigravity")) jobs.push(collect("/api/antigravity/status", "quota"));
    if (has("codex")) jobs.push(collect("/api/codex/status", "quota"));
    if (has("openrouter")) jobs.push(collect("/api/openrouter/status", "credits"));
    if (has("opencode_go")) jobs.push(collect("/api/opencode-go/status", "quota"));

    const pluginMeteredAccounts = accountsToDisplay.filter((account) => {
      const plugin = (state.plugins || []).find((item) => item.id === account.provider_id);
      return Boolean(plugin && plugin.supports_usage_meter);
    });
    if (pluginMeteredAccounts.length > 0) {
      jobs.push(collect("/api/provider-usage/status", "quota"));
    }

    await Promise.all(jobs);
  }

  function quotaBarColor(pct) {
    return pct > 40 ? "#34d399" : pct > 10 ? "#fbbf24" : "#f43f5e";
  }

  function renderPctBadge(pct) {
    return `<span style="font-family: var(--font-mono); font-size: 11.5px; font-weight: 600; color: ${quotaBarColor(pct)};">${pct}%</span>`;
  }

  async function renderStatusAccountsGrid(fetchFreshQuota = false) {
    if (!el.statusAccountsGrid) return;

    if (el.statusAccountsGrid.children.length === 0) {
      el.statusAccountsGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--ink-muted);">
          Loading accounts status &amp; quota limits...
        </div>
      `;
    }

    const accountsToDisplay = currentStatusAccounts();
    if (accountsToDisplay.length === 0) {
      el.statusAccountsGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--ink-muted);">
          No accounts configured for this provider.
        </div>
      `;
      if (el.statusPoolSummaryBadge) el.statusPoolSummaryBadge.textContent = "0 pools";
      return;
    }

    await loadQuotaData(accountsToDisplay, fetchFreshQuota);

    const quotaDataMap = getQuotaDataMap();
    const { pools } = buildUsagePools(accountsToDisplay, quotaDataMap);

    const usageProviderIds = new Set();
    accountsToDisplay.forEach((acc) => {
      if (quotaHasRenderableUsage(quotaDataMap[acc.id])) usageProviderIds.add(acc.provider_id);
    });
    state.usageProviderIds = usageProviderIds;
    if (state.statusProviderFilter && !usageProviderIds.has(state.statusProviderFilter)) {
      state.statusProviderFilter = "";
      renderStatusProviderTabs();
    }

    if (pools.length === 0) {
      el.statusAccountsGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--ink-muted);">
          No providers are reporting usage yet. Usage appears here after a metered provider returns a quota snapshot.
        </div>
      `;
      if (el.statusPoolSummaryBadge) el.statusPoolSummaryBadge.textContent = "0 pools";
      return;
    }

    if (el.statusPoolSummaryBadge) {
      el.statusPoolSummaryBadge.textContent = `${pools.length} pool${pools.length === 1 ? "" : "s"}`;
    }
    if (el.statusPoolSubtitle) {
      const accountCount = pools.reduce((sum, pool) => sum + pool.accountTotal, 0);
      el.statusPoolSubtitle.textContent = `${accountCount} account${accountCount === 1 ? "" : "s"} grouped into ${pools.length} usage window${pools.length === 1 ? "" : "s"}`;
    }

    el.statusAccountsGrid.innerHTML = "";

    pools.forEach((pool) => {
      const card = document.createElement("div");
      const isExpanded = state.expandedStatusGroups.has(pool.key);
      const hiddenSet = state.hiddenLimits[pool.key] || new Set();

      const tierClass = pool.tierPro ? "is-pro" : /free/i.test(pool.tierKey) ? "is-free" : "";
      const statusBits = [
        `<span class="badge badge-active" style="font-size: 10.5px;">${pool.activeCount} active</span>`,
      ];
      if (pool.cooldownCount > 0) {
        statusBits.push(`<span class="badge" style="font-size: 10.5px; color: var(--accent-red);">${pool.cooldownCount} cooling</span>`);
      }
      if (pool.errorCount > 0) {
        statusBits.push(`<span class="badge" style="font-size: 10.5px; color: var(--accent-amber);">${pool.errorCount} no data</span>`);
      }

      let limitsHtml = "";
      pool.groups.forEach((group) => {
        group.buckets.forEach((bucket) => {
          const bucketKey = bucket.bucketId || bucket.displayName || "limit";
          if (hiddenSet.has(bucketKey)) return;
          const pct = Math.round(bucket.remainingFraction * 100);
          const resetStr = formatTimeRemaining(bucket.resetTime);
          const coverage = bucket.accountCount > 1
            ? `across ${bucket.accountCount} accounts`
            : group.displayName;
          limitsHtml += `
            <div class="limit-item">
              <div class="limit-header">
                <span style="font-size: 12px; font-weight: 600; color: var(--ink-primary);">${escapeHtml(bucket.displayName)}</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                  ${renderPctBadge(pct)}
                  <button class="btn-hide-limit" data-pool="${escapeHtml(pool.key)}" data-key="${escapeHtml(bucketKey)}" title="Hide this limit">✕</button>
                </div>
              </div>
              <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 5px; overflow: hidden;">
                <div style="background: ${quotaBarColor(pct)}; height: 100%; width: ${pct}%;"></div>
              </div>
              <div style="display: flex; justify-content: space-between; gap: 8px; font-size: 10.5px; color: var(--ink-muted);">
                <span>${escapeHtml(coverage)}</span>
                <span>Resets: ${escapeHtml(resetStr)}</span>
              </div>
            </div>
          `;
        });
      });

      if (pool.hasCredits && pool.creditSum > 0) {
        limitsHtml += `
          <div class="credits-panel">
            <span class="credits-value">$${pool.creditSum.toFixed(2)}</span>
            <span class="credits-label">pooled credits (${pool.accountTotal} accounts)</span>
          </div>
        `;
      }

      pool.balances.forEach((value, label) => {
        limitsHtml += `
          <div class="credits-panel">
            <span class="credits-value">$${value.toFixed(2)}</span>
            <span class="credits-label">${escapeHtml(label)} (pooled)</span>
          </div>
        `;
      });

      if (pool.summaries.size > 0) {
        const rows = Array.from(pool.summaries.entries()).map(([label, value]) => `
          <div class="usage-summary-row">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(String(Math.round(value * 100) / 100))}</strong>
          </div>
        `).join("");
        limitsHtml += `<div class="usage-summary-panel">${rows}</div>`;
      }

      if (!limitsHtml) {
        limitsHtml = `<div style="font-size: 11.5px; color: var(--ink-muted); padding: 4px 0;">No live limits reported for this pool yet.</div>`;
      }

      const accountRows = pool.accounts.map(({ acc, quota }) => {
        const inCooldown = acc.rate_limited_until && acc.rate_limited_until > Date.now() / 1000;
        const accColor = quota && quota.error
          ? "var(--accent-amber)"
          : inCooldown
            ? "var(--accent-red)"
            : "var(--accent-green)";
        const accState = quota && quota.error ? "No data" : inCooldown ? "Cooling" : acc.enabled ? "Active" : "Disabled";
        const accToken = quota && typeof quota.available_credits === "number"
          ? `$${quota.available_credits.toFixed(2)}`
          : "";
        return `
          <div class="pool-account-row">
            <span class="pool-account-name" title="${escapeHtml(acc.name)}">${escapeHtml(acc.name)}</span>
            <span style="display: inline-flex; align-items: center; gap: 8px; flex-shrink: 0;">
              ${accToken ? `<span class="pool-account-stat" style="color: var(--ink-secondary);">${escapeHtml(accToken)}</span>` : ""}
              <span class="pool-account-stat" style="color: ${accColor}; font-family: var(--font-sans); font-size: 10.5px;">${accState}</span>
            </span>
          </div>
        `;
      }).join("");

      const hiddenCount = hiddenSet.size;

      card.className = `account-square-card ${pool.providerPinned ? "is-pinned" : ""}`;
      card.innerHTML = `
        <div class="account-square-header">
          <div style="min-width: 0; flex: 1;">
            <div class="account-square-title">${escapeHtml(pool.providerName)}</div>
            <div class="pool-header-meta">
              <span class="pool-tier-badge ${tierClass}">${escapeHtml(pool.tierLabel)}</span>
              <span class="pool-count-pill">${pool.accountTotal} account${pool.accountTotal === 1 ? "" : "s"}</span>
            </div>
          </div>
          <div style="display: flex; align-items: center; gap: 6px; flex-shrink: 0;">
            <button class="btn-pin ${pool.providerPinned ? "pinned" : ""} btn-toggle-pool-pin" data-pin-provider="${escapeHtml(pool.providerId)}" title="${pool.providerPinned ? "Unpin provider" : "Pin provider to top"}">
              ${renderPinSvg(pool.providerPinned, 12)}
            </button>
          </div>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 5px;">${statusBits.join("")}</div>
        <div style="display: flex; flex-direction: column; gap: 8px; flex: 1;">
          ${limitsHtml}
        </div>
        <button class="pool-accounts-toggle" type="button">
          <span>${isExpanded ? "Hide" : "Show"} accounts</span>
          <span style="display: inline-flex; align-items: center; gap: 6px;">
            ${hiddenCount > 0 ? `<span class="pool-count-pill">${hiddenCount} hidden</span>` : ""}
            <span>${isExpanded ? "▲" : "▼"}</span>
          </span>
        </button>
        <div class="pool-accounts-drawer" ${isExpanded ? "" : "hidden"}>
          ${accountRows}
          ${hiddenCount > 0 ? `<button class="btn btn-secondary btn-sm btn-unhide-pool" style="font-size: 10.5px; padding: 2px 8px; align-self: flex-start;">Unhide all (${hiddenCount})</button>` : ""}
        </div>
      `;

      const pinBtn = card.querySelector(".btn-toggle-pool-pin");
      if (pinBtn) {
        pinBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const plugin = state.plugins.find((p) => p.id === pool.providerId);
          if (plugin) togglePluginPin(plugin);
        });
      }

      const toggleBtn = card.querySelector(".pool-accounts-toggle");
      if (toggleBtn) {
        toggleBtn.addEventListener("click", () => {
          if (state.expandedStatusGroups.has(pool.key)) {
            state.expandedStatusGroups.delete(pool.key);
          } else {
            state.expandedStatusGroups.add(pool.key);
          }
          renderStatusAccountsGrid(false);
        });
      }

      card.querySelectorAll(".btn-hide-limit").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          const poolKey = btn.dataset.pool;
          const key = btn.dataset.key;
          if (!state.hiddenLimits[poolKey]) state.hiddenLimits[poolKey] = new Set();
          state.hiddenLimits[poolKey].add(key);
          saveHiddenLimits();
          renderStatusAccountsGrid(false);
        });
      });

      const unhideBtn = card.querySelector(".btn-unhide-pool");
      if (unhideBtn) {
        unhideBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          delete state.hiddenLimits[pool.key];
          saveHiddenLimits();
          renderStatusAccountsGrid(false);
        });
      }

      el.statusAccountsGrid.appendChild(card);
    });
  }

  // --- Request Audit Logs --------------------------------------------------

  async function fetchLogs() {
    try {
      const data = await api("/api/logs?limit=1000");
      state.logs = data || [];
      renderLogsTable();
    } catch (err) {
      console.error("Failed to load logs", err);
    }
  }

  function filteredLogs() {
    if (!state.statusProviderFilter) return state.logs;
    return state.logs.filter((log) => log.plugin_id === state.statusProviderFilter);
  }

  function renderLogsTable() {
    if (!el.logsTableBody) return;
    el.logsTableBody.innerHTML = "";

    const logs = filteredLogs();
    if (el.logsFilterBadge) {
      const provider = state.plugins.find((p) => p.id === state.statusProviderFilter);
      el.logsFilterBadge.textContent = provider ? provider.name : "All Providers";
    }

    if (logs.length === 0) {
      el.logsTableBody.innerHTML = `
        <tr>
          <td colspan="9" style="text-align: center; color: var(--ink-muted); padding: 24px;">
            No requests logged yet for this filter. Make requests to <code>/v1/chat/completions</code> to see real-time persistent audit records.
          </td>
        </tr>
      `;
      return;
    }

    logs.slice(0, 100).forEach((log) => {
      const tr = document.createElement("tr");
      const timeStr = log.timestamp ? new Date(log.timestamp * 1000).toLocaleTimeString() : "";
      const isSuccess = log.status === "success" && (log.status_code >= 200 && log.status_code < 300);

      const statusBadge = isSuccess
        ? `<span class="badge badge-active">${log.status_code || 200} OK</span>`
        : `<span class="badge" style="color: var(--accent-red); border-color: var(--accent-red-border);">${log.status_code || 500} ERR</span>`;

      const cacheRead = log.cache_read_input_tokens || 0;
      const cacheCreated = log.cache_creation_input_tokens || 0;
      const cachePrompt = Math.max(log.prompt_tokens || 0, cacheRead + cacheCreated);
      const cacheHitRate = cachePrompt > 0 ? Math.round((cacheRead / cachePrompt) * 1000) / 10 : 0;
      const cacheTitle = `${cacheRead} cached read / ${cacheCreated} cached written`;
      const cacheCell = cacheRead > 0
        ? `<span class="badge badge-active" title="${escapeHtml(cacheTitle)}">${cacheHitRate}% hit</span>`
        : cacheCreated > 0
          ? `<span class="badge" title="${escapeHtml(cacheTitle)}">${cacheCreated} written</span>`
          : `<span style="color: var(--ink-muted);">—</span>`;

      tr.innerHTML = `
        <td data-label="Time" class="cell-time">${escapeHtml(timeStr)}</td>
        <td data-label="Status">${statusBadge}</td>
        <td data-label="Model"><strong class="cell-strong">${escapeHtml(log.model)}</strong></td>
        <td data-label="Provider"><span class="badge">${escapeHtml(log.plugin_id || "load_balancer")}</span></td>
        <td data-label="API Key"><code class="cell-key">${escapeHtml(log.api_key_id || "direct")}</code></td>
        <td data-label="Latency" class="cell-num">${log.latency_ms || 0} ms</td>
        <td data-label="Tokens" class="cell-num">${log.prompt_tokens || 0} / ${log.completion_tokens || 0}</td>
        <td data-label="Cache">${cacheCell}</td>
        <td data-label="Details" class="cell-details">${escapeHtml(log.error_message || "Completed")}</td>
      `;
      el.logsTableBody.appendChild(tr);
    });
  }

  // --- Graph view ----------------------------------------------------------

  const GRAPH_COLORS = [
    "#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa",
    "#22d3ee", "#fb923c", "#4ade80", "#e879f9", "#facc15",
    "#38bdf8", "#f87171",
  ];

  const GRAPH_RANGE_MS = {
    "1h": 3600 * 1000,
    "6h": 6 * 3600 * 1000,
    "24h": 24 * 3600 * 1000,
    "7d": 7 * 24 * 3600 * 1000,
  };

  function graphProviderColor(providerId) {
    const ids = Array.from(new Set(state.logs.map((log) => log.plugin_id || "load_balancer"))).sort();
    const index = Math.max(0, ids.indexOf(providerId));
    return GRAPH_COLORS[index % GRAPH_COLORS.length];
  }

  function graphProviderName(providerId) {
    const plugin = state.plugins.find((p) => p.id === providerId);
    return plugin ? plugin.name : providerId;
  }

  function graphLogs() {
    const now = Date.now();
    const rangeMs = state.graphRange === "all" ? null : GRAPH_RANGE_MS[state.graphRange];
    return state.logs.filter((log) => {
      if (!log.timestamp) return false;
      if (state.graphProviders.size > 0 && !state.graphProviders.has(log.plugin_id || "load_balancer")) {
        return false;
      }
      if (rangeMs !== null && now - log.timestamp * 1000 > rangeMs) return false;
      return true;
    });
  }

  function graphBuckets(logs) {
    const now = Date.now();
    const rangeMs = state.graphRange === "all" ? null : GRAPH_RANGE_MS[state.graphRange];
    let start;
    let bucketCount;
    if (rangeMs === null) {
      const oldest = logs.reduce((min, log) => Math.min(min, log.timestamp * 1000), now);
      const span = Math.max(now - oldest, 15 * 60 * 1000);
      bucketCount = 24;
      start = now - span;
    } else {
      start = now - rangeMs;
      bucketCount = state.graphRange === "7d" ? 14 : 12;
    }
    const bucketMs = Math.max(1, (now - start) / bucketCount);

    const buckets = Array.from({ length: bucketCount }, (_, index) => ({
      start: start + index * bucketMs,
      end: start + (index + 1) * bucketMs,
      totals: {},
      requests: 0,
      promptTokens: 0,
      completionTokens: 0,
      cacheRead: 0,
      cacheablePrompt: 0,
      latencySum: 0,
      latencyCount: 0,
      errors: 0,
    }));

    logs.forEach((log) => {
      const ts = log.timestamp * 1000;
      let index = Math.floor((ts - start) / bucketMs);
      if (index < 0) index = 0;
      if (index >= bucketCount) index = bucketCount - 1;
      const bucket = buckets[index];
      const providerId = log.plugin_id || "load_balancer";
      const prompt = Number(log.prompt_tokens || 0);
      const completion = Number(log.completion_tokens || 0);
      const cacheRead = Number(log.cache_read_input_tokens || 0);
      const cacheCreated = Number(log.cache_creation_input_tokens || 0);

      bucket.requests += 1;
      bucket.promptTokens += prompt;
      bucket.completionTokens += completion;
      bucket.cacheRead += cacheRead;
      bucket.cacheablePrompt += Math.max(prompt, cacheRead + cacheCreated);
      bucket.latencySum += Number(log.latency_ms || 0);
      bucket.latencyCount += 1;
      if (log.status !== "success" || (log.status_code && (log.status_code < 200 || log.status_code >= 300))) {
        bucket.errors += 1;
      }

      const entry = bucket.totals[providerId] || { requests: 0, tokens: 0, cacheRead: 0, cacheablePrompt: 0, latencySum: 0, latencyCount: 0, errors: 0 };
      entry.requests += 1;
      entry.tokens += prompt + completion;
      entry.cacheRead += cacheRead;
      entry.cacheablePrompt += Math.max(prompt, cacheRead + cacheCreated);
      entry.latencySum += Number(log.latency_ms || 0);
      entry.latencyCount += 1;
      if (log.status !== "success") entry.errors += 1;
      bucket.totals[providerId] = entry;
    });

    return buckets;
  }

  function bucketMetricValue(bucket, providerId) {
    const entry = bucket.totals[providerId];
    if (state.graphMetric === "requests") return entry ? entry.requests : 0;
    if (state.graphMetric === "cache") {
      if (!entry || entry.cacheablePrompt <= 0) return null;
      return (entry.cacheRead / entry.cacheablePrompt) * 100;
    }
    if (state.graphMetric === "latency") {
      if (!entry || entry.latencyCount <= 0) return null;
      return entry.latencySum / entry.latencyCount;
    }
    return entry ? entry.tokens : 0;
  }

  function bucketMetricTotal(bucket) {
    if (state.graphMetric === "requests") return bucket.requests;
    if (state.graphMetric === "cache") {
      return bucket.cacheablePrompt > 0 ? (bucket.cacheRead / bucket.cacheablePrompt) * 100 : null;
    }
    if (state.graphMetric === "latency") {
      return bucket.latencyCount > 0 ? bucket.latencySum / bucket.latencyCount : null;
    }
    return bucket.promptTokens + bucket.completionTokens;
  }

  function formatCompactNumber(value) {
    if (value === null || value === undefined || isNaN(value)) return "—";
    const abs = Math.abs(value);
    if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
    return `${Math.round(value * 10) / 10}`;
  }

  function metricLabel() {
    return {
      tokens: "Tokens",
      requests: "Requests",
      cache: "Cache Hit Rate",
      latency: "Average Latency",
    }[state.graphMetric];
  }

  function metricValueLabel(value) {
    if (value === null || value === undefined) return "—";
    if (state.graphMetric === "cache") return `${Math.round(value * 10) / 10}%`;
    if (state.graphMetric === "latency") return `${Math.round(value)} ms`;
    return formatCompactNumber(value);
  }

  function renderGraphProviderFilters() {
    if (!el.graphProviderFilters) return;
    const providerIds = Array.from(new Set(state.logs.map((log) => log.plugin_id || "load_balancer"))).sort();
    el.graphProviderFilters.innerHTML = "";

    const allBtn = document.createElement("button");
    allBtn.type = "button";
    allBtn.className = `status-provider-tab ${state.graphProviders.size === 0 ? "active" : ""}`;
    allBtn.textContent = "All Providers";
    allBtn.addEventListener("click", () => {
      state.graphProviders.clear();
      renderGraphView();
    });
    el.graphProviderFilters.appendChild(allBtn);

    providerIds.forEach((providerId) => {
      const isActive = state.graphProviders.has(providerId);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `status-provider-tab ${isActive ? "active" : ""}`;
      btn.innerHTML = `<span class="graph-provider-dot" style="background: ${graphProviderColor(providerId)};${isActive ? "" : "opacity:0.55;"}"></span><span>${escapeHtml(graphProviderName(providerId))}</span>`;
      btn.addEventListener("click", () => {
        if (state.graphProviders.has(providerId)) {
          state.graphProviders.delete(providerId);
        } else {
          state.graphProviders.add(providerId);
        }
        renderGraphView();
      });
      el.graphProviderFilters.appendChild(btn);
    });
  }

  function renderGraphKpis(logs) {
    const requests = logs.length;
    const successes = logs.filter((log) => log.status === "success").length;
    const successRate = requests > 0 ? Math.round((successes / requests) * 1000) / 10 : 100;
    const promptTokens = logs.reduce((sum, log) => sum + Number(log.prompt_tokens || 0), 0);
    const completionTokens = logs.reduce((sum, log) => sum + Number(log.completion_tokens || 0), 0);
    const cacheRead = logs.reduce((sum, log) => sum + Number(log.cache_read_input_tokens || 0), 0);
    const cacheable = logs.reduce((sum, log) => sum + Math.max(Number(log.prompt_tokens || 0), Number(log.cache_read_input_tokens || 0) + Number(log.cache_creation_input_tokens || 0)), 0);
    const cacheRate = cacheable > 0 ? Math.round((cacheRead / cacheable) * 1000) / 10 : 0;
    const latency = requests > 0
      ? logs.reduce((sum, log) => sum + Number(log.latency_ms || 0), 0) / requests
      : 0;
    const errors = logs.filter((log) => log.status !== "success").length;

    if (el.graphKpiRequests) el.graphKpiRequests.textContent = String(requests);
    if (el.graphKpiSuccess) el.graphKpiSuccess.textContent = `${successRate}% success`;
    if (el.graphKpiTokens) el.graphKpiTokens.textContent = formatCompactNumber(promptTokens + completionTokens);
    if (el.graphKpiTokensSub) el.graphKpiTokensSub.textContent = `${formatCompactNumber(promptTokens)} in / ${formatCompactNumber(completionTokens)} out`;
    if (el.graphKpiCache) el.graphKpiCache.textContent = `${cacheRate}%`;
    if (el.graphKpiCacheSub) el.graphKpiCacheSub.textContent = `${formatCompactNumber(cacheRead)} cached tokens`;
    if (el.graphKpiLatency) el.graphKpiLatency.textContent = `${Math.round(latency)} ms`;
    if (el.graphKpiErrors) el.graphKpiErrors.textContent = `${errors} error${errors === 1 ? "" : "s"}`;
  }

  function renderGraphTimeseries(logs) {
    if (!el.graphTimeseriesContainer) return;
    const container = el.graphTimeseriesContainer;
    container.innerHTML = "";

    if (logs.length === 0) {
      container.innerHTML = `<div class="graph-empty-state">No requests match this provider filter and time range yet.</div>`;
      if (el.graphTimeseriesLegend) el.graphTimeseriesLegend.innerHTML = "";
      return;
    }

    const buckets = graphBuckets(logs);
    const providerIds = Array.from(new Set(logs.map((log) => log.plugin_id || "load_balancer"))).sort();

    if (el.graphTimeseriesLegend) {
      el.graphTimeseriesLegend.innerHTML = providerIds.map((providerId) => `
        <span class="graph-legend-item">
          <span class="graph-legend-swatch" style="background: ${graphProviderColor(providerId)};"></span>
          <span>${escapeHtml(graphProviderName(providerId))}</span>
        </span>
      `).join("");
    }

    const width = 860;
    const height = 270;
    const padLeft = 48;
    const padRight = 14;
    const padTop = 14;
    const padBottom = 26;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    const isLineMetric = state.graphMetric === "cache" || state.graphMetric === "latency";
    let maxValue = 0;
    buckets.forEach((bucket) => {
      if (isLineMetric) {
        const value = bucketMetricTotal(bucket);
        if (value !== null && value > maxValue) maxValue = value;
      } else {
        const total = providerIds.reduce((sum, providerId) => sum + (bucketMetricValue(bucket, providerId) || 0), 0);
        if (total > maxValue) maxValue = total;
      }
    });
    if (maxValue <= 0) maxValue = 1;

    const svgNs = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNs, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.setAttribute("class", "graph-svg");

    const add = (tag, attrs, text) => {
      const node = document.createElementNS(svgNs, tag);
      Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
      if (text !== undefined) node.textContent = text;
      svg.appendChild(node);
      return node;
    };

    // Gridlines + y labels
    const gridSteps = 4;
    for (let step = 0; step <= gridSteps; step += 1) {
      const y = padTop + plotH - (plotH * step) / gridSteps;
      add("line", {
        x1: padLeft, y1: y, x2: width - padRight, y2: y,
        stroke: "rgba(255,255,255,0.07)", "stroke-width": 1,
      });
      const value = (maxValue * step) / gridSteps;
      const label = state.graphMetric === "cache"
        ? `${Math.round(value)}%`
        : state.graphMetric === "latency"
          ? `${Math.round(value)}ms`
          : formatCompactNumber(value);
      add("text", {
        x: padLeft - 8, y: y + 3.5, "text-anchor": "end",
        fill: "#71717a", "font-size": "10", "font-family": "var(--font-mono)",
      }, label);
    }

    const slotW = plotW / buckets.length;
    const barGap = Math.min(3, slotW * 0.18);
    const barW = Math.max(1, slotW - barGap);

    if (!isLineMetric) {
      buckets.forEach((bucket, index) => {
        const x = padLeft + index * slotW + barGap / 2;
        let stackY = padTop + plotH;
        providerIds.forEach((providerId) => {
          const value = bucketMetricValue(bucket, providerId) || 0;
          if (value <= 0) return;
          const h = Math.max(0.4, (value / maxValue) * plotH);
          stackY -= h;
          add("rect", {
            x, y: stackY, width: barW, height: h,
            fill: graphProviderColor(providerId),
            opacity: 0.88,
            rx: Math.min(2, barW / 3),
          });
        });
      });
    } else {
      providerIds.forEach((providerId) => {
        const points = [];
        buckets.forEach((bucket, index) => {
          const value = bucketMetricValue(bucket, providerId);
          if (value === null) return;
          const x = padLeft + index * slotW + slotW / 2;
          const y = padTop + plotH - (value / maxValue) * plotH;
          points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
        });
        if (points.length === 0) return;
        add("polyline", {
          points: points.join(" "),
          fill: "none",
          stroke: graphProviderColor(providerId),
          "stroke-width": 2,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        });
        buckets.forEach((bucket, index) => {
          const value = bucketMetricValue(bucket, providerId);
          if (value === null) return;
          const x = padLeft + index * slotW + slotW / 2;
          const y = padTop + plotH - (value / maxValue) * plotH;
          add("circle", {
            cx: x, cy: y, r: 2.4,
            fill: graphProviderColor(providerId),
            stroke: "#151519", "stroke-width": 1,
          });
        });
      });
    }

    // X labels (max ~7)
    const labelEvery = Math.max(1, Math.ceil(buckets.length / 7));
    buckets.forEach((bucket, index) => {
      if (index % labelEvery !== 0 && index !== buckets.length - 1) return;
      const x = padLeft + index * slotW + slotW / 2;
      const date = new Date(bucket.start);
      const label = state.graphRange === "7d" || state.graphRange === "all"
        ? `${date.getMonth() + 1}/${date.getDate()} ${String(date.getHours()).padStart(2, "0")}:00`
        : `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
      add("text", {
        x, y: height - 8, "text-anchor": "middle",
        fill: "#71717a", "font-size": "10", "font-family": "var(--font-mono)",
      }, label);
    });

    // Hover columns
    const tooltip = document.createElement("div");
    tooltip.className = "graph-tooltip";
    tooltip.hidden = true;

    buckets.forEach((bucket, index) => {
      const hover = document.createElementNS(svgNs, "rect");
      hover.setAttribute("x", padLeft + index * slotW);
      hover.setAttribute("y", padTop);
      hover.setAttribute("width", slotW);
      hover.setAttribute("height", plotH);
      hover.setAttribute("fill", "transparent");
      hover.style.cursor = "crosshair";
      hover.addEventListener("mousemove", (event) => {
        const stageRect = container.getBoundingClientRect();
        const rows = providerIds
          .map((providerId) => ({ providerId, value: bucketMetricValue(bucket, providerId) }))
          .filter((row) => row.value !== null && row.value !== undefined && row.value !== 0)
          .sort((a, b) => (b.value || 0) - (a.value || 0));
        const timeLabel = new Date(bucket.start).toLocaleString();
        tooltip.innerHTML = `
          <div class="graph-tooltip-title">${escapeHtml(timeLabel)}</div>
          ${rows.length === 0 ? `<div class="graph-tooltip-row"><span style="color: var(--ink-muted);">No activity</span></div>` : rows.map((row) => `
            <div class="graph-tooltip-row">
              <span style="display: inline-flex; align-items: center; gap: 6px; min-width: 0;">
                <span class="graph-legend-swatch" style="background: ${graphProviderColor(row.providerId)};"></span>
                <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(graphProviderName(row.providerId))}</span>
              </span>
              <strong>${escapeHtml(metricValueLabel(row.value))}</strong>
            </div>
          `).join("")}
          <div class="graph-tooltip-row" style="border-top: 1px solid var(--border-subtle); margin-top: 4px; padding-top: 4px;">
            <span style="color: var(--ink-muted);">Total</span>
            <strong>${escapeHtml(metricValueLabel(bucketMetricTotal(bucket)))}</strong>
          </div>
        `;
        tooltip.hidden = false;
        const left = Math.min(stageRect.width - 200, Math.max(6, event.clientX - stageRect.left + 14));
        const top = Math.max(6, event.clientY - stageRect.top - 12);
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
      });
      hover.addEventListener("mouseleave", () => {
        tooltip.hidden = true;
      });
      svg.appendChild(hover);
    });

    container.appendChild(svg);
    container.appendChild(tooltip);
  }

  function renderGraphPoolCapacity() {
    if (!el.graphPoolCapacityContainer) return;
    const container = el.graphPoolCapacityContainer;
    container.innerHTML = "";

    const accountsToDisplay = currentStatusAccounts();
    const { pools } = buildUsagePools(accountsToDisplay, getQuotaDataMap());
    if (el.graphPoolCountBadge) {
      el.graphPoolCountBadge.textContent = `${pools.length} pool${pools.length === 1 ? "" : "s"}`;
    }

    const visiblePools = pools.filter((pool) => pool.poolRemaining !== null);
    if (visiblePools.length === 0) {
      container.innerHTML = `<div class="graph-empty-state" style="min-height: 120px;">No live quota windows reported.</div>`;
      return;
    }

    visiblePools.forEach((pool) => {
      const pct = Math.round(pool.poolRemaining * 100);
      const row = document.createElement("div");
      row.className = "graph-hbar-row";
      row.innerHTML = `
        <div class="graph-hbar-head">
          <span class="graph-hbar-label">
            <span class="pool-tier-badge ${pool.tierPro ? "is-pro" : ""}" style="font-size: 10px;">${escapeHtml(pool.tierLabel)}</span>
            <span class="truncate">${escapeHtml(pool.providerName)}</span>
            <span class="pool-count-pill">×${pool.accountTotal}</span>
          </span>
          <span class="graph-hbar-value" style="color: ${quotaBarColor(pct)};">${pct}% left</span>
        </div>
        <div class="graph-hbar-track">
          <div class="graph-hbar-fill" style="width: ${pct}%; background: ${quotaBarColor(pct)};"></div>
        </div>
      `;
      container.appendChild(row);
    });
  }

  function renderGraphProviderBreakdown(logs) {
    if (!el.graphProviderBreakdownContainer) return;
    const container = el.graphProviderBreakdownContainer;
    container.innerHTML = "";
    if (el.graphBreakdownBadge) {
      el.graphBreakdownBadge.textContent = "Requests & Tokens";
    }

    if (logs.length === 0) {
      container.innerHTML = `<div class="graph-empty-state" style="min-height: 120px;">No request activity in this range.</div>`;
      return;
    }

    const byProvider = new Map();
    logs.forEach((log) => {
      const providerId = log.plugin_id || "load_balancer";
      const entry = byProvider.get(providerId) || { requests: 0, tokens: 0 };
      entry.requests += 1;
      entry.tokens += Number(log.prompt_tokens || 0) + Number(log.completion_tokens || 0);
      byProvider.set(providerId, entry);
    });

    const rows = Array.from(byProvider.entries()).sort((a, b) => b[1].tokens - a[1].tokens);
    const maxTokens = Math.max(1, ...rows.map(([, entry]) => entry.tokens));

    rows.forEach(([providerId, entry]) => {
      const share = Math.round((entry.tokens / maxTokens) * 100);
      const row = document.createElement("div");
      row.className = "graph-hbar-row";
      row.innerHTML = `
        <div class="graph-hbar-head">
          <span class="graph-hbar-label">
            <span class="graph-legend-swatch" style="background: ${graphProviderColor(providerId)};"></span>
            <span class="truncate">${escapeHtml(graphProviderName(providerId))}</span>
          </span>
          <span class="graph-hbar-value">${entry.requests} req · ${formatCompactNumber(entry.tokens)} tok</span>
        </div>
        <div class="graph-hbar-track">
          <div class="graph-hbar-fill" style="width: ${share}%; background: ${graphProviderColor(providerId)};"></div>
        </div>
      `;
      container.appendChild(row);
    });
  }

  function renderGraphView() {
    if (!el.statusViewGraph) return;
    renderGraphProviderFilters();
    const logs = graphLogs();
    renderGraphKpis(logs);

    if (el.graphChartTitle) el.graphChartTitle.textContent = `${metricLabel()} Over Time`;
    if (el.graphChartSubtitle) {
      const count = state.graphProviders.size;
      el.graphChartSubtitle.textContent = count === 0
        ? "All Providers"
        : count === 1
          ? graphProviderName(Array.from(state.graphProviders)[0])
          : `${count} providers`;
    }

    renderGraphTimeseries(logs);
    renderGraphPoolCapacity();
    renderGraphProviderBreakdown(logs);

    if (el.graphMetricTabs) {
      el.graphMetricTabs.querySelectorAll(".graph-seg-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.metric === state.graphMetric);
      });
    }
    if (el.graphRangeTabs) {
      el.graphRangeTabs.querySelectorAll(".graph-seg-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.range === state.graphRange);
      });
    }
  }

  function setStatusView(view) {
    state.statusView = view === "graph" ? "graph" : "overview";
    if (el.statusViewOverview) {
      el.statusViewOverview.hidden = state.statusView !== "overview";
      el.statusViewOverview.classList.toggle("active", state.statusView === "overview");
    }
    if (el.statusViewGraph) {
      el.statusViewGraph.hidden = state.statusView !== "graph";
      el.statusViewGraph.classList.toggle("active", state.statusView === "graph");
    }
    if (el.statusViewBtns) {
      el.statusViewBtns.forEach((btn) => {
        const active = btn.dataset.statusView === state.statusView;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
      });
    }
    if (state.statusView === "graph") {
      renderGraphView();
    }
  }

  if (el.statusViewBtns) {
    el.statusViewBtns.forEach((btn) => {
      btn.addEventListener("click", () => setStatusView(btn.dataset.statusView));
    });
  }

  if (el.graphMetricTabs) {
    el.graphMetricTabs.addEventListener("click", (e) => {
      const btn = e.target.closest(".graph-seg-btn");
      if (!btn) return;
      state.graphMetric = btn.dataset.metric || "tokens";
      renderGraphView();
    });
  }

  if (el.graphRangeTabs) {
    el.graphRangeTabs.addEventListener("click", (e) => {
      const btn = e.target.closest(".graph-seg-btn");
      if (!btn) return;
      state.graphRange = btn.dataset.range || "all";
      renderGraphView();
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
        }, 3000);
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
  // PROVIDER IDENTITY EDITING (ID, NAME, SOURCE FILENAME)
  // =========================================================================

  function openProviderIdentityModal(plugin) {
    const idInput = document.getElementById("identity-provider-id");
    const nameInput = document.getElementById("identity-provider-name");
    const fnInput = document.getElementById("identity-provider-filename");
    const feedback = document.getElementById("identity-provider-feedback");

    if (feedback) {
      feedback.style.display = "none";
      feedback.className = "feedback-banner";
      feedback.textContent = "";
    }
    if (idInput) idInput.value = plugin.id || "";
    if (nameInput) nameInput.value = plugin.name || "";
    if (fnInput) fnInput.value = plugin.filename || `${plugin.id}.py`;

    openModal(document.getElementById("modal-provider-identity"));
    if (idInput) idInput.focus();
  }

  const btnSaveProviderIdentity = document.getElementById("btn-save-provider-identity");
  if (btnSaveProviderIdentity) {
    btnSaveProviderIdentity.addEventListener("click", async () => {
      const pluginId = state.currentFullscreenPluginId;
      if (!pluginId) return;

      const idInput = document.getElementById("identity-provider-id");
      const nameInput = document.getElementById("identity-provider-name");
      const fnInput = document.getElementById("identity-provider-filename");
      const feedback = document.getElementById("identity-provider-feedback");

      const newId = (idInput ? idInput.value : "").trim();
      const newName = (nameInput ? nameInput.value : "").trim();
      const newFilename = (fnInput ? fnInput.value : "").trim();

      const showError = (message) => {
        if (!feedback) {
          showToast(message, "error");
          return;
        }
        feedback.style.display = "block";
        feedback.className = "feedback-banner error";
        feedback.textContent = message;
      };

      if (!newId) return showError("Provider ID is required");
      if (!newName) return showError("Display name is required");

      btnSaveProviderIdentity.disabled = true;
      try {
        const res = await api(`/api/plugins/${pluginId}/identity`, {
          method: "PUT",
          body: JSON.stringify({ id: newId, name: newName, filename: newFilename }),
        });

        const updated = res.plugin || {};
        const finalId = updated.id || newId;
        state.currentFullscreenPluginId = finalId;

        const titleEl = document.getElementById("fs-provider-title");
        const idBadge = document.getElementById("fs-provider-id-badge");
        const fnBadge = document.getElementById("fs-provider-filename-badge");
        if (titleEl) titleEl.textContent = `Configure ${updated.name || newName}`;
        if (idBadge) idBadge.textContent = finalId;
        if (fnBadge) fnBadge.textContent = updated.filename || `${finalId}.py`;

        closeModal(document.getElementById("modal-provider-identity"));
        showToast("Provider identity updated");
        await fetchPlugins();
        // The provider ID changed, so re-render the sub-tabs with the refreshed
        // object and reload the editor from the (possibly renamed) source file.
        const updatedPlugin = state.plugins.find((p) => p.id === finalId);
        if (updatedPlugin) {
          renderFullscreenAccounts(finalId);
          renderFullscreenConfig(updatedPlugin);
          renderFullscreenModels(updatedPlugin);
        }
        await loadFullscreenPluginCode(finalId);
      } catch (err) {
        showError(`Update failed: ${err.message}`);
      } finally {
        btnSaveProviderIdentity.disabled = false;
      }
    });
  }

  // =========================================================================
  // Google Antigravity OAuth & Account Helpers
  // =========================================================================


  let currentPluginOAuthPollInterval = null;
  let currentPluginOAuth = null;

  function stopPluginOAuthPolling() {
    if (currentPluginOAuthPollInterval) {
      clearInterval(currentPluginOAuthPollInterval);
      currentPluginOAuthPollInterval = null;
    }
  }

  function resetPluginOAuthUI() {
    stopPluginOAuthPolling();
    currentPluginOAuth = null;
    const startBtn = document.getElementById("btn-start-plugin-oauth");
    const label = document.getElementById("plugin-oauth-btn-label");
    const flowBox = document.getElementById("plugin-oauth-flow-box");
    const statusText = document.getElementById("plugin-oauth-status-text");
    const authLink = document.getElementById("btn-open-plugin-oauth");
    const codeRow = document.getElementById("plugin-oauth-code-row");
    const codeEl = document.getElementById("plugin-oauth-user-code");
    const pasteRow = document.getElementById("plugin-oauth-paste-row");
    const pasteInput = document.getElementById("plugin-oauth-callback-url");
    if (startBtn) startBtn.disabled = false;
    if (label) label.textContent = "Sign in";
    if (flowBox) flowBox.style.display = "none";
    if (statusText) statusText.textContent = "Waiting for authorization...";
    if (authLink) authLink.href = "#";
    if (codeRow) codeRow.style.display = "none";
    if (codeEl) codeEl.textContent = "";
    if (pasteRow) pasteRow.style.display = "none";
    if (pasteInput) pasteInput.value = "";
  }

  function updateAccountModalVisibility() {
    const provSelect = document.getElementById("account-provider-select");
    const selectedProvider = provSelect ? provSelect.value : "";
    const selectedPlugin = (state.plugins || []).find((plugin) => plugin.id === selectedProvider);
    const isOAuth = Boolean(selectedPlugin && selectedPlugin.oauth_login);

    const oauthSection = document.getElementById("plugin-oauth-section");
    const oauthTitle = document.getElementById("plugin-oauth-title");
    const oauthHint = document.getElementById("plugin-oauth-hint");
    const importBtn = document.getElementById("btn-import-local-plugin-oauth");
    const keyGroup = document.getElementById("account-key-group");
    const usageGroup = document.getElementById("account-usage-tracking-group");

    if (oauthSection) oauthSection.style.display = isOAuth ? "block" : "none";
    if (oauthTitle) oauthTitle.textContent = selectedPlugin ? `${selectedPlugin.name} Connection` : "OAuth Connection";
    if (oauthHint) oauthHint.textContent = (selectedPlugin && selectedPlugin.oauth_hint) || "OAuth sign-in with auto-refreshing tokens";
    if (importBtn) {
      importBtn.style.display = isOAuth && selectedPlugin && selectedPlugin.oauth_local_import ? "inline-flex" : "none";
    }
    if (keyGroup) keyGroup.style.display = isOAuth ? "none" : "block";

    const tracksUsage = Boolean(selectedPlugin && selectedPlugin.supports_usage_meter);
    if (usageGroup) usageGroup.style.display = tracksUsage ? "flex" : "none";
  }

  const accountProvSelect = document.getElementById("account-provider-select");
  if (accountProvSelect) {
    accountProvSelect.addEventListener("change", () => {
      resetPluginOAuthUI();
      updateAccountModalVisibility();
    });
  }

  const btnStartPluginOAuth = document.getElementById("btn-start-plugin-oauth");
  if (btnStartPluginOAuth) {
    btnStartPluginOAuth.addEventListener("click", async () => {
      const provSelect = document.getElementById("account-provider-select");
      const pluginId = provSelect ? provSelect.value : "";
      const plugin = (state.plugins || []).find((item) => item.id === pluginId);
      const flowBox = document.getElementById("plugin-oauth-flow-box");
      const statusText = document.getElementById("plugin-oauth-status-text");
      const authLink = document.getElementById("btn-open-plugin-oauth");
      const codeRow = document.getElementById("plugin-oauth-code-row");
      const codeEl = document.getElementById("plugin-oauth-user-code");
      const pasteRow = document.getElementById("plugin-oauth-paste-row");
      const label = document.getElementById("plugin-oauth-btn-label");
      if (!pluginId) return;

      stopPluginOAuthPolling();
      resetPluginOAuthUI();
      btnStartPluginOAuth.disabled = true;
      if (label) label.textContent = "Starting...";
      if (statusText) statusText.textContent = "Requesting a sign-in flow...";
      try {
        const flow = await api(`/api/plugins/${pluginId}/oauth/start`, { method: "POST" });
        currentPluginOAuth = { pluginId, flowId: flow.flow_id, mode: flow.mode || "loopback" };
        const url = flow.auth_url || flow.verification_uri_complete || "#";
        if (authLink) authLink.href = url;
        if (flowBox) flowBox.style.display = "block";
        if (statusText) statusText.textContent = flow.instructions || `Waiting for ${plugin ? plugin.name : "provider"} authorization...`;
        if (flow.user_code) {
          if (codeRow) codeRow.style.display = "flex";
          if (codeEl) codeEl.textContent = flow.user_code;
        }
        if (flow.supports_callback_paste || flow.mode === "loopback") {
          if (pasteRow) pasteRow.style.display = "block";
        }
        if (url && url !== "#") {
          window.open(url, "_blank", "noopener");
        }

        currentPluginOAuthPollInterval = setInterval(async () => {
          try {
            const check = await api(`/api/plugins/${pluginId}/oauth/status/${flow.flow_id}`);
            if (check.status === "pending") {
              if (statusText) statusText.textContent = flow.user_code
                ? `Waiting for approval — code ${flow.user_code}`
                : `Waiting for ${plugin ? plugin.name : "provider"} authorization...`;
              return;
            }
            if (check.status === "complete") {
              stopPluginOAuthPolling();
              if (statusText) statusText.textContent = "Connected.";
              showToast(`Connected ${plugin ? plugin.name : "provider"} account '${(check.account && check.account.name) || ""}'!`);
              closeModal(el.modalAccount);
              await fetchAccounts();
              if (state.currentFullscreenPluginId) {
                renderFullscreenAccounts(state.currentFullscreenPluginId);
              }
              return;
            }
            stopPluginOAuthPolling();
            const message = check.error || "OAuth sign-in failed.";
            if (statusText) statusText.textContent = message;
            showToast(message, "error");
            btnStartPluginOAuth.disabled = false;
            if (label) label.textContent = "Sign in";
          } catch (err) {
            stopPluginOAuthPolling();
            if (statusText) statusText.textContent = `Sign-in status check failed: ${err.message}`;
            btnStartPluginOAuth.disabled = false;
            if (label) label.textContent = "Sign in";
          }
        }, 5000);
      } catch (err) {
        if (statusText) statusText.textContent = `Could not start sign-in: ${err.message}`;
        btnStartPluginOAuth.disabled = false;
        if (label) label.textContent = "Sign in";
      }
    });
  }

  const btnImportLocalPluginOAuth = document.getElementById("btn-import-local-plugin-oauth");
  if (btnImportLocalPluginOAuth) {
    btnImportLocalPluginOAuth.addEventListener("click", async () => {
      const provSelect = document.getElementById("account-provider-select");
      const pluginId = provSelect ? provSelect.value : "";
      const plugin = (state.plugins || []).find((item) => item.id === pluginId);
      if (!pluginId) return;
      btnImportLocalPluginOAuth.disabled = true;
      try {
        const res = await api(`/api/plugins/${pluginId}/oauth/import-local`, { method: "POST" });
        showToast(`Imported ${plugin ? plugin.name : "provider"} account '${(res.account && res.account.name) || ""}'!`);
        closeModal(el.modalAccount);
        await fetchAccounts();
        if (state.currentFullscreenPluginId) {
          renderFullscreenAccounts(state.currentFullscreenPluginId);
        }
      } catch (err) {
        showToast(`Import failed: ${err.message}`, "error");
      } finally {
        btnImportLocalPluginOAuth.disabled = false;
      }
    });
  }

  const btnCopyPluginOAuthUrl = document.getElementById("btn-copy-plugin-oauth-url");
  if (btnCopyPluginOAuthUrl) {
    btnCopyPluginOAuthUrl.addEventListener("click", async () => {
      const authLink = document.getElementById("btn-open-plugin-oauth");
      if (!authLink || !authLink.href || authLink.href === "#") return;
      try {
        await navigator.clipboard.writeText(authLink.href);
        showToast("Sign-in link copied");
      } catch (_) {
        showToast("Could not copy the link", "error");
      }
    });
  }

  const btnSubmitPluginOAuthCallback = document.getElementById("btn-submit-plugin-oauth-callback");
  if (btnSubmitPluginOAuthCallback) {
    btnSubmitPluginOAuthCallback.addEventListener("click", async () => {
      const pasteInput = document.getElementById("plugin-oauth-callback-url");
      const value = pasteInput ? pasteInput.value.trim() : "";
      const flow = currentPluginOAuth;
      if (!flow) {
        showToast("Start the sign-in first", "error");
        return;
      }
      if (!value) {
        showToast("Paste the callback URL or code first", "error");
        return;
      }
      btnSubmitPluginOAuthCallback.disabled = true;
      btnSubmitPluginOAuthCallback.textContent = "Connecting...";
      try {
        const res = await api(`/api/plugins/${flow.pluginId}/oauth/callback`, {
          method: "POST",
          body: JSON.stringify({ flow_id: flow.flowId, callback_url: value }),
        });
        if (res.status === "complete" && res.account) {
          stopPluginOAuthPolling();
          showToast(`Connected account '${res.account.name}'!`);
          closeModal(el.modalAccount);
          await fetchAccounts();
          if (state.currentFullscreenPluginId) {
            renderFullscreenAccounts(state.currentFullscreenPluginId);
          }
        } else {
          showToast(res.error || "OAuth completion failed", "error");
        }
      } catch (err) {
        showToast(`Connection failed: ${err.message}`, "error");
      } finally {
        btnSubmitPluginOAuthCallback.disabled = false;
        btnSubmitPluginOAuthCallback.textContent = "Connect";
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
    const usageTrackingInput = document.getElementById("account-usage-tracking-toggle");
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

    resetPluginOAuthUI();

    if (account) {
      title.textContent = "Edit Provider Account";
      idInput.value = account.id;
      nameInput.value = account.name;
      keyInput.value = "";
      keyInput.placeholder = account.api_key
        ? `Saved credential: ${account.api_key} — leave blank to keep`
        : "Leave blank to keep the current credential";
      keyInput.type = "password";
      baseInput.value = account.base_url || "";
      priorityInput.value = account.priority || 1;
      weightInput.value = account.weight || 1;
      enabledInput.checked = !!account.enabled;
      if (usageTrackingInput) usageTrackingInput.checked = !account.disable_usage_tracking;
    } else {
      title.textContent = "Connect Provider Account";
      idInput.value = "";
      nameInput.value = "";
      keyInput.value = "";
      keyInput.placeholder = "Enter API key or bearer token";
      keyInput.type = "password";
      baseInput.value = "";
      const initialProvId = defaultProviderId || (provSelect.value || (state.plugins[0] ? state.plugins[0].id : ""));
      priorityInput.value = getNextAvailablePriority(initialProvId);
      weightInput.value = 1;
      enabledInput.checked = true;
      if (usageTrackingInput) usageTrackingInput.checked = true;
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
      const providerSelect = document.getElementById("account-provider-select");
      const formId = document.getElementById("account-form-id");
      const oauthPlugin = providerSelect
        ? (state.plugins || []).find((plugin) => plugin.id === providerSelect.value)
        : null;
      if (oauthPlugin && oauthPlugin.oauth_login && !(formId && formId.value)) {
        showToast("Use the OAuth sign-in button to connect this provider.", "warning");
        return;
      }
      const idInput = document.getElementById("account-form-id");
      const provSelect = document.getElementById("account-provider-select");
      const nameInput = document.getElementById("account-name");
      const keyInput = document.getElementById("account-api-key");
      const baseInput = document.getElementById("account-base-url");
      const priorityInput = document.getElementById("account-priority");
      const weightInput = document.getElementById("account-weight");
      const enabledInput = document.getElementById("account-enabled-toggle");
      const usageTrackingInput = document.getElementById("account-usage-tracking-toggle");

      const name = nameInput.value.trim();
      const provId = provSelect.value;
      const isEditing = Boolean(idInput.value);
      const selectedPlugin = (state.plugins || []).find((plugin) => plugin.id === provId);
      const tracksUsage = Boolean(selectedPlugin && selectedPlugin.supports_usage_meter);
      const disableUsageTracking = tracksUsage
        ? !(usageTrackingInput && usageTrackingInput.checked)
        : undefined;

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
          if (keyInput.value) {
            updatePayload.api_key = keyInput.value;
          }
          if (disableUsageTracking !== undefined) {
            updatePayload.disable_usage_tracking = disableUsageTracking;
          }

          await api(`/api/accounts/${idInput.value}`, {
            method: "PUT",
            body: JSON.stringify(updatePayload),
          });
          showToast(`Account '${name}' updated`);
        } else {
          // Create
          const createPayload = {
            provider_id: provId,
            name,
            api_key: keyInput.value,
            base_url: baseInput.value || undefined,
            priority: chosenPriority,
            weight: parseInt(weightInput.value, 10) || 1,
            enabled: enabledInput.checked,
          };
          if (disableUsageTracking !== undefined) {
            createPayload.disable_usage_tracking = disableUsageTracking;
          }
          await api("/api/accounts", {
            method: "POST",
            body: JSON.stringify(createPayload),
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
    btnToggleKey.addEventListener("click", async () => {
      const keyInput = document.getElementById("account-api-key");
      const idInput = document.getElementById("account-form-id");
      if (!keyInput) return;

      // Editing starts with a blank field so the secret never arrives with
      // the list response. Reveal it on demand instead.
      if (idInput && idInput.value && !keyInput.value) {
        try {
          const revealed = await api(`/api/accounts/${idInput.value}/reveal`);
          keyInput.value = revealed.api_key || "";
        } catch (err) {
          showToast(`Failed to reveal credential: ${err.message}`, "error");
          return;
        }
      }

      const isPass = keyInput.type === "password";
      keyInput.type = isPass ? "text" : "password";
      btnToggleKey.textContent = isPass ? "Hide" : "Show";
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
      // Close the topmost open modal first. Nested dialogs (add/edit account and
      // model) are rendered after the provider fullscreen modal in the DOM, so
      // taking the last active overlay keeps ESC from closing the provider
      // configuration behind an open account/model dialog.
      const activeModals = Array.from(document.querySelectorAll(".modal-overlay.active"));
      if (activeModals.length > 0) {
        const topmost = activeModals[activeModals.length - 1];
        if (topmost === el.modalProviderFullscreen) {
          requestCloseFullscreen();
        } else {
          closeModal(topmost);
        }
        return;
      }
    }
  });

  // Admin login gate
  async function handleAuthSubmit() {
    const candidate = (el.authTokenInput ? el.authTokenInput.value : "").trim();
    if (!candidate) {
      showAuthGate("Enter the admin token to continue.");
      return;
    }
    if (el.authSubmitBtn) el.authSubmitBtn.disabled = true;
    const valid = await verifyAdminToken(candidate);
    if (el.authSubmitBtn) el.authSubmitBtn.disabled = false;
    if (!valid) {
      showAuthGate("That admin token is not valid.");
      return;
    }
    setAdminToken(candidate);
    hideAuthGate();
    await loadDashboard();
  }

  if (el.authSubmitBtn) {
    el.authSubmitBtn.addEventListener("click", handleAuthSubmit);
  }
  if (el.authTokenInput) {
    el.authTokenInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleAuthSubmit();
    });
  }
  if (el.btnAdminLock) {
    el.btnAdminLock.addEventListener("click", () => {
      setAdminToken("");
      window.location.reload();
    });
  }

  async function loadDashboard() {
    enhanceAllSelects();
    await fetchStats();
    await loadTemplates();
    await fetchPlugins();
    await fetchAccounts();
    await fetchApiKeys();
    await fetchTemporaryApiKey();
    await fetchRules();
    await fetchManagementInfo();
    populateSnippetModels();
    updateSnippet();
  }

  // Global initial boot
  async function init() {
    syncResponsiveLayout();
    if (!adminToken) {
      showAuthGate();
      return;
    }
    const valid = await verifyAdminToken(adminToken);
    if (!valid) {
      setAdminToken("");
      showAuthGate("Enter the admin token to unlock the dashboard.");
      return;
    }
    hideAuthGate();
    await loadDashboard();
  }

  window.addEventListener("DOMContentLoaded", init);
})();
