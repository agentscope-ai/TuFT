// Custom JavaScript for TuFT documentation (Sphinx Awesome Theme)

(function() {
  'use strict';

  function isZh() {
    return location.pathname.includes('/zh/');
  }

  function setupVersionSwitcher() {
    const headerBrand = document.querySelector('header .container > div.hidden') || document.querySelector('header .container');
    if (!headerBrand || headerBrand.querySelector('.tuft-version-select')) {
      return;
    }
    const brandLink = headerBrand.querySelector('a');
    if (!brandLink) {
      return;
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'tuft-version-select';
    wrapper.innerHTML = `
      <div class="tuft-version-select-inner">
        <span class="tuft-version-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 5.5h6.5a2 2 0 0 1 2 2v12.5H6.5a2 2 0 0 0-2 2V5.5z"/>
            <path d="M20 5.5h-6.5a2 2 0 0 0-2 2v12.5h6.5a2 2 0 0 1 2 2V5.5z"/>
          </svg>
        </span>
        <select id="tuft-version-select" aria-label="Version" title="Version">
          <option value="latest" selected>latest</option>
        </select>
      </div>
    `;

    brandLink.insertAdjacentElement('afterend', wrapper);
    const select = wrapper.querySelector('select');

    const currentPath = location.pathname;
    const currentVersionMatch = currentPath.match(/\/(en|zh)\/([^/]+)\//);
    const currentVersion = currentVersionMatch ? currentVersionMatch[2] : 'latest';

    const basePath = location.pathname.replace(/\/(en|zh)\/.*$/, '/');
    const switcherUrl = basePath.endsWith('/') ? `${basePath}switcher.json` : `${basePath}/switcher.json`;

    fetch(switcherUrl)
      .then((response) => response.json())
      .then((items) => {
        select.innerHTML = '';
        items.forEach((item) => {
          const option = document.createElement('option');
          option.value = item.url;
          option.textContent = item.name;
          select.appendChild(option);
        });

        const match = items.find((item) => item.version === currentVersion)
          || items.find((item) => item.preferred);
        if (match) {
          select.value = match.url;
        }
      })
      .catch(() => {
        select.innerHTML = '';
        const option = document.createElement('option');
        option.value = 'latest';
        option.textContent = 'latest';
        select.appendChild(option);
      });

    select.addEventListener('change', () => {
      let target = select.value;
      if (isZh()) {
        target = target.replace('/en/', '/zh/');
      }
      if (target) {
        location.href = target;
      }
    });
  }

  function setupLanguageSwitch() {
    const target = document.querySelector('header nav') || document.querySelector('header .container');
    if (!target || target.querySelector('.tuft-lang-switch')) {
      return;
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'tuft-lang-switch';
    button.innerHTML = '<span></span>';

    function updateLabel() {
      const span = button.querySelector('span');
      if (span) {
        span.textContent = isZh() ? 'EN' : '中';
      }
      button.title = isZh() ? 'Switch to English' : '切换到中文';
    }

    button.addEventListener('click', () => {
      const match = location.pathname.match(/^(.*?\/)(en|zh)(\/.*)$/);
      if (match) {
        const other = match[2] === 'en' ? 'zh' : 'en';
        const suffix = match[3] || '/';
        location.href = match[1] + other + suffix + location.search + location.hash;
        return;
      }
      location.href = isZh() ? '/en/latest/' : '/zh/latest/';
    });

    updateLabel();
    target.insertAdjacentElement('afterbegin', button);
  }

  function addControls() {
    const sidebar = document.getElementById('left-sidebar');
    if (!sidebar) {
      if (document.readyState !== 'complete') {
        setTimeout(addControls, 100);
      }
      return;
    }

    setupVersionSwitcher();
    setupLanguageSwitch();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addControls);
  } else {
    addControls();
  }
  window.addEventListener('load', addControls);
})();
