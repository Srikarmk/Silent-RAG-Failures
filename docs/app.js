(() => {
  const D = window.SRF;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const pct = (s) => `${Math.round(s * 100)}%`;
  const f3 = (x) => x.toFixed(3);
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* hero monitor */
  const slider = $('[data-sev]');
  const examples = D.examples.slice(0, 3);
  let sev = 0, ex = 0, touched = false, lastTitles = [];

  const tabs = $('[data-tabs]');
  examples.forEach((e, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.role = 'tab'; b.textContent = e.text;
    b.setAttribute('aria-selected', i === 0);
    b.addEventListener('click', () => { ex = i; render(); });
    tabs.append(b);
  });

  function render() {
    const L = D.levels[sev], base = D.levels[0];
    slider.value = sev;
    $('[data-sev-out]').textContent = pct(L.severity);
    $$('button', tabs).forEach((b, i) => b.setAttribute('aria-selected', i === ex));
    const res = examples[ex].levels[sev].results;
    $('[data-results]').innerHTML = res.map((r, i) => `
      <li class="${lastTitles.includes(r.title) ? '' : 'is-new'}">
        <span class="rank">${i + 1}</span>
        <span class="t" title="${esc(r.title)}">${esc(r.title)}</span>
        <span class="s">${r.score.toFixed(3)}</span>
        <span class="rel ${r.rel ? '' : 'rel--no'}">${r.rel ? 'relevant' : 'not relevant'}</span>
      </li>`).join('');
    lastTitles = res.map((r) => r.title);
    $('[data-ndcg]').textContent = f3(L.ndcg);
    const d = (L.ndcg - base.ndcg) / base.ndcg;
    $('[data-ndcg-delta]').innerHTML = sev ? `<span class="delta">−${Math.abs(Math.round(d * 100))}%</span>` : '';
    $('[data-top1]').textContent = f3(L.top1_mean);
    $('[data-reembed]').textContent = L.reembed_mismatch;
    const prev = D.levels[sev - 1], next = D.levels[sev + 1];
    $('[data-prev]').textContent = prev ? `Severity ${pct(prev.severity)}, nDCG@10 ${f3(prev.ndcg)}` : 'Clean index';
    $('[data-next]').textContent = next ? `Severity ${pct(next.severity)}, nDCG@10 ${f3(next.ndcg)}` : 'Every swap applied';
  }
  slider.addEventListener('input', () => { touched = true; sev = +slider.value; render(); });
  render();

  /* pinned hero: scrolling advances the reel while the sentence holds */
  const hero = $('.hero');
  const steps = [0, 3, 6, 9];
  const wide = matchMedia('(min-width: 761px)');
  addEventListener('scroll', () => {
    if (touched || !wide.matches) return;
    const r = hero.getBoundingClientRect();
    const p = Math.min(1, Math.max(0, -r.top / (r.height - innerHeight)));
    const s = steps[Math.min(steps.length - 1, Math.floor(p * steps.length))];
    if (s !== sev) { sev = s; render(); }
  }, { passive: true });

  /* detector bands */
  $$('[data-band]').forEach((band) => {
    const sw = $('.switch', band);
    const set = (on) => { band.toggleAttribute('data-on', on); sw.setAttribute('aria-checked', on); };
    sw.addEventListener('click', () => set(!band.hasAttribute('data-on')));
  });

  /* bars standing on the edge */
  const bars = $('[data-bars]');
  const max = Math.max(...D.levels.map((l) => l.ndcg));
  const accent = 1, printed = [0, 1, 9];
  bars.innerHTML = D.levels.map((l, i) => `
    <div class="bar-col ${i === accent ? 'is-accent' : ''} ${printed.includes(i) ? 'is-labelled' : ''}" tabindex="0" style="--w:${(l.ndcg / max * 100).toFixed(1)}%">
      ${printed.includes(i) ? `<span class="v">${f3(l.ndcg)}</span>` : ''}
      <i style="height:${(l.ndcg / max * 100).toFixed(1)}%"></i>
      <span class="tip">${pct(l.severity)} upgraded · ${f3(l.ndcg)}</span>
    </div>`).join('');

  /* type a query */
  const input = $('[data-q]');
  const answer = $('[data-answer]');
  const pane = (title, r) => `
    <div class="pane">
      <div class="pane__head"><span>${title}</span><span>nDCG@10 <b>${f3(r.ndcg)}</b></span></div>
      <ol>${r.r.map(([t, rel]) => `<li><span class="t" title="${esc(t)}">${esc(t)}</span><span class="rel ${rel ? '' : 'rel--no'}">${rel ? 'relevant' : 'not relevant'}</span></li>`).join('')}</ol>
    </div>`;
  function find(q) {
    q = q.trim().toLowerCase();
    if (!q) return null;
    return D.all.find((x) => x.q.toLowerCase() === q)
      || D.all.find((x) => x.q.toLowerCase().startsWith(q))
      || D.all.find((x) => x.q.toLowerCase().includes(q));
  }
  function answerFor() {
    const hit = find(input.value);
    drawSpread(hit);
    if (!hit) {
      answer.innerHTML = `<p class="try__match">No NFCorpus test query matches that. Try a health topic, like “vitamin d” or “sugar”.</p>`;
      return;
    }
    answer.innerHTML = `<p class="try__match">Top 3 for the NFCorpus test query “${esc(hit.q)}”</p>`
      + pane('Clean index', hit.clean) + pane('Half the index upgraded', hit.mixed);
  }

  /* every query's change at 50%, sorted; the typed query is highlighted */
  const spread = $('[data-spread]');
  const deltas = D.all.map((x) => ({ x, d: x.mixed.ndcg - x.clean.ndcg })).sort((a, b) => a.d - b.d);
  const worse = deltas.filter((e) => e.d < -1e-9).length, better = deltas.filter((e) => e.d > 1e-9).length;
  $('[data-counts]').innerHTML = `<b>${worse}</b> worse · <b>${deltas.length - worse - better}</b> unchanged · <b>${better}</b> better`;
  function drawSpread(hit) {
    const ctx = spread.getContext('2d'), W = spread.width, H = spread.height, mid = H / 2;
    const bw = W / deltas.length;
    const top = Math.max(...deltas.map((e) => Math.abs(e.d)));
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#2a3550'; ctx.fillRect(0, mid, W, 1);
    deltas.forEach((e, i) => {
      const h = e.d / top * (H / 2 - 8);
      ctx.fillStyle = e.x === hit ? '#e9edf3' : e.d < 0 ? '#5a9bfc' : e.d > 0 ? '#4fc28a' : '#2a3550';
      const w = e.x === hit ? Math.max(bw, 4) : Math.max(bw - 0.6, 1);
      ctx.fillRect(i * bw, h < 0 ? mid : mid - h, w, Math.max(Math.abs(h), e.x === hit ? 6 : 1));
    });
  }
  input.addEventListener('input', answerFor);
  const picks = D.all.filter((x) => x.clean.ndcg - x.mixed.ndcg > 0.3 && x.q.length < 22).slice(0, 3).map((x) => x.q);
  $('[data-examples]').innerHTML = '<span>Or try</span>' + picks.map((q) => `<button type="button">${esc(q)}</button>`).join('');
  $$('[data-examples] button').forEach((b) => b.addEventListener('click', () => { input.value = b.textContent; answerFor(); }));
  answerFor();

  /* index map: lines draw out from each doc's clean position when the map comes into view */
  const map = $('[data-map]');
  $('[data-map-cos]').textContent = D.map.cos.toFixed(2);
  function drawMap(t) {
    const ctx = map.getContext('2d'), W = map.width, H = map.height, pad = 28;
    const X = (v) => pad + v * (W - 2 * pad), Y = (v) => H - pad - v * (H - 2 * pad);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#3b4a6e';
    D.map.docs.forEach(([x, y]) => ctx.fillRect(X(x) - 1, Y(y) - 1, 2.2, 2.2));
    ctx.strokeStyle = 'rgba(90,155,252,.7)'; ctx.lineWidth = 1.1; ctx.fillStyle = '#5a9bfc';
    D.map.from.forEach(([x0, y0], i) => {
      const [x1, y1] = D.map.to[i];
      const x = X(x0 + (x1 - x0) * t), y = Y(y0 + (y1 - y0) * t);
      ctx.beginPath(); ctx.moveTo(X(x0), Y(y0)); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillRect(x - 2, y - 2, 4, 4);
    });
  }
  drawMap(reduced || !('IntersectionObserver' in window) ? 1 : 0);
  if (!reduced && 'IntersectionObserver' in window) {
    const mo = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      mo.disconnect();
      const t0 = performance.now();
      const step = (now) => { const t = Math.min(1, (now - t0) / 1600); drawMap(1 - Math.pow(1 - t, 3)); if (t < 1) requestAnimationFrame(step); };
      requestAnimationFrame(step);
    }, { threshold: 0.3 });
    mo.observe(map);
  }

  /* entrances: heading, then text, then fragment */
  const groups = [
    ['.how .field', ['.how__title', '.ticket', '.how__clause']],
    ['.failures__col', ['.failures__title', '.failures__intro', '.rows']],
    ['.detectors .band', ['.band__text', '.switch']],
    ['.map', ['.map__title', '.map__text p', '.map__fig']],
    ['.failures', ['.failures__side']],
    ['.drop', ['.drop__figure', '.drop__claim', '.drop__note', '.bar-col']],
    ['.try', ['.try__line', '.try__answer', '.try__examples', '.spread']],
    ['.status', ['.status__title', '.status__text p', '.baseline', '.milestones li']],
  ];
  if (reduced || !('IntersectionObserver' in window)) return;
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in'); io.unobserve(e.target);
  }), { rootMargin: '0px 0px -8% 0px' });
  groups.forEach(([root, parts]) => $$(root).forEach((r) => {
    let n = 0;
    parts.forEach((sel) => $$(sel, r).forEach((el) => {
      el.dataset.rise = ''; el.style.setProperty('--d', `${Math.min(n++, 10) * 70}ms`); io.observe(el);
    }));
  }));
})();
