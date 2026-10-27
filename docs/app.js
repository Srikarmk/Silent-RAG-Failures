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
    if (!hit) {
      answer.innerHTML = `<p class="try__match">No NFCorpus test query matches that. Try a health topic, like “vitamin d” or “sugar”.</p>`;
      return;
    }
    answer.innerHTML = `<p class="try__match">Top 3 for the NFCorpus test query “${esc(hit.q)}”</p>`
      + pane('Clean index', hit.clean) + pane('Half the index upgraded', hit.mixed);
  }
  input.addEventListener('input', answerFor);
  const picks = D.all.filter((x) => x.clean.ndcg - x.mixed.ndcg > 0.3 && x.q.length < 22).slice(0, 3).map((x) => x.q);
  $('[data-examples]').innerHTML = '<span>Or try</span>' + picks.map((q) => `<button type="button">${esc(q)}</button>`).join('');
  $$('[data-examples] button').forEach((b) => b.addEventListener('click', () => { input.value = b.textContent; answerFor(); }));
  answerFor();

  /* Boston clock */
  const clock = $('[data-clock]');
  const tick = () => { clock.textContent = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' }).format(new Date()); };
  tick(); setInterval(tick, 30000);

  /* entrances: heading, then text, then fragment */
  const groups = [
    ['.how .field', ['.how__title', '.ticket', '.how__clause']],
    ['.failures__col', ['.failures__title', '.failures__intro', '.rows']],
    ['.detectors .band', ['.band__text', '.switch']],
    ['.plate', ['img', 'figcaption']],
    ['.failures', ['.failures__pic']],
    ['.drop', ['.drop__figure', '.drop__claim', '.drop__note', '.bar-col']],
    ['.try', ['.try__line', '.try__answer', '.try__examples', '.try__pic']],
    ['.letter__col', [':scope > *']],
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
