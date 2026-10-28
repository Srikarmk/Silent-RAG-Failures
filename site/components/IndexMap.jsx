"use client";

import { useEffect, useRef } from "react";
import { asset } from "@/lib/format";

function draw(canvas, map, t) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 28;
  const X = (v) => pad + v * (W - 2 * pad);
  const Y = (v) => H - pad - v * (H - 2 * pad);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#334058";
  map.docs.forEach(([x, y]) => ctx.fillRect(X(x) - 1, Y(y) - 1, 2, 2));
  ctx.strokeStyle = "rgba(110,168,254,.55)";
  ctx.fillStyle = "#6ea8fe";
  ctx.lineWidth = 1;
  map.from.forEach(([x0, y0], i) => {
    const [x1, y1] = map.to[i];
    const x = X(x0 + (x1 - x0) * t), y = Y(y0 + (y1 - y0) * t);
    ctx.beginPath(); ctx.moveTo(X(x0), Y(y0)); ctx.lineTo(x, y); ctx.stroke();
    ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
  });
}

// Every document on the clean index's first two principal components; lines grow from a
// sample of documents to where the second model places the same text.
export default function IndexMap() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    let raf, io, alive = true;
    fetch(asset("/data/map.json")).then((r) => r.json()).then((map) => {
      if (!alive) return;
      const still = matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window);
      draw(canvas, map, still ? 1 : 0);
      if (still) return;
      io = new IntersectionObserver(([e]) => {
        if (!e.isIntersecting) return;
        io.disconnect();
        const t0 = performance.now();
        const step = (now) => {
          const t = Math.min(1, (now - t0) / 1600);
          draw(canvas, map, 1 - Math.pow(1 - t, 3));
          if (t < 1) raf = requestAnimationFrame(step);
        };
        raf = requestAnimationFrame(step);
      }, { threshold: 0.3 });
      io.observe(canvas);
    });
    return () => { alive = false; io?.disconnect(); cancelAnimationFrame(raf); };
  }, []);

  return (
    <div className="figure" data-ui>
      <canvas
        ref={ref} width="1200" height="720" role="img"
        aria-label="Scatter of 3,633 document vectors with 260 lines showing how far each document moves when re-embedded by a second model"
      />
    </div>
  );
}
