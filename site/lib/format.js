export const pct = (s) => `${Math.round(s * 100)}%`;
export const f3 = (x) => x.toFixed(3);
export const asset = (p) => `${process.env.NEXT_PUBLIC_BASE_PATH || ""}${p}`;
