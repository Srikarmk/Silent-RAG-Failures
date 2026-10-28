// Static export for GitHub Pages. The deploy workflow sets NEXT_PUBLIC_BASE_PATH to /Silent-RAG-Failures.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

export default {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
};
