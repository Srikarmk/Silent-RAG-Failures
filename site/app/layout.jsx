import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], display: "swap" });

export const metadata = {
  title: "Silent Failures: detecting RAG retrieval damage without labels",
  description:
    "A DS5500 benchmark that breaks RAG retrieval on purpose and tests which label-free signals notice.",
};

export const viewport = { themeColor: "#0b0e14" };

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.className}>
      <body>{children}</body>
    </html>
  );
}
