import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Privacy Policy Clause Classifier",
  description:
    "CSE440 project — classifies a privacy-policy clause into one of 8 data-practice categories using a fine-tuned BERT Base model.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
