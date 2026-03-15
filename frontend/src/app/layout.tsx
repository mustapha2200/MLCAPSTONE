import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "ImmobilierNK — Prédiction des Prix à Nouakchott",
  description: "Estimez le prix de votre bien immobilier à Nouakchott grâce au Machine Learning",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body className="antialiased min-h-screen bg-[#f8f9fa]">
        <Navbar />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <footer className="mt-16 border-t border-gray-200 bg-white py-6 text-center text-sm text-gray-500">
          Projet Capstone ML — Master 1 SupNum Mauritanie · 2026 ·{' '}
          <span className="text-mauritania-green font-medium">RMSLE = 0.6576</span>
        </footer>
      </body>
    </html>
  );
}
