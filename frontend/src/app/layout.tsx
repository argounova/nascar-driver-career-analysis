// src/app/layout.tsx
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import ThemeProvider from "@/components/providers/ThemeProvider";
import QueryProvider from "@/components/providers/QueryProvider";

/**
 * Root Layout Component
 * 
 * This is the root layout that wraps your entire NASCAR analytics application.
 * It provides the foundation layers that all pages and components will use:
 * 
 * 1. Font Configuration - Geist Sans & Mono fonts
 * 2. Theme Provider - MUI dark theme with NASCAR colors
 * 3. Query Provider - React Query for API state management
 * 
 * The provider order is important:
 * - ThemeProvider (outermost) - provides MUI theme context
 * - QueryProvider (inner) - provides React Query context
 * - Page content (innermost) - your actual app content
 */

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NASCAR Driver Analytics",
  description: "Advanced analysis and visualization of NASCAR driver career performance, archetypes, and predictions",
  keywords: ["NASCAR", "racing", "analytics", "data visualization", "driver analysis", "machine learning"],
  authors: [{ name: "NASCAR Analytics Team" }],
  
  // Open Graph meta tags for social sharing
  openGraph: {
    title: "NASCAR Driver Analytics",
    description: "Discover NASCAR driver career patterns and performance insights through advanced analytics",
    type: "website",
    siteName: "NASCAR Analytics",
  },
  
  // Additional meta tags
  robots: "index, follow",
};

// Viewport configuration (Next.js 15+ requirement)
export const viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#FF6B35", // NASCAR orange for browser theme
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {/* 
          Provider Stack:
          1. ThemeProvider - MUI theme with NASCAR colors and dark mode
          2. QueryProvider - React Query for API state management
        */}
        <ThemeProvider>
          <QueryProvider>
            {children}
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}