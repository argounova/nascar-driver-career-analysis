// src/components/providers/ThemeProvider.tsx
'use client';

import React from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { nascarTheme } from '@/lib/theme';

interface ThemeProviderProps {
  children: React.ReactNode;
}

export default function ThemeProvider({ children }: ThemeProviderProps) {
  return (
    <MuiThemeProvider theme={nascarTheme}>
      {/* CssBaseline kickstarts elegant, consistent, and simple baseline to build upon */}
      <CssBaseline />
      {children}
    </MuiThemeProvider>
  );
}