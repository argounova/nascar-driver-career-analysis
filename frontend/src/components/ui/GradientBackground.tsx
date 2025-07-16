// src/components/ui/GradientBackground.tsx
'use client';

import React from 'react';
import { Box } from '@mui/material';
import { nascarColors } from '@/lib/theme';

interface GradientBackgroundProps {
  children: React.ReactNode;
}

export default function GradientBackground({ children }: GradientBackgroundProps) {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
        background: `
          linear-gradient(
            135deg,
            ${nascarColors.primary}40 0%,
            ${nascarColors.archetypes[1]}30 25%,
            ${nascarColors.archetypes[2]}20 50%,
            ${nascarColors.archetypes[4]}15 75%,
            #0a0a0a 100%
          )
        `,
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            radial-gradient(
              ellipse at top left,
              ${nascarColors.primary}20 0%,
              transparent 50%
            ),
            radial-gradient(
              ellipse at bottom right,
              ${nascarColors.archetypes[1]}15 0%,
              transparent 50%
            ),
            linear-gradient(
              45deg,
              transparent 30%,
              ${nascarColors.archetypes[2]}08 50%,
              transparent 70%
            )
          `,
          pointerEvents: 'none',
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            repeating-linear-gradient(
              45deg,
              transparent,
              transparent 100px,
              ${nascarColors.primary}05 100px,
              ${nascarColors.primary}05 102px
            )
          `,
          pointerEvents: 'none',
        },
      }}
    >
      <Box sx={{ position: 'relative', zIndex: 1 }}>
        {children}
      </Box>
    </Box>
  );
}