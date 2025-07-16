// src/components/ui/BackgroundVariations.tsx
'use client';

import React from 'react';
import { Box } from '@mui/material';
import { nascarColors } from '@/lib/theme';

interface BackgroundProps {
  children: React.ReactNode;
}

// Original Gradient Background (for reference)
export function GradientBackground({ children }: BackgroundProps) {
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

// Variation 1: Darker with More Black
export function DarkGradientBackground({ children }: BackgroundProps) {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
        background: `
          linear-gradient(
            135deg,
            ${nascarColors.primary}25 0%,
            ${nascarColors.archetypes[1]}20 15%,
            ${nascarColors.archetypes[2]}15 30%,
            #1a1a1a 45%,
            #0f0f0f 60%,
            #060606 80%,
            #000000 100%
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
              ${nascarColors.primary}15 0%,
              transparent 40%
            ),
            radial-gradient(
              ellipse at bottom right,
              ${nascarColors.archetypes[1]}10 0%,
              transparent 40%
            ),
            linear-gradient(
              45deg,
              transparent 40%,
              ${nascarColors.archetypes[4]}05 60%,
              transparent 80%
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
              transparent 120px,
              ${nascarColors.primary}03 120px,
              ${nascarColors.primary}03 122px
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

// Variation 2: Sharp Lines Similar to Uploaded Image
export function SharpGradientBackground({ children }: BackgroundProps) {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
        background: `
          linear-gradient(
            135deg,
            ${nascarColors.primary} 0%,
            ${nascarColors.primary} 8%,
            ${nascarColors.archetypes[4]} 8%,
            ${nascarColors.archetypes[4]} 16%,
            ${nascarColors.archetypes[5]} 16%,
            ${nascarColors.archetypes[5]} 24%,
            ${nascarColors.archetypes[1]} 24%,
            ${nascarColors.archetypes[1]} 32%,
            #4a0e4e 32%,
            #4a0e4e 40%,
            #2d1b69 40%,
            #2d1b69 48%,
            #1a1a2e 48%,
            #1a1a2e 60%,
            #16213e 60%,
            #16213e 72%,
            #0f0f23 72%,
            #0f0f23 84%,
            #0a0a0a 84%,
            #000000 100%
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
            repeating-linear-gradient(
              135deg,
              transparent 0px,
              transparent 40px,
              rgba(255, 255, 255, 0.02) 40px,
              rgba(255, 255, 255, 0.02) 42px,
              transparent 42px,
              transparent 80px
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
            linear-gradient(
              135deg,
              transparent 0%,
              rgba(0, 0, 0, 0.3) 50%,
              rgba(0, 0, 0, 0.7) 100%
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

// Variation 3: Ultra Sharp Bands (Most Similar to Your Image)
export function BandedGradientBackground({ children }: BackgroundProps) {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
        background: `
          linear-gradient(
            135deg,
            ${nascarColors.primary} 0%,
            ${nascarColors.primary} 12%,
            ${nascarColors.warning} 12%,
            ${nascarColors.warning} 18%,
            ${nascarColors.danger} 18%,
            ${nascarColors.danger} 24%,
            ${nascarColors.archetypes[5]} 24%,
            ${nascarColors.archetypes[5]} 30%,
            #8E44AD 30%,
            #8E44AD 36%,
            #5D4E75 36%,
            #5D4E75 42%,
            #3C415C 42%,
            #3C415C 48%,
            #2C3E50 48%,
            #2C3E50 54%,
            #1C2833 54%,
            #1C2833 60%,
            #17202A 60%,
            #17202A 70%,
            #0B1426 70%,
            #0B1426 80%,
            #080A12 80%,
            #080A12 90%,
            #000000 90%,
            #000000 100%
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
            repeating-linear-gradient(
              135deg,
              transparent 0px,
              transparent 2px,
              rgba(255, 255, 255, 0.03) 2px,
              rgba(255, 255, 255, 0.03) 3px
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

// Variation 4: Blurred Banded Gradient (Fantasy Effect)
export function BlurredBandedBackground({ children }: BackgroundProps) {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
      }}
    >
      {/* Background layer with blur */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            linear-gradient(
              135deg,
              ${nascarColors.primary} 0%,
              ${nascarColors.primary} 12%,
              ${nascarColors.warning} 12%,
              ${nascarColors.warning} 18%,
              ${nascarColors.danger} 18%,
              ${nascarColors.danger} 24%,
              ${nascarColors.archetypes[5]} 24%,
              ${nascarColors.archetypes[5]} 30%,
              #8E44AD 30%,
              #8E44AD 36%,
              #5D4E75 36%,
              #5D4E75 42%,
              #3C415C 42%,
              #3C415C 48%,
              #2C3E50 48%,
              #2C3E50 54%,
              #1C2833 54%,
              #1C2833 60%,
              #17202A 60%,
              #17202A 70%,
              #0B1426 70%,
              #0B1426 80%,
              #080A12 80%,
              #080A12 90%,
              #000000 90%,
              #000000 100%
            )
          `,
          filter: 'blur(8px)',
          transform: 'scale(1.1)',
          zIndex: 0,
        }}
      />
      
      {/* Additional blur layers */}
      <Box
        sx={{
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
            )
          `,
          filter: 'blur(12px)',
          zIndex: 1,
        }}
      />
      
      {/* Content layer - no blur */}
      <Box sx={{ position: 'relative', zIndex: 10 }}>
        {children}
      </Box>
    </Box>
  );
}

// Variation 5: Heavy Blur Fantasy Effect
export function FantasyBlurBackground({ children }: BackgroundProps) {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
      }}
    >
      {/* Main background layer with blur */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            linear-gradient(
              135deg,
              ${nascarColors.primary} 0%,
              ${nascarColors.primary} 12%,
              ${nascarColors.warning} 12%,
              ${nascarColors.warning} 18%,
              ${nascarColors.danger} 18%,
              ${nascarColors.danger} 24%,
              ${nascarColors.archetypes[5]} 24%,
              ${nascarColors.archetypes[5]} 30%,
              #8E44AD 30%,
              #8E44AD 36%,
              #5D4E75 36%,
              #5D4E75 42%,
              #3C415C 42%,
              #3C415C 48%,
              #2C3E50 48%,
              #2C3E50 54%,
              #1C2833 54%,
              #1C2833 60%,
              #17202A 60%,
              #17202A 70%,
              #0B1426 70%,
              #0B1426 80%,
              #080A12 80%,
              #080A12 90%,
              #000000 90%,
              #000000 100%
            )
          `,
          filter: 'blur(16px)',
          transform: 'scale(1.15)',
          zIndex: 0,
        }}
      />
      
      {/* Radial gradient blur layer */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            radial-gradient(
              circle at 20% 30%,
              ${nascarColors.primary}40 0%,
              transparent 40%
            ),
            radial-gradient(
              circle at 80% 70%,
              ${nascarColors.archetypes[1]}30 0%,
              transparent 40%
            ),
            radial-gradient(
              circle at 50% 50%,
              ${nascarColors.archetypes[5]}20 0%,
              transparent 50%
            )
          `,
          filter: 'blur(20px)',
          zIndex: 1,
        }}
      />
      
      {/* Overlay gradient */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            linear-gradient(
              135deg,
              transparent 0%,
              rgba(255, 107, 53, 0.1) 25%,
              rgba(148, 103, 189, 0.1) 50%,
              rgba(0, 0, 0, 0.3) 75%,
              rgba(0, 0, 0, 0.8) 100%
            )
          `,
          filter: 'blur(8px)',
          zIndex: 2,
        }}
      />
      
      {/* Content layer - completely sharp */}
      <Box sx={{ position: 'relative', zIndex: 10 }}>
        {children}
      </Box>
    </Box>
  );
}