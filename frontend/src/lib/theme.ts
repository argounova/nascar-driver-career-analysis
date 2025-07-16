// src/lib/theme.ts
import { createTheme, ThemeOptions } from '@mui/material/styles';

// NASCAR Brand Colors from config.yaml
export const nascarColors = {
  // Primary brand colors
  primary: '#FF6B35',      // NASCAR orange
  secondary: '#1f77b4',    // Blue
  accent: '#2ca02c',       // Green
  warning: '#ff7f0e',      // Orange
  danger: '#d62728',       // Red
  
  // Driver archetype colors (6 distinct colors for data visualization)
  archetypes: [
    '#FF6B35',  // Dominant Champions - NASCAR Orange
    '#1f77b4',  // Elite Competitors - Blue  
    '#2ca02c',  // Solid Performers - Green
    '#ff7f0e',  // Mid-Pack Drivers - Orange
    '#d62728',  // Developing Drivers - Red
    '#9467bd'   // Late Bloomers - Purple
  ],
  
  // Dark theme base colors
  dark: {
    background: {
      default: '#0a0a0a',      // Deep black background
      paper: '#121212',        // Slightly lighter for cards
      elevated: '#1a1a1a',     // For elevated components
    },
    surface: {
      primary: '#1e1e1e',      // Primary surface color
      secondary: '#252525',    // Secondary surface
      tertiary: '#2a2a2a',     // Tertiary surface
    },
    text: {
      primary: '#ffffff',      // Pure white for main text
      secondary: '#b3b3b3',    // Light gray for secondary text
      tertiary: '#808080',     // Medium gray for tertiary text
      disabled: '#4a4a4a',     // Dark gray for disabled text
    },
    border: {
      primary: '#333333',      // Primary border color
      secondary: '#404040',    // Secondary border
      accent: '#FF6B35',       // NASCAR orange for accent borders
    }
  }
};

// Create the NASCAR-themed MUI theme
export const nascarTheme = createTheme({
  palette: {
    mode: 'dark',
    
    // Primary color (NASCAR Orange)
    primary: {
      main: nascarColors.primary,
      light: '#ff8659',
      dark: '#e55a2b',
      contrastText: '#ffffff',
    },
    
    // Secondary color (Blue)
    secondary: {
      main: nascarColors.secondary,
      light: '#4d94d1',
      dark: '#0d5aa7',
      contrastText: '#ffffff',
    },
    
    // Background colors
    background: {
      default: nascarColors.dark.background.default,
      paper: nascarColors.dark.background.paper,
    },
    
    // Surface colors (custom property for consistency)
    surface: {
      primary: nascarColors.dark.surface.primary,
      secondary: nascarColors.dark.surface.secondary,
      tertiary: nascarColors.dark.surface.tertiary,
    } as any,
    
    // Text colors
    text: {
      primary: nascarColors.dark.text.primary,
      secondary: nascarColors.dark.text.secondary,
      disabled: nascarColors.dark.text.disabled,
    },
    
    // Success, Warning, Error with NASCAR flair
    success: {
      main: nascarColors.accent,
      light: '#4caf50',
      dark: '#2e7d32',
    },
    warning: {
      main: nascarColors.warning,
      light: '#ffb74d',
      dark: '#f57c00',
    },
    error: {
      main: nascarColors.danger,
      light: '#e57373',
      dark: '#c62828',
    },
    
    // Custom colors for archetype visualization
    archetype: {
      dominantChampions: nascarColors.archetypes[0],
      eliteCompetitors: nascarColors.archetypes[1], 
      solidPerformers: nascarColors.archetypes[2],
      midPackDrivers: nascarColors.archetypes[3],
      developingDrivers: nascarColors.archetypes[4],
      lateBloomers: nascarColors.archetypes[5],
    } as any,
    
    // Divider color
    divider: nascarColors.dark.border.primary,
  },
  
  typography: {
    fontFamily: '"Geist Sans", "Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
      color: nascarColors.dark.text.primary,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
      color: nascarColors.dark.text.primary,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      color: nascarColors.dark.text.primary,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: nascarColors.dark.text.primary,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      color: nascarColors.dark.text.primary,
    },
    h6: {
      fontSize: '1.125rem',
      fontWeight: 600,
      color: nascarColors.dark.text.primary,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
      color: nascarColors.dark.text.primary,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
      color: nascarColors.dark.text.secondary,
    },
    caption: {
      fontSize: '0.75rem',
      color: nascarColors.dark.text.tertiary,
    },
  },
  
  shape: {
    borderRadius: 8, // Slightly rounded for modern feel
  },
  
  components: {
    // AppBar styling
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: nascarColors.dark.background.paper,
          borderBottom: `1px solid ${nascarColors.dark.border.primary}`,
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    
    // Paper/Card styling
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: nascarColors.dark.background.paper,
          border: `1px solid ${nascarColors.dark.border.primary}`,
          '&.elevated': {
            backgroundColor: nascarColors.dark.background.elevated,
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.4)',
          },
        },
        elevation1: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.3)',
        },
        elevation2: {
          boxShadow: '0 2px 6px rgba(0, 0, 0, 0.4)',
        },
        elevation3: {
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
        },
      },
    },
    
    // Card styling with NASCAR flair
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: nascarColors.dark.surface.primary,
          border: `1px solid ${nascarColors.dark.border.primary}`,
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            borderColor: nascarColors.dark.border.accent,
            boxShadow: `0 4px 12px rgba(255, 107, 53, 0.2)`, // NASCAR orange glow
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    
    // Button styling
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 6,
          transition: 'all 0.2s ease-in-out',
        },
        contained: {
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
          '&:hover': {
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.4)',
            transform: 'translateY(-1px)',
          },
        },
        outlined: {
          borderWidth: '2px',
          '&:hover': {
            borderWidth: '2px',
          },
        },
      },
    },
    
    // Chip styling for stats and categories
    MuiChip: {
      styleOverrides: {
        root: {
          fontSize: '0.75rem',
          fontWeight: 600,
          '&.archetype-chip': {
            color: '#ffffff',
            fontWeight: 700,
          },
        },
        filled: {
          '&.wins-chip': {
            backgroundColor: nascarColors.archetypes[0],
            color: '#ffffff',
          },
          '&.top5-chip': {
            backgroundColor: nascarColors.archetypes[1],
            color: '#ffffff',
          },
          '&.seasons-chip': {
            backgroundColor: nascarColors.archetypes[2],
            color: '#ffffff',
          },
        },
      },
    },
    
    // TextField styling
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: nascarColors.dark.surface.secondary,
            '& fieldset': {
              borderColor: nascarColors.dark.border.primary,
            },
            '&:hover fieldset': {
              borderColor: nascarColors.dark.border.secondary,
            },
            '&.Mui-focused fieldset': {
              borderColor: nascarColors.primary,
            },
          },
        },
      },
    },
    
    // Table styling for driver data
    MuiTable: {
      styleOverrides: {
        root: {
          backgroundColor: nascarColors.dark.surface.primary,
        },
      },
    },
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: nascarColors.dark.surface.secondary,
          '& .MuiTableCell-head': {
            fontWeight: 700,
            color: nascarColors.dark.text.primary,
            borderBottom: `2px solid ${nascarColors.primary}`,
          },
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: nascarColors.dark.surface.tertiary,
          },
          '&.driver-row': {
            borderLeft: `4px solid transparent`,
            '&:hover': {
              borderLeftColor: nascarColors.primary,
            },
          },
        },
      },
    },
    
    // Drawer/Navigation styling
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: nascarColors.dark.background.paper,
          borderRight: `1px solid ${nascarColors.dark.border.primary}`,
        },
      },
    },
    
    // List item styling for navigation
    MuiListItem: {
      styleOverrides: {
        root: {
          '&.nav-item': {
            borderRadius: 6,
            margin: '4px 8px',
            '&:hover': {
              backgroundColor: nascarColors.dark.surface.tertiary,
            },
            '&.active': {
              backgroundColor: `${nascarColors.primary}20`, // NASCAR orange with transparency
              borderLeft: `4px solid ${nascarColors.primary}`,
            },
          },
        },
      },
    },
  },
} as ThemeOptions);

// Utility function to get archetype color by name
export const getArchetypeColor = (archetypeName: string): string => {
  const colorMap: Record<string, string> = {
    'Dominant Champions': nascarColors.archetypes[0],
    'Elite Competitors': nascarColors.archetypes[1],
    'Solid Performers': nascarColors.archetypes[2],
    'Mid-Pack Drivers': nascarColors.archetypes[3],
    'Developing Drivers': nascarColors.archetypes[4],
    'Late Bloomers': nascarColors.archetypes[5],
  };
  
  return colorMap[archetypeName] || nascarColors.primary;
};

// Utility function for consistent spacing
export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

export default nascarTheme;