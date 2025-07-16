// src/app/page.tsx
'use client';

import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  Stack,
  Card,
  CardContent,
  Chip,
  InputAdornment,
  Fade,
} from '@mui/material';
import {
  Search as SearchIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { nascarColors, getArchetypeColor } from '@/lib/theme';
import { GradientBackground as GradientBackground } from '@/components/ui/BackgroundVariations';

export default function HomePage() {
  const [searchValue, setSearchValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Sample popular drivers for suggestions
  const popularDrivers = [
    'Kyle Larson', 'Chase Elliott', 'Kyle Busch', 'Denny Hamlin',
    'Martin Truex Jr', 'Kevin Harvick', 'Joey Logano', 'Brad Keselowski'
  ];

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchValue.trim()) {
      // TODO: Navigate to driver page or results
      console.log('Searching for:', searchValue);
    }
  };

  const handleDriverSelect = (driverName: string) => {
    setSearchValue(driverName);
    setShowSuggestions(false);
  };

  return (
    <GradientBackground>
      <Container maxWidth="md">
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            textAlign: 'center',
            py: 4,
          }}
        >
          {/* Hero Section */}
          <Fade in timeout={1000}>
            <Box sx={{ mb: 6 }}>
              <Stack direction="row" spacing={3} justifyContent="center" alignItems="center" sx={{ mb: 4 }}>
                <SpeedIcon sx={{ fontSize: 60, color: nascarColors.primary }} />
                <TrendingUpIcon sx={{ fontSize: 60, color: nascarColors.archetypes[1] }} />
              </Stack>
              
              <Typography 
                variant="h1" 
                sx={{ 
                  mb: 2,
                  fontSize: { xs: '2.5rem', md: '3.5rem' },
                  fontWeight: 800,
                  background: `linear-gradient(135deg, ${nascarColors.primary} 0%, ${nascarColors.archetypes[1]} 100%)`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  textShadow: `0 0 40px ${nascarColors.primary}40`,
                }}
              >
                NASCAR Analytics
              </Typography>
              
              <Typography 
                variant="h5" 
                color="text.secondary" 
                sx={{ 
                  mb: 1,
                  fontWeight: 300,
                  opacity: 0.9,
                }}
              >
                Discover Your Favorite Driver's Career Story
              </Typography>
              
              <Typography 
                variant="body1" 
                color="text.secondary" 
                sx={{ 
                  opacity: 0.7,
                  maxWidth: 500,
                  mx: 'auto',
                }}
              >
                Advanced analytics, career patterns, and performance insights for NASCAR drivers
              </Typography>
            </Box>
          </Fade>

          {/* Search Section */}
          <Fade in timeout={1500}>
            <Box sx={{ width: '100%', maxWidth: 600, mb: 4 }}>
              <form onSubmit={handleSearchSubmit}>
                <TextField
                  fullWidth
                  value={searchValue}
                  onChange={(e) => setSearchValue(e.target.value)}
                  onFocus={() => setShowSuggestions(true)}
                  placeholder="Search for a driver..."
                  variant="outlined"
                  slotProps={{
                    input: {
                      startAdornment: (
                        <InputAdornment position="start">
                          <SearchIcon sx={{ color: nascarColors.primary }} />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <Button
                            type="submit"
                            variant="contained"
                            sx={{
                              minWidth: 100,
                              fontWeight: 600,
                            }}
                          >
                            Search
                          </Button>
                        </InputAdornment>
                      ),
                    },
                  }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '1.2rem',
                      padding: '12px 16px',
                      backgroundColor: 'rgba(18, 18, 18, 0.8)',
                      backdropFilter: 'blur(10px)',
                      borderRadius: 2,
                      '& fieldset': {
                        borderWidth: 2,
                        borderColor: nascarColors.dark.border.primary,
                      },
                      '&:hover fieldset': {
                        borderColor: nascarColors.primary,
                      },
                      '&.Mui-focused fieldset': {
                        borderColor: nascarColors.primary,
                        boxShadow: `0 0 0 3px ${nascarColors.primary}20`,
                      },
                    },
                  }}
                />
              </form>

              {/* Popular Drivers Suggestions */}
              {showSuggestions && (
                <Fade in timeout={300}>
                  <Card sx={{ 
                    mt: 2, 
                    backgroundColor: 'rgba(18, 18, 18, 0.9)',
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${nascarColors.dark.border.primary}`,
                  }}>
                    <CardContent sx={{ p: 2 }}>
                      <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                        Popular Drivers:
                      </Typography>
                      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        {popularDrivers.map((driver, index) => (
                          <Chip
                            key={driver}
                            label={driver}
                            onClick={() => handleDriverSelect(driver)}
                            sx={{
                              backgroundColor: nascarColors.archetypes[index % nascarColors.archetypes.length],
                              color: 'white',
                              fontWeight: 500,
                              cursor: 'pointer',
                              '&:hover': {
                                filter: 'brightness(1.2)',
                                transform: 'scale(1.05)',
                              },
                              transition: 'all 0.2s ease-in-out',
                            }}
                          />
                        ))}
                      </Stack>
                    </CardContent>
                  </Card>
                </Fade>
              )}
            </Box>
          </Fade>

          {/* Quick Stats */}
          <Fade in timeout={2000}>
            <Stack 
              direction={{ xs: 'column', sm: 'row' }} 
              spacing={4} 
              sx={{ 
                opacity: 0.8,
                '& > *': {
                  textAlign: 'center',
                }
              }}
            >
              <Box>
                <Typography variant="h4" sx={{ color: nascarColors.primary, fontWeight: 700 }}>
                  289
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Drivers
                </Typography>
              </Box>
              <Box>
                <Typography variant="h4" sx={{ color: nascarColors.archetypes[1], fontWeight: 700 }}>
                  6
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Archetypes
                </Typography>
              </Box>
              <Box>
                <Typography variant="h4" sx={{ color: nascarColors.archetypes[2], fontWeight: 700 }}>
                  50+
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Years
                </Typography>
              </Box>
            </Stack>
          </Fade>

          {/* Bottom CTA */}
          <Fade in timeout={2500}>
            <Box sx={{ mt: 6 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, opacity: 0.7 }}>
                Or explore driver archetypes and career patterns
              </Typography>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                <Button 
                  variant="outlined" 
                  sx={{ 
                    borderWidth: 2,
                    '&:hover': { borderWidth: 2 }
                  }}
                >
                  Browse Archetypes
                </Button>
                <Button 
                  variant="text"
                  sx={{ color: 'text.secondary' }}
                >
                  Learn More
                </Button>
              </Stack>
            </Box>
          </Fade>
        </Box>
      </Container>
    </GradientBackground>
  );
}