// src/app/page.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
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
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Divider,
  IconButton,
  Autocomplete,
} from '@mui/material';
import {
  Search as SearchIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
  Person as PersonIcon,
  CheckCircle as ActiveIcon,
  Cancel as InactiveIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import { nascarColors, getArchetypeColor } from '@/lib/theme';
import { GradientBackground } from '@/components/ui/BackgroundVariations';
import { nascarApi, searchDrivers } from '@/lib/services/nascarApi';
import { queryKeys } from '@/lib/queryClient';
import type { DriverSearchResult } from '@/types/api';

export default function HomePage() {
  const router = useRouter();
  const [searchValue, setSearchValue] = useState('');
  const [selectedDriver, setSelectedDriver] = useState<DriverSearchResult | null>(null);
  const [isSearchFocused, setIsSearchFocused] = useState(false);

  // Query for popular drivers (prefetched in QueryProvider)
  const {
    data: popularDrivers = [],
    isLoading: isLoadingPopular,
    error: popularError,
  } = useQuery({
    queryKey: queryKeys.drivers.popularDrivers(),
    queryFn: () => nascarApi.search.getPopularDrivers(),
    staleTime: 10 * 60 * 1000, // Cache for 10 minutes
    retry: 2,
  });

  // Query for search results (only runs when search term is >= 2 characters)
  const {
    data: searchResults = [],
    isLoading: isSearching,
    error: searchError,
    isFetching: isSearchFetching,
  } = useQuery({
    queryKey: queryKeys.drivers.search(searchValue),
    queryFn: () => searchDrivers(searchValue, 8),
    enabled: searchValue.trim().length >= 2, // Only search when user types 2+ characters
    staleTime: 5 * 60 * 1000, // Cache search results for 5 minutes
    retry: 1,
  });

  // Backend connection health check
  const {
    data: healthData,
    isError: isBackendDown,
  } = useQuery({
    queryKey: queryKeys.health.status(),
    queryFn: () => nascarApi.health.checkHealth(),
    staleTime: 30 * 1000, // Check every 30 seconds
    retry: 2,
    refetchInterval: 30 * 1000, // Auto-refresh every 30 seconds
  });

  const handleDriverSelect = (driver: DriverSearchResult | null) => {
    if (driver) {
      setSelectedDriver(driver);
      setSearchValue(driver.name);
      setIsSearchFocused(false);
      
      // Navigate to driver profile page
      router.push(`/driver/${driver.id}`);
    }
  };

  const handleSearchClear = () => {
    setSearchValue('');
    setSelectedDriver(null);
    setIsSearchFocused(true);
  };

  const handleViewDriver = () => {
    if (selectedDriver) {
      // Navigate to driver profile page
      router.push(`/driver/${selectedDriver.id}`);
    }
  };

  // Show search results or popular drivers based on search state
  const displayResults = searchValue.trim().length >= 2 ? searchResults : popularDrivers;
  const isLoadingResults = searchValue.trim().length >= 2 ? isSearching : isLoadingPopular;
  const resultsError = searchValue.trim().length >= 2 ? searchError : popularError;

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
          {/* Backend Status Alert */}
          {isBackendDown && (
            <Alert 
              severity="warning" 
              sx={{ mb: 3, width: '100%', maxWidth: 600 }}
            >
              NASCAR Analytics backend is currently offline. Some features may not work properly.
            </Alert>
          )}

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
              <Autocomplete
                freeSolo
                options={displayResults}
                value={selectedDriver}
                inputValue={searchValue}
                onInputChange={(event, newInputValue) => {
                  setSearchValue(newInputValue);
                  // Clear selected driver when search input is cleared
                  if (!newInputValue) {
                    setSelectedDriver(null);
                  }
                }}
                onChange={(event, newValue) => {
                  if (typeof newValue === 'object' && newValue !== null) {
                    handleDriverSelect(newValue);
                  }
                }}
                getOptionLabel={(option) => {
                  if (typeof option === 'string') return option;
                  return option.name;
                }}
                renderOption={(props, option) => (
                  <ListItem {...props} key={option.id}>
                    <ListItemButton>
                      <PersonIcon sx={{ mr: 2, color: 'text.secondary' }} />
                      <ListItemText
                        primary={option.name}
                        secondary={
                          <Stack direction="row" spacing={1} alignItems="center">
                            <Typography variant="caption" color="text.secondary">
                              {option.career_span} • {option.total_wins} wins
                            </Typography>
                            {option.is_active ? (
                              <ActiveIcon sx={{ fontSize: 16, color: 'success.main' }} />
                            ) : (
                              <InactiveIcon sx={{ fontSize: 16, color: 'text.disabled' }} />
                            )}
                          </Stack>
                        }
                      />
                    </ListItemButton>
                  </ListItem>
                )}
                loading={isLoadingResults || isSearchFetching}
                loadingText="Searching NASCAR drivers..."
                noOptionsText={
                  searchValue.trim().length < 2 
                    ? "Type at least 2 characters to search" 
                    : "No drivers found"
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    fullWidth
                    placeholder="Search for a driver..."
                    variant="outlined"
                    onFocus={() => setIsSearchFocused(true)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        backgroundColor: 'rgba(0, 0, 0, 0.2)',
                        backdropFilter: 'blur(10px)',
                        borderRadius: 3,
                        fontSize: '1.2rem',
                        '& fieldset': {
                          borderColor: 'rgba(255, 107, 53, 0.3)',
                          borderWidth: 2,
                        },
                        '&:hover fieldset': {
                          borderColor: nascarColors.primary,
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: nascarColors.primary,
                          boxShadow: `0 0 20px ${nascarColors.primary}40`,
                        },
                      },
                      '& input': {
                        padding: '16px 14px',
                        color: 'white',
                        '&::placeholder': {
                          color: 'rgba(255, 255, 255, 0.6)',
                          opacity: 1,
                        },
                      },
                    }}
                    InputProps={{
                      ...params.InputProps,
                      startAdornment: (
                        <InputAdornment position="start">
                          <SearchIcon sx={{ color: nascarColors.primary }} />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <Stack direction="row" spacing={1} alignItems="center">
                          {(isLoadingResults || isSearchFetching) && (
                            <CircularProgress size={20} sx={{ color: nascarColors.primary }} />
                          )}
                          {params.InputProps.endAdornment}
                        </Stack>
                      ),
                    }}
                  />
                )}
              />

              {/* Error Message */}
              {resultsError && !isBackendDown && (
                <Alert 
                  severity="error" 
                  sx={{ mt: 2 }}
                >
                  Failed to search drivers. Please check your connection and try again.
                </Alert>
              )}

              {/* Selected Driver Action Button */}
              {selectedDriver && (
                <Fade in timeout={300}>
                  <Box sx={{ mt: 3 }}>
                    <Button
                      variant="contained"
                      size="large"
                      onClick={handleViewDriver}
                      sx={{
                        backgroundColor: nascarColors.primary,
                        color: 'white',
                        px: 4,
                        py: 1.5,
                        fontSize: '1.1rem',
                        fontWeight: 600,
                        borderRadius: 3,
                        boxShadow: `0 4px 20px ${nascarColors.primary}40`,
                        '&:hover': {
                          backgroundColor: '#E55A2B', // Darker shade of NASCAR orange
                          boxShadow: `0 6px 25px ${nascarColors.primary}60`,
                          transform: 'translateY(-2px)',
                        },
                        transition: 'all 0.3s ease',
                      }}
                    >
                      View {selectedDriver.name}'s Profile
                    </Button>
                  </Box>
                </Fade>
              )}
            </Box>
          </Fade>

          {/* Popular Drivers Section (when no search) */}
          {!selectedDriver && searchValue.trim().length < 2 && (
            <Fade in timeout={2000}>
              <Box sx={{ width: '100%', maxWidth: 600 }}>
                <Typography 
                  variant="h6" 
                  color="text.secondary" 
                  sx={{ mb: 3, opacity: 0.8 }}
                >
                  Popular Drivers
                </Typography>
                
                {isLoadingPopular ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                    <CircularProgress sx={{ color: nascarColors.primary }} />
                  </Box>
                ) : popularError ? (
                  <Alert severity="info">
                    Unable to load popular drivers. You can still search manually above.
                  </Alert>
                ) : (
                  <Stack 
                    direction="row" 
                    spacing={1} 
                    flexWrap="wrap" 
                    justifyContent="center"
                    sx={{ gap: 1 }}
                  >
                    {popularDrivers.map((driver) => (
                      <Chip
                        key={driver.id}
                        label={`${driver.name} (${driver.total_wins} wins)`}
                        onClick={() => handleDriverSelect(driver)}
                        variant="outlined"
                        sx={{
                          borderColor: nascarColors.primary,
                          color: 'white',
                          '&:hover': {
                            backgroundColor: `${nascarColors.primary}20`,
                            borderColor: nascarColors.primary,
                          },
                          cursor: 'pointer',
                        }}
                      />
                    ))}
                  </Stack>
                )}

                {/* Explore Archetypes Button */}
                <Box sx={{ textAlign: 'center', mt: 4 }}>
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={() => router.push('/archetypes')}
                    sx={{
                      borderColor: nascarColors.archetypes[1],
                      color: nascarColors.archetypes[1],
                      px: 4,
                      py: 1.5,
                      fontWeight: 600,
                      '&:hover': {
                        backgroundColor: `${nascarColors.archetypes[1]}20`,
                        borderColor: nascarColors.archetypes[1],
                      },
                    }}
                  >
                    Explore Driver Archetypes
                  </Button>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1, opacity: 0.7 }}>
                    Discover 6 driver types identified by machine learning
                  </Typography>
                </Box>
              </Box>
            </Fade>
          )}

          {/* Connection Status */}
          <Box sx={{ mt: 4, opacity: 0.6 }}>
            <Typography variant="caption" color="text.secondary">
              Backend Status: {healthData ? (
                <Chip 
                  label="Connected" 
                  size="small" 
                  color="success" 
                  variant="outlined"
                />
              ) : (
                <Chip 
                  label="Offline" 
                  size="small" 
                  color="error" 
                  variant="outlined"
                />
              )}
            </Typography>
          </Box>
        </Box>
      </Container>
    </GradientBackground>
  );
}