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
  Grid2 as Grid,
} from '@mui/material';
import {
  Search as SearchIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
  Person as PersonIcon,
  CheckCircle as ActiveIcon,
  Cancel as InactiveIcon,
  Clear as ClearIcon,
  Psychology as BrainIcon,
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

  const handleSearchSubmit = (event?: React.FormEvent) => {
    if (event) {
      event.preventDefault();
    }
    
    // If there's a selected driver, navigate to their profile
    if (selectedDriver) {
      router.push(`/driver/${selectedDriver.id}`);
    } else if (searchResults.length > 0) {
      // If no driver selected but there are results, use the first one
      const firstResult = searchResults[0];
      router.push(`/driver/${firstResult.id}`);
    }
  };

  const handleSearchIconClick = () => {
    handleSearchSubmit();
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleSearchSubmit();
    }
  };

  // Show search results
  const displayResults = searchResults;
  const isLoadingResults = isSearching;
  const resultsError = searchError;

  return (
    <GradientBackground>
      <Container maxWidth="lg">
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
              sx={{ mb: 3, width: '100%', maxWidth: 800 }}
            >
              NASCAR Analytics backend is currently offline. Some features may not work properly.
            </Alert>
          )}

          {/* Main Title */}
          <Fade in timeout={1000}>
            <Box sx={{ mb: 6 }}>
              <Typography
                variant="h1"
                sx={{
                  fontWeight: 800,
                  fontSize: { xs: '2.5rem', md: '4rem' },
                  background: `linear-gradient(135deg, ${nascarColors.primary} 0%, #ff8659 100%)`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  mb: 2,
                }}
              >
                NASCAR Analytics
              </Typography>
              <Typography
                variant="h5"
                color="text.secondary"
                sx={{ mb: 4, opacity: 0.8, lineHeight: 1.4 }}
              >
                Dive deep into NASCAR Cup Series driver performance and discover data-driven insights
              </Typography>
            </Box>
          </Fade>

          {/* Main Content Grid */}
          <Fade in timeout={1500}>
            <Grid container spacing={4} sx={{ width: '100%', maxWidth: 1000 }}>
              {/* Driver Search Section */}
              <Grid size={{ xs: 12, md: 6 }}>
                <Card
                  sx={{
                    height: '100%',
                    background: 'rgba(0, 0, 0, 0.3)',
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${nascarColors.primary}40`,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: `0 8px 30px ${nascarColors.primary}40`,
                      border: `1px solid ${nascarColors.primary}60`,
                    },
                  }}
                >
                  <CardContent sx={{ p: 4, height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ textAlign: 'center', mb: 3 }}>
                      <PersonIcon sx={{ fontSize: 48, color: nascarColors.primary, mb: 2 }} />
                      <Typography variant="h4" sx={{ fontWeight: 700, mb: 2 }}>
                        Search Drivers
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                        Find detailed performance analysis for any NASCAR Cup Series driver
                      </Typography>
                    </Box>

                    {/* Search Input */}
                    <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      <Autocomplete
                        freeSolo
                        options={displayResults}
                        value={selectedDriver}
                        inputValue={searchValue}
                        loading={isLoadingResults || isSearchFetching}
                        onInputChange={(event, newInputValue) => {
                          setSearchValue(newInputValue);
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
                        loadingText="Searching NASCAR drivers..."
                        noOptionsText={
                          searchValue.trim().length < 2 
                            ? "Type 2+ characters to search"
                            : "No drivers found"
                        }
                        renderInput={(params) => (
                          <TextField
                            {...params}
                            fullWidth
                            variant="outlined"
                            placeholder="Type driver name..."
                            onFocus={() => setIsSearchFocused(true)}
                            onBlur={() => setIsSearchFocused(false)}
                            onKeyDown={handleKeyPress}
                            slotProps={{
                              input: {
                                ...params.InputProps,
                                startAdornment: (
                                  <InputAdornment position="start">
                                    <IconButton
                                      onClick={handleSearchIconClick}
                                      sx={{ 
                                        color: nascarColors.primary,
                                        '&:hover': { backgroundColor: `${nascarColors.primary}20` }
                                      }}
                                    >
                                      <SearchIcon />
                                    </IconButton>
                                  </InputAdornment>
                                ),
                                endAdornment: (
                                  <>
                                    {searchValue && (
                                      <InputAdornment position="end">
                                        <IconButton
                                          onClick={handleSearchClear}
                                          sx={{ color: 'text.secondary' }}
                                        >
                                          <ClearIcon />
                                        </IconButton>
                                      </InputAdornment>
                                    )}
                                  </>
                                ),
                                sx: {
                                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                  backdropFilter: 'blur(5px)',
                                  '& .MuiOutlinedInput-notchedOutline': {
                                    borderColor: 'rgba(255, 255, 255, 0.2)',
                                  },
                                  '&:hover .MuiOutlinedInput-notchedOutline': {
                                    borderColor: nascarColors.primary,
                                  },
                                  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                                    borderColor: nascarColors.primary,
                                  },
                                },
                              },
                            }}
                          />
                        )}
                      />

                      {/* Search Error */}
                      {resultsError && (
                        <Alert severity="error" sx={{ mt: 2 }}>
                          Please check your connection and try again.
                        </Alert>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* Driver Archetypes Section */}
              <Grid size={{ xs: 12, md: 6 }}>
                <Card
                  sx={{
                    height: '100%',
                    background: 'rgba(0, 0, 0, 0.3)',
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${nascarColors.archetypes[1]}40`,
                    transition: 'all 0.3s ease',
                    cursor: 'pointer',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: `0 8px 30px ${nascarColors.archetypes[1]}40`,
                      border: `1px solid ${nascarColors.archetypes[1]}60`,
                    },
                  }}
                  onClick={() => router.push('/archetypes')}
                >
                  <CardContent sx={{ p: 4, height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ textAlign: 'center', mb: 3 }}>
                      <BrainIcon sx={{ fontSize: 48, color: nascarColors.archetypes[1], mb: 2 }} />
                      <Typography variant="h4" sx={{ fontWeight: 700, mb: 2 }}>
                        Driver Archetypes
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                        Explore 6 distinct driver types identified through machine learning analysis
                      </Typography>
                    </Box>

                    <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      <Stack spacing={3}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Button
                            fullWidth
                            variant="outlined"
                            size="large"
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
                            Explore All Archetypes
                          </Button>
                        </Box>

                        {/* <Stack direction="row" spacing={1} justifyContent="center" flexWrap="wrap" sx={{ gap: 1 }}>
                          <Chip
                            label="Elite Champions"
                            size="small"
                            sx={{
                              backgroundColor: `${nascarColors.archetypes[0]}30`,
                              color: nascarColors.archetypes[0],
                              border: `1px solid ${nascarColors.archetypes[0]}`,
                            }}
                          />
                          <Chip
                            label="Solid Performers"
                            size="small"
                            sx={{
                              backgroundColor: `${nascarColors.archetypes[2]}30`,
                              color: nascarColors.archetypes[2],
                              border: `1px solid ${nascarColors.archetypes[2]}`,
                            }}
                          />
                          <Chip
                            label="Mid-Pack Drivers"
                            size="small"
                            sx={{
                              backgroundColor: `${nascarColors.archetypes[3]}30`,
                              color: nascarColors.archetypes[3],
                              border: `1px solid ${nascarColors.archetypes[3]}`,
                            }}
                          />
                        </Stack> */}
                      </Stack>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Fade>

          {/* Connection Status */}
          <Box sx={{ mt: 6, opacity: 0.6 }}>
            <Typography variant="caption" color="text.secondary">
              Status: {healthData ? (
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