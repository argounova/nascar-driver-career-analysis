// src/app/drivers/page.tsx
'use client';

import React, { useState, useMemo, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import {
  Container,
  Typography,
  Box,
  TextField,
  InputAdornment,
  Stack,
  Chip,
  CircularProgress,
  Alert,
  Fade,
  ToggleButton,
  ToggleButtonGroup,
  Card,
  CardContent,
  CardActionArea,
  Fab,
  Zoom,
} from '@mui/material';
import {
  Search as SearchIcon,
  GridView as GridIcon,
  ViewComfy as ComfyIcon,
  CheckCircle as ActiveIcon,
  Cancel as InactiveIcon,
  EmojiEvents as TrophyIcon,
  Timeline as TimelineIcon,
  Person as PersonIcon,
  KeyboardArrowUp as ArrowUpIcon,
} from '@mui/icons-material';
import { nascarColors } from '@/lib/theme';
import { GradientBackground } from '@/components/ui/BackgroundVariations';
import { nascarApi } from '@/lib/services/nascarApi';
import { queryKeys } from '@/lib/queryClient';
import type { DriverSearchResult } from '@/types/api';

type ViewMode = 'compact' | 'comfortable';
type FilterMode = 'all' | 'active' | 'winners';

export default function DriversPage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('comfortable');
  const [filterMode, setFilterMode] = useState<FilterMode>('all');
  const [showBackToTop, setShowBackToTop] = useState(false);

  // Handle scroll to show/hide back to top button
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      setShowBackToTop(scrollTop > 300); // Show after scrolling 300px
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Scroll to top function
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  // Fetch all drivers using the drivers list API
  const {
    data: driversResponse,
    isLoading,
    error,
    isError,
  } = useQuery({
    queryKey: queryKeys.drivers.list({ limit: 1000 }),
    queryFn: () => nascarApi.list.getDrivers({ limit: 1000 }),
    staleTime: 15 * 60 * 1000, // Cache for 15 minutes
    retry: 2,
  });

  // Convert Driver[] to DriverSearchResult[] for consistency
  const allDrivers: DriverSearchResult[] = driversResponse?.drivers.map(driver => ({
    id: driver.id,
    name: driver.name,
    total_wins: driver.career_stats.total_wins,
    is_active: driver.is_active,
    career_span: driver.career_span,
  })) || [];

  // Filter and search drivers
  const filteredDrivers = useMemo(() => {
    let drivers = [...allDrivers];

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      drivers = drivers.filter(driver =>
        driver.name.toLowerCase().includes(query)
      );
    }

    // Apply category filter
    switch (filterMode) {
      case 'active':
        drivers = drivers.filter(driver => driver.is_active);
        break;
      case 'winners':
        drivers = drivers.filter(driver => driver.total_wins > 0);
        break;
      case 'all':
      default:
        // No additional filtering
        break;
    }

    // Sort drivers alphabetically by name
    return drivers.sort((a, b) => a.name.localeCompare(b.name));
  }, [allDrivers, searchQuery, filterMode]);

  // Generate random card heights for masonry effect
  const getCardHeight = (index: number, mode: ViewMode) => {
    const baseHeight = mode === 'compact' ? 160 : 180;
    const variations = mode === 'compact' ? [0, 20, 30] : [0, 25, 40];
    return baseHeight + variations[index % variations.length];
  };

  // Loading state
  if (isLoading) {
    return (
      <GradientBackground>
        <Container maxWidth="xl">
          <Box
            sx={{
              minHeight: '80vh',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <CircularProgress 
              size={60} 
              sx={{ color: nascarColors.primary, mb: 2 }} 
            />
            <Typography variant="h6" color="text.secondary">
              Loading NASCAR drivers...
            </Typography>
          </Box>
        </Container>
      </GradientBackground>
    );
  }

  // Error state
  if (isError) {
    return (
      <GradientBackground>
        <Container maxWidth="xl">
          <Box sx={{ py: 6 }}>
            <Alert 
              severity="error" 
              sx={{ maxWidth: 600, mx: 'auto' }}
            >
              {error?.message || 'Failed to load drivers data'}
            </Alert>
          </Box>
        </Container>
      </GradientBackground>
    );
  }

  return (
    <GradientBackground>
      <Container maxWidth="xl">
        <Box sx={{ py: 4 }}>
          {/* Header Section */}
          <Box sx={{ mb: 4, textAlign: 'center' }}>
            <Typography
              variant="h2"
              sx={{
                fontWeight: 700,
                mb: 2,
                background: `linear-gradient(45deg, ${nascarColors.primary}, #ff8659)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              NASCAR Drivers
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
              Explore the complete gallery of NASCAR Cup Series drivers
            </Typography>
            
            {/* Sorting Note */}
            <Typography variant="body2" color="text.secondary" sx={{ mb: 4, fontStyle: 'italic' }}>
              Drivers are sorted alphabetically by name
            </Typography>

            {/* Stats Row */}
            <Stack 
              direction={{ xs: 'column', sm: 'row' }} 
              spacing={3} 
              justifyContent="center"
              sx={{ mb: 4 }}
            >
              <Chip
                icon={<GridIcon sx={{ fontSize: 16 }} />}
                label={`${allDrivers.length} Total Drivers`}
                variant="outlined"
                sx={{
                  borderColor: nascarColors.primary,
                  color: 'white',
                  '& .MuiChip-icon': { color: nascarColors.primary },
                }}
              />
              <Chip
                icon={<ActiveIcon sx={{ fontSize: 16, color: '#4caf50' }} />}
                label={`${allDrivers.filter((d: DriverSearchResult) => d.is_active).length} Active`}
                variant="outlined"
                sx={{
                  borderColor: '#4caf50',
                  color: 'white',
                }}
              />
              <Chip
                icon={<InactiveIcon sx={{ fontSize: 16, color: '#757575' }} />}
                label={`${allDrivers.filter((d: DriverSearchResult) => !d.is_active).length} Inactive`}
                variant="outlined"
                sx={{
                  borderColor: '#757575',
                  color: 'white',
                }}
              />
            </Stack>
          </Box>

          {/* Controls Section */}
          <Box sx={{ mb: 4 }}>
            <Stack 
              direction={{ xs: 'column', lg: 'row' }} 
              spacing={3} 
              alignItems={{ xs: 'stretch', lg: 'center' }}
              justifyContent="space-between"
            >
              {/* Search Bar */}
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Search drivers by name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon sx={{ color: 'text.secondary' }} />
                    </InputAdornment>
                  ),
                  sx: {
                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                    backdropFilter: 'blur(10px)',
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
                }}
                sx={{ maxWidth: { lg: 400 } }}
              />

              {/* Controls Row */}
              <Stack direction="row" spacing={2} alignItems="center">
                {/* Filter Toggle */}
                <ToggleButtonGroup
                  value={filterMode}
                  exclusive
                  onChange={(_, newFilter) => newFilter && setFilterMode(newFilter)}
                  size="small"
                  sx={{
                    '& .MuiToggleButton-root': {
                      color: 'text.secondary',
                      borderColor: 'rgba(255, 255, 255, 0.2)',
                      '&.Mui-selected': {
                        backgroundColor: `${nascarColors.primary}30`,
                        color: nascarColors.primary,
                        borderColor: nascarColors.primary,
                      },
                      '&:hover': {
                        backgroundColor: `${nascarColors.primary}20`,
                      },
                    },
                  }}
                >
                  <ToggleButton value="all">All</ToggleButton>
                  <ToggleButton value="active">Active</ToggleButton>
                  <ToggleButton value="winners">Winners</ToggleButton>
                </ToggleButtonGroup>

                {/* View Mode Toggle */}
                <ToggleButtonGroup
                  value={viewMode}
                  exclusive
                  onChange={(_, newMode) => newMode && setViewMode(newMode)}
                  size="small"
                  sx={{
                    '& .MuiToggleButton-root': {
                      color: 'text.secondary',
                      borderColor: 'rgba(255, 255, 255, 0.2)',
                      '&.Mui-selected': {
                        backgroundColor: `${nascarColors.primary}30`,
                        color: nascarColors.primary,
                        borderColor: nascarColors.primary,
                      },
                      '&:hover': {
                        backgroundColor: `${nascarColors.primary}20`,
                      },
                    },
                  }}
                >
                  <ToggleButton value="compact">
                    <GridIcon sx={{ fontSize: 18 }} />
                  </ToggleButton>
                  <ToggleButton value="comfortable">
                    <ComfyIcon sx={{ fontSize: 18 }} />
                  </ToggleButton>
                </ToggleButtonGroup>
              </Stack>
            </Stack>

            {/* Active Filters Display */}
            {(searchQuery || filterMode !== 'all') && (
              <Box sx={{ mt: 2 }}>
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                  <Typography variant="body2" color="text.secondary">
                    Showing {filteredDrivers.length} drivers
                  </Typography>
                  {searchQuery && (
                    <Chip
                      label={`Search: "${searchQuery}"`}
                      size="small"
                      onDelete={() => setSearchQuery('')}
                      sx={{
                        backgroundColor: `${nascarColors.primary}20`,
                        color: nascarColors.primary,
                        '& .MuiChip-deleteIcon': { color: nascarColors.primary },
                      }}
                    />
                  )}
                  {filterMode !== 'all' && (
                    <Chip
                      label={`Filter: ${filterMode.charAt(0).toUpperCase() + filterMode.slice(1)}`}
                      size="small"
                      onDelete={() => setFilterMode('all')}
                      sx={{
                        backgroundColor: `${nascarColors.archetypes[1]}20`,
                        color: nascarColors.archetypes[1],
                        '& .MuiChip-deleteIcon': { color: nascarColors.archetypes[1] },
                      }}
                    />
                  )}
                </Stack>
              </Box>
            )}
          </Box>

          {/* Drivers Grid */}
          <Fade in={!isLoading} timeout={600}>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: {
                  xs: '1fr',
                  sm: viewMode === 'compact' 
                    ? 'repeat(auto-fill, minmax(280px, 1fr))' 
                    : 'repeat(auto-fill, minmax(320px, 1fr))',
                  md: viewMode === 'compact' 
                    ? 'repeat(auto-fill, minmax(280px, 1fr))' 
                    : 'repeat(auto-fill, minmax(320px, 1fr))',
                  lg: viewMode === 'compact' 
                    ? 'repeat(auto-fill, minmax(280px, 1fr))' 
                    : 'repeat(auto-fill, minmax(320px, 1fr))',
                },
                gap: viewMode === 'compact' ? 2 : 3,
                justifyItems: 'center', // Center cards in their grid areas
                minHeight: filteredDrivers.length === 0 ? '400px' : 'auto',
              }}
            >
              {filteredDrivers.length === 0 ? (
                <Box
                  sx={{
                    gridColumn: '1 / -1',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    py: 8,
                  }}
                >
                  <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
                    No drivers found
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Try adjusting your search or filters
                  </Typography>
                </Box>
              ) : (
                filteredDrivers.map((driver, index) => {
                  const cardSize = viewMode === 'compact' ? 280 : 320;
                  
                  return (
                    <Card
                      key={driver.id}
                      sx={{
                        width: cardSize,
                        height: cardSize,
                        background: 'rgba(0, 0, 0, 0.3)',
                        backdropFilter: 'blur(10px)',
                        border: driver.is_active 
                          ? `2px solid #4caf50` 
                          : `1px solid rgba(255, 255, 255, 0.1)`,
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: driver.is_active 
                            ? '0 8px 30px rgba(76, 175, 80, 0.3)'
                            : `0 8px 30px ${nascarColors.primary}40`,
                          border: driver.is_active 
                            ? `2px solid #4caf50` 
                            : `1px solid ${nascarColors.primary}`,
                        },
                      }}
                    >
                      <CardActionArea
                        onClick={() => router.push(`/driver/${driver.id}`)}
                        sx={{ height: '100%' }}
                      >
                        <CardContent sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
                          {/* Driver Image Placeholder */}
                          <Box
                            sx={{
                              width: '100%',
                              height: viewMode === 'compact' ? 140 : 160,
                              backgroundColor: 'rgba(255, 255, 255, 0.05)',
                              border: '1px dashed rgba(255, 255, 255, 0.2)',
                              borderRadius: 1,
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              justifyContent: 'center',
                              mb: 2,
                              position: 'relative',
                              overflow: 'hidden',
                            }}
                          >
                            {/* Future: Conditional image rendering */}
                            {/* {driver.image_url ? (
                              <img
                                src={driver.image_url}
                                alt={driver.name}
                                style={{
                                  width: '100%',
                                  height: '100%',
                                  objectFit: 'cover',
                                  borderRadius: '4px',
                                }}
                              />
                            ) : ( */}
                            <>
                              <Box
                                sx={{
                                  width: 48,
                                  height: 48,
                                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                  borderRadius: '50%',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  mb: 1,
                                }}
                              >
                                <PersonIcon sx={{ fontSize: 24, color: 'text.secondary' }} />
                              </Box>
                              <Typography 
                                variant="caption" 
                                color="text.secondary"
                                sx={{ textAlign: 'center', px: 1 }}
                              >
                                Photo Coming Soon
                              </Typography>
                            </>
                            {/* )} */}
                          </Box>

                          {/* Driver Info Section */}
                          <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                            {/* Driver Name & Status */}
                            <Stack direction="row" justifyContent="space-between" alignItems="flex-start" sx={{ mb: 1 }}>
                              <Typography
                                variant={viewMode === 'compact' ? 'subtitle1' : 'h6'}
                                sx={{
                                  fontWeight: 600,
                                  color: 'white',
                                  lineHeight: 1.2,
                                  display: '-webkit-box',
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                  flex: 1,
                                  pr: 1,
                                }}
                              >
                                {driver.name}
                              </Typography>
                              {driver.is_active ? (
                                <ActiveIcon sx={{ fontSize: 18, color: '#4caf50', flexShrink: 0 }} />
                              ) : (
                                <InactiveIcon sx={{ fontSize: 18, color: '#757575', flexShrink: 0 }} />
                              )}
                            </Stack>

                            {/* Stats - Compact Layout */}
                            <Stack spacing={1} sx={{ mt: 'auto' }}>
                              {/* Career Wins */}
                              <Stack direction="row" alignItems="center" justifyContent="space-between">
                                <Stack direction="row" alignItems="center" spacing={0.5}>
                                  <TrophyIcon sx={{ fontSize: 14, color: 'gold' }} />
                                  <Typography variant="caption" color="text.secondary">
                                    Wins
                                  </Typography>
                                </Stack>
                                <Typography 
                                  variant="body2" 
                                  sx={{ fontWeight: 600, color: 'white' }}
                                >
                                  {driver.total_wins}
                                </Typography>
                              </Stack>

                              {/* Career Span */}
                              <Stack direction="row" alignItems="center" justifyContent="space-between">
                                <Stack direction="row" alignItems="center" spacing={0.5}>
                                  <TimelineIcon sx={{ fontSize: 14, color: nascarColors.primary }} />
                                  <Typography variant="caption" color="text.secondary">
                                    Career
                                  </Typography>
                                </Stack>
                                <Typography 
                                  variant="caption" 
                                  sx={{ 
                                    color: 'white',
                                    fontWeight: 500,
                                  }}
                                >
                                  {driver.career_span}
                                </Typography>
                              </Stack>
                            </Stack>
                          </Box>
                        </CardContent>
                      </CardActionArea>
                    </Card>
                  );
                })
              )}
            </Box>
          </Fade>

          {/* Back to Top Button */}
          <Zoom in={showBackToTop}>
            <Fab
              onClick={scrollToTop}
              sx={{
                position: 'fixed',
                bottom: 24,
                right: 24,
                backgroundColor: nascarColors.primary,
                color: 'white',
                zIndex: 1000,
                '&:hover': {
                  backgroundColor: '#E55A2B',
                  transform: 'scale(1.1)',
                },
                transition: 'all 0.3s ease',
              }}
              aria-label="scroll back to top"
            >
              <ArrowUpIcon />
            </Fab>
          </Zoom>
        </Box>
      </Container>
    </GradientBackground>
  );
}