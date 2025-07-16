// src/app/archetypes/page.tsx
'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import {
  Container,
  Typography,
  Box,
  Grid2 as Grid,
  Card,
  CardContent,
  Stack,
  Chip,
  Button,
  CircularProgress,
  Alert,
  IconButton,
  Paper,
  Divider,
  LinearProgress,
} from '@mui/material';
import {
  ArrowBack as BackIcon,
  Psychology as BrainIcon,
  Group as GroupIcon,
  EmojiEvents as TrophyIcon,
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Star as StarIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';
import { nascarColors } from '@/lib/theme';
import { GradientBackground } from '@/components/ui/BackgroundVariations';
import { nascarApi } from '@/lib/services/nascarApi';
import { queryKeys } from '@/lib/queryClient';
import type { ArchetypeSummary } from '@/types/api';

export default function ArchetypeExplorerPage() {
  const router = useRouter();

  // Fetch all archetypes
  const {
    data: archetypesResponse,
    isLoading,
    error,
    isError,
  } = useQuery({
    queryKey: queryKeys.archetypes.list(),
    queryFn: () => nascarApi.archetypes.getAllArchetypes(),
    staleTime: 30 * 60 * 1000, // Cache for 30 minutes
    retry: 2,
  });

  // Extract archetypes array from response
  const archetypesData = archetypesResponse || [];

  // Debug logging
  React.useEffect(() => {
    if (archetypesData.length > 0) {
      console.log('=== ARCHETYPE DEBUG INFO ===');
      console.log('Total archetypes:', archetypesData.length);
      console.log('First archetype structure:', archetypesData[0]);
      console.log('All archetype names:', archetypesData.map(a => a?.name));
      console.log('Sample archetype keys:', Object.keys(archetypesData[0] || {}));
      console.log('============================');
    }
  }, [archetypesData]);

  // Loading state
  if (isLoading) {
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
            }}
          >
            <CircularProgress 
              size={60} 
              sx={{ color: nascarColors.primary, mb: 2 }} 
            />
            <Typography variant="h6" color="text.secondary">
              Loading driver archetypes...
            </Typography>
          </Box>
        </Container>
      </GradientBackground>
    );
  }

  // Error state
  if (isError || !archetypesData) {
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
            }}
          >
            <Alert 
              severity="error" 
              sx={{ mb: 3, maxWidth: 500 }}
            >
              {error?.message || 'Failed to load driver archetypes.'}
            </Alert>
            <Button
              variant="contained"
              onClick={() => router.push('/')}
              sx={{
                backgroundColor: nascarColors.primary,
                '&:hover': { backgroundColor: '#E55A2B' },
              }}
            >
              Back to Search
            </Button>
          </Box>
        </Container>
      </GradientBackground>
    );
  }

  // Helper function to get archetype color
  const getArchetypeColor = (archetypeName: string): string => {
    const colorMap: Record<string, string> = {
      'Dominant Champions': nascarColors.archetypes[0],
      'Elite Competitors': nascarColors.archetypes[1],
      'Consistent Veterans': nascarColors.archetypes[2],
      'Solid Performers': nascarColors.archetypes[3],
      'Developing Talents': nascarColors.archetypes[4],
      'Journey Drivers': nascarColors.archetypes[5],
    };
    return colorMap[archetypeName] || nascarColors.primary;
  };

  // Helper function to format percentages
  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // Helper function to get archetype description
  const getArchetypeDescription = (archetype: ArchetypeSummary): string => {
    const descriptions: Record<string, string> = {
      'Dominant Champions': 'Elite drivers with exceptional win rates and consistent top-5 performances. These are the legends of NASCAR.',
      'Elite Competitors': 'Highly successful drivers with strong win rates and competitive averages. Regular contenders for championships.',
      'Consistent Veterans': 'Reliable drivers with solid performance records and long careers. The backbone of competitive NASCAR racing.',
      'Solid Performers': 'Dependable drivers with moderate success and steady careers. Consistent points scorers and occasional winners.',
      'Developing Talents': 'Emerging drivers building their careers with potential for growth. Future stars in development.',
      'Journey Drivers': 'Drivers focused on participation and improvement. Every driver\'s journey contributes to NASCAR\'s rich history.',
    };
    return descriptions[archetype.name] || 'Driver classification based on machine learning analysis of career patterns.';
  };

  return (
    <GradientBackground>
      <Container maxWidth="lg">
        <Box sx={{ py: 4 }}>
          {/* Header Section */}
          <Box sx={{ mb: 4 }}>
            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
              <IconButton
                onClick={() => router.push('/')}
                sx={{ 
                  color: 'text.secondary',
                  '&:hover': { color: nascarColors.primary },
                }}
              >
                <BackIcon />
              </IconButton>
              <Typography variant="caption" color="text.secondary">
                Back to Search
              </Typography>
            </Stack>

            <Stack direction="row" spacing={3} alignItems="center" sx={{ mb: 3 }}>
              <BrainIcon sx={{ fontSize: 60, color: nascarColors.primary }} />
              <AnalyticsIcon sx={{ fontSize: 60, color: nascarColors.archetypes[1] }} />
            </Stack>
            
            <Typography 
              variant="h2" 
              sx={{ 
                fontWeight: 800,
                fontSize: { xs: '2rem', md: '3rem' },
                background: `linear-gradient(135deg, ${nascarColors.primary} 0%, ${nascarColors.archetypes[1]} 100%)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 2,
              }}
            >
              Driver Archetypes
            </Typography>
            
            <Typography 
              variant="h5" 
              color="text.secondary" 
              sx={{ 
                mb: 2,
                fontWeight: 300,
                opacity: 0.9,
              }}
            >
              Machine Learning Classification of NASCAR Career Patterns
            </Typography>
            
            <Typography 
              variant="body1" 
              color="text.secondary" 
              sx={{ 
                opacity: 0.7,
                maxWidth: 800,
              }}
            >
              Using advanced clustering analysis on 76 years of NASCAR data, we've identified 6 distinct driver archetypes based on 
              career performance patterns, win rates, consistency, and trajectory analysis.
            </Typography>
          </Box>

          {/* Archetype Overview Stats */}
          <Paper sx={{ 
            p: 3, 
            mb: 4,
            background: 'rgba(0, 0, 0, 0.3)', 
            backdropFilter: 'blur(10px)',
            border: `1px solid rgba(255, 255, 255, 0.1)`,
          }}>
            <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
              Clustering Analysis Overview
            </Typography>
            
            <Grid container spacing={3}>
              <Grid size={{ xs: 6, md: 3 }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: nascarColors.primary }}>
                    {archetypesData.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Archetypes Identified
                  </Typography>
                </Box>
              </Grid>
              
              <Grid size={{ xs: 6, md: 3 }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: nascarColors.archetypes[1] }}>
                    {archetypesData.reduce((sum, arch) => sum + (arch.driver_count ?? 0), 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Drivers Analyzed
                  </Typography>
                </Box>
              </Grid>
              
              <Grid size={{ xs: 6, md: 3 }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: nascarColors.archetypes[2] }}>
                    K-Means
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    ML Algorithm
                  </Typography>
                </Box>
              </Grid>
              
              <Grid size={{ xs: 6, md: 3 }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: nascarColors.archetypes[3] }}>
                    76
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Years of Data
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>

          {/* Archetype Cards */}
          <Grid container spacing={3}>
            {archetypesData.map((archetype, index) => {
              // Debug logging and safety checks
              if (!archetype) {
                console.warn(`Archetype at index ${index} is null/undefined`);
                return null;
              }
              
              // Log the structure for debugging
              if (index === 0) {
                console.log('First archetype structure:', archetype);
              }
              
              return (
                <Grid size={{ xs: 12, md: 6 }} key={archetype.id || index}>
                  <Card sx={{ 
                    height: '100%',
                    background: 'rgba(0, 0, 0, 0.3)', 
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${getArchetypeColor(archetype.name || '')}40`,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: `0 8px 30px ${getArchetypeColor(archetype.name || '')}40`,
                      border: `1px solid ${getArchetypeColor(archetype.name || '')}60`,
                    },
                  }}>
                    <CardContent sx={{ p: 3 }}>
                      {/* Archetype Header */}
                      <Stack direction="row" justifyContent="space-between" alignItems="flex-start" sx={{ mb: 2 }}>
                        <Box>
                          <Chip
                            label={archetype.name || 'Unknown Archetype'}
                            sx={{
                              backgroundColor: getArchetypeColor(archetype.name || ''),
                              color: 'white',
                              fontWeight: 600,
                              fontSize: '0.9rem',
                              mb: 1,
                            }}
                          />
                          <Stack direction="row" spacing={1} alignItems="center">
                            <GroupIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
                            <Typography variant="body2" color="text.secondary">
                              {archetype.driver_count || 0} drivers
                            </Typography>
                          </Stack>
                        </Box>
                        <Box sx={{ textAlign: 'right' }}>
                          <Typography variant="h5" sx={{ fontWeight: 700, color: getArchetypeColor(archetype.name || '') }}>
                            {(archetype.characteristics?.avg_wins_per_season ?? 0).toFixed(1)}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Wins/Season
                          </Typography>
                        </Box>
                      </Stack>

                      {/* Description */}
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 3, lineHeight: 1.6 }}>
                        {getArchetypeDescription(archetype)}
                      </Typography>

                      {/* Key Metrics */}
                      <Box sx={{ mb: 3 }}>
                        <Stack spacing={2}>
                          {/* Win Rate */}
                          <Box>
                            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                              <Typography variant="body2">Win Rate</Typography>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {formatPercentage(archetype.characteristics?.avg_win_rate ?? 0)}
                              </Typography>
                            </Stack>
                            <LinearProgress
                              variant="determinate"
                              value={(archetype.characteristics?.avg_win_rate ?? 0) * 100}
                              sx={{
                                height: 6,
                                borderRadius: 3,
                                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                '& .MuiLinearProgress-bar': {
                                  backgroundColor: getArchetypeColor(archetype.name || ''),
                                  borderRadius: 3,
                                },
                              }}
                            />
                          </Box>

                          {/* Top 5 Rate */}
                          <Box>
                            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                              <Typography variant="body2">Top 5 Rate</Typography>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {formatPercentage(archetype.characteristics?.avg_top5_rate ?? 0)}
                              </Typography>
                            </Stack>
                            <LinearProgress
                              variant="determinate"
                              value={(archetype.characteristics?.avg_top5_rate ?? 0) * 100}
                              sx={{
                                height: 6,
                                borderRadius: 3,
                                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                '& .MuiLinearProgress-bar': {
                                  backgroundColor: getArchetypeColor(archetype.name || ''),
                                  borderRadius: 3,
                                  opacity: 0.8,
                                },
                              }}
                            />
                          </Box>
                        </Stack>
                      </Box>

                      {/* Additional Stats */}
                      <Grid container spacing={2} sx={{ mb: 3 }}>
                        <Grid size={{ xs: 6 }}>
                          <Box sx={{ textAlign: 'center', p: 1 }}>
                            <Typography variant="h6" sx={{ fontWeight: 600, color: 'white' }}>
                              {(archetype.characteristics?.avg_finish ?? 0).toFixed(1)}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Avg Finish
                            </Typography>
                          </Box>
                        </Grid>
                        <Grid size={{ xs: 6 }}>
                          <Box sx={{ textAlign: 'center', p: 1 }}>
                            <Typography variant="h6" sx={{ fontWeight: 600, color: 'white' }}>
                              {(archetype.characteristics?.avg_seasons ?? 0).toFixed(1)}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Avg Seasons
                            </Typography>
                          </Box>
                        </Grid>
                      </Grid>

                      {/* Representative Drivers */}
                      <Box sx={{ mb: 3 }}>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          Representative Drivers:
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 0.5 }}>
                          {(() => {
                            // representative_drivers is a string, so split it into an array
                            const driversString = archetype.representative_drivers || '';
                            const drivers = driversString.split(',').map(d => d.trim()).filter(d => d.length > 0);
                            
                            return drivers.slice(0, 3).map((driver, idx) => (
                              <Chip
                                key={idx}
                                label={driver}
                                size="small"
                                variant="outlined"
                                sx={{
                                  borderColor: getArchetypeColor(archetype.name || ''),
                                  color: 'white',
                                  fontSize: '0.75rem',
                                }}
                              />
                            ));
                          })()}
                        </Stack>
                      </Box>

                      {/* Explore Button */}
                      <Button
                        fullWidth
                        variant="outlined"
                        onClick={() => router.push(`/archetypes/${archetype.id || ''}`)}
                        sx={{
                          borderColor: getArchetypeColor(archetype.name || ''),
                          color: getArchetypeColor(archetype.name || ''),
                          fontWeight: 600,
                          '&:hover': {
                            backgroundColor: `${getArchetypeColor(archetype.name || '')}20`,
                            borderColor: getArchetypeColor(archetype.name || ''),
                          },
                        }}
                      >
                        Explore {archetype.name || 'Archetype'}
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              );
            }).filter(Boolean)}
          </Grid>

          {/* Back to Search Button */}
          <Box sx={{ textAlign: 'center', mt: 6 }}>
            <Button
              variant="outlined"
              size="large"
              onClick={() => router.push('/')}
              sx={{
                borderColor: nascarColors.primary,
                color: nascarColors.primary,
                px: 4,
                py: 1.5,
                '&:hover': {
                  backgroundColor: `${nascarColors.primary}20`,
                  borderColor: nascarColors.primary,
                },
              }}
            >
              Back to Driver Search
            </Button>
          </Box>
        </Box>
      </Container>
    </GradientBackground>
  );
}