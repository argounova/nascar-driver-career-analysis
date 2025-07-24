// src/app/driver/[id]/page.tsx
'use client';

import React from 'react';
import { useParams, useRouter } from 'next/navigation';
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
  Divider,
  LinearProgress,
  Paper,
  IconButton,
} from '@mui/material';
import {
  ArrowBack as BackIcon,
  Speed as SpeedIcon,
  EmojiEvents as TrophyIcon,
  TrendingUp as TrendingUpIcon,
  Timeline as TimelineIcon,
  Flag as FlagIcon,
  CheckCircle as ActiveIcon,
  Cancel as InactiveIcon,
  Star as StarIcon,
} from '@mui/icons-material';
import { nascarColors, getArchetypeColor } from '@/lib/theme';
import { GradientBackground } from '@/components/ui/BackgroundVariations';
import { nascarApi } from '@/lib/services/nascarApi';
import { queryKeys } from '@/lib/queryClient';
import type { Driver } from '@/types/api';

export default function DriverProfilePage() {
  const params = useParams();
  const router = useRouter();
  const driverId = params.id as string;

  // Fetch driver data
  const {
    data: driverData,
    isLoading,
    error,
    isError,
  } = useQuery({
    queryKey: queryKeys.drivers.detail(driverId),
    queryFn: () => nascarApi.drivers.getDriverById(driverId),
    retry: 2,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  });

  const driver = driverData;

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
              Loading driver profile...
            </Typography>
          </Box>
        </Container>
      </GradientBackground>
    );
  }

  // Error state
  if (isError || !driver) {
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
              {error?.message || `Driver "${driverId}" not found.`}
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

  // Helper function to format percentages
  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // Helper function to get performance rating color
  const getPerformanceColor = (rate: number): string => {
    if (rate >= 0.4) return nascarColors.archetypes[0]; // Elite
    if (rate >= 0.3) return nascarColors.archetypes[1]; // Great
    if (rate >= 0.2) return nascarColors.archetypes[2]; // Good
    if (rate >= 0.1) return nascarColors.archetypes[3]; // Average
    return nascarColors.archetypes[4]; // Developing
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

            <Stack 
              direction={{ xs: 'column', md: 'row' }} 
              justifyContent="space-between" 
              alignItems={{ xs: 'flex-start', md: 'center' }}
              spacing={2}
            >
              <Box>
                <Typography 
                  variant="h2" 
                  sx={{ 
                    fontWeight: 800,
                    fontSize: { xs: '2rem', md: '3rem' },
                    background: `linear-gradient(135deg, ${nascarColors.primary} 0%, ${nascarColors.archetypes[1]} 100%)`,
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    mb: 1,
                  }}
                >
                  {driver.name}
                </Typography>
                <Stack direction="row" spacing={2} alignItems="center">
                  <Chip
                    icon={driver.is_active ? <ActiveIcon /> : <InactiveIcon />}
                    label={driver.is_active ? 'Active Driver' : 'Retired'}
                    color={driver.is_active ? 'success' : 'default'}
                    variant="outlined"
                  />
                  <Typography variant="h6" color="text.secondary">
                    {driver.career_span}
                  </Typography>
                </Stack>
              </Box>

              <Box sx={{ textAlign: { xs: 'left', md: 'right' } }}>
                <Typography variant="h4" color={nascarColors.primary} sx={{ fontWeight: 700 }}>
                  {driver.career_stats.total_wins}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Career Wins
                </Typography>
              </Box>
            </Stack>
          </Box>

          {/* Key Stats Overview */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid ${nascarColors.primary}40`,
              }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <TrophyIcon sx={{ fontSize: 40, color: nascarColors.primary, mb: 1 }} />
                  <Typography variant="h4" sx={{ fontWeight: 700, color: 'white' }}>
                    {driver.career_stats.total_races}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Races
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid ${getPerformanceColor(driver.career_stats.career_top5_rate)}40`,
              }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <StarIcon sx={{ fontSize: 40, color: getPerformanceColor(driver.career_stats.career_top5_rate), mb: 1 }} />
                  <Typography variant="h4" sx={{ fontWeight: 700, color: 'white' }}>
                    {formatPercentage(driver.career_stats.career_top5_rate)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Top 5 Rate
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid ${nascarColors.archetypes[2]}40`,
              }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <FlagIcon sx={{ fontSize: 40, color: nascarColors.archetypes[2], mb: 1 }} />
                  <Typography variant="h4" sx={{ fontWeight: 700, color: 'white' }}>
                    {driver.career_stats.career_avg_finish.toFixed(1)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Finish
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid ${nascarColors.archetypes[3]}40`,
              }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <TimelineIcon sx={{ fontSize: 40, color: nascarColors.archetypes[3], mb: 1 }} />
                  <Typography variant="h4" sx={{ fontWeight: 700, color: 'white' }}>
                    {driver.total_seasons}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Seasons
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Detailed Performance Analysis */}
          <Grid container spacing={3}>
            {/* Career Performance Breakdown */}
            <Grid size={{ xs: 12, md: 8 }}>
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid rgba(255, 255, 255, 0.1)`,
                mb: 3,
              }}>
                <CardContent>
                  <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                    Performance Metrics
                  </Typography>

                  <Stack spacing={3}>
                    {/* Win Rate */}
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="body1">Win Rate</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {formatPercentage(driver.career_stats.career_win_rate)}
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={driver.career_stats.career_win_rate * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: nascarColors.primary,
                            borderRadius: 4,
                          },
                        }}
                      />
                    </Box>

                    {/* Top 5 Rate */}
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="body1">Top 5 Rate</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {formatPercentage(driver.career_stats.career_top5_rate)}
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={driver.career_stats.career_top5_rate * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getPerformanceColor(driver.career_stats.career_top5_rate),
                            borderRadius: 4,
                          },
                        }}
                      />
                    </Box>

                    {/* Top 10 Rate */}
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="body1">Top 10 Rate</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {formatPercentage(driver.career_stats.career_top10_rate)}
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={driver.career_stats.career_top10_rate * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: nascarColors.archetypes[1],
                            borderRadius: 4,
                          },
                        }}
                      />
                    </Box>
                  </Stack>
                </CardContent>
              </Card>

              {/* Recent vs Career Performance */}
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid rgba(255, 255, 255, 0.1)`,
              }}>
                <CardContent>
                  <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                    Recent Performance Comparison
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Last {driver.recent_performance.seasons_analyzed} seasons vs. career average
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid size={{ xs: 6 }}>
                      <Paper sx={{ 
                        p: 2, 
                        textAlign: 'center',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: `1px solid ${nascarColors.archetypes[4]}40`,
                      }}>
                        <Typography variant="h6" color={nascarColors.archetypes[4]}>
                          Career Average
                        </Typography>
                        <Typography variant="h4" sx={{ fontWeight: 700, my: 1 }}>
                          {driver.career_stats.career_avg_finish.toFixed(1)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Average Finish
                        </Typography>
                      </Paper>
                    </Grid>

                    <Grid size={{ xs: 6 }}>
                      <Paper sx={{ 
                        p: 2, 
                        textAlign: 'center',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: `1px solid ${nascarColors.primary}40`,
                      }}>
                        <Typography variant="h6" color={nascarColors.primary}>
                          Recent Form
                        </Typography>
                        <Typography variant="h4" sx={{ fontWeight: 700, my: 1 }}>
                          {driver.recent_performance.recent_avg_finish.toFixed(1)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Average Finish
                        </Typography>
                      </Paper>
                    </Grid>

                    <Grid size={{ xs: 6 }}>
                      <Paper sx={{ 
                        p: 2, 
                        textAlign: 'center',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: `1px solid ${nascarColors.archetypes[4]}40`,
                      }}>
                        <Typography variant="h6" color={nascarColors.archetypes[4]}>
                          Career Total
                        </Typography>
                        <Typography variant="h4" sx={{ fontWeight: 700, my: 1 }}>
                          {driver.career_stats.total_wins}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Total Wins
                        </Typography>
                      </Paper>
                    </Grid>

                    <Grid size={{ xs: 6 }}>
                      <Paper sx={{ 
                        p: 2, 
                        textAlign: 'center',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: `1px solid ${nascarColors.primary}40`,
                      }}>
                        <Typography variant="h6" color={nascarColors.primary}>
                          Recent Wins
                        </Typography>
                        <Typography variant="h4" sx={{ fontWeight: 700, my: 1 }}>
                          {driver.recent_performance.recent_wins}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Last {driver.recent_performance.seasons_analyzed} Seasons
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Driver Summary Sidebar */}
            <Grid size={{ xs: 12, md: 4 }}>
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid rgba(255, 255, 255, 0.1)`,
                mb: 3,
              }}>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                    Career Highlights
                  </Typography>

                  <Stack spacing={2}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        First Season
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {driver.first_season}
                      </Typography>
                    </Box>

                    <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Last Season
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {driver.last_season}
                      </Typography>
                    </Box>

                    <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Career Length
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {driver.total_seasons} Seasons
                      </Typography>
                    </Box>

                    <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Races per Season
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {(driver.career_stats.total_races / driver.total_seasons).toFixed(1)}
                      </Typography>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>

              {/* Performance Rating */}
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid ${getPerformanceColor(driver.career_stats.career_top5_rate)}40`,
                mb: 3,
              }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Performance Grade
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography 
                      variant="h2" 
                      sx={{ 
                        fontWeight: 800,
                        color: getPerformanceColor(driver.career_stats.career_top5_rate),
                      }}
                    >
                      {driver.career_stats.career_top5_rate >= 0.4 ? 'A+' :
                       driver.career_stats.career_top5_rate >= 0.3 ? 'A' :
                       driver.career_stats.career_top5_rate >= 0.2 ? 'B+' :
                       driver.career_stats.career_top5_rate >= 0.15 ? 'B' :
                       driver.career_stats.career_top5_rate >= 0.1 ? 'C' : 'D'}
                    </Typography>
                  </Box>

                  <Typography variant="body2" color="text.secondary">
                    Based on career top-5 performance and win rate
                  </Typography>
                </CardContent>
              </Card>

              {/* Driver Archetype */}
              {driver.archetype && (
                <Card sx={{ 
                  background: 'rgba(0, 0, 0, 0.3)', 
                  backdropFilter: 'blur(10px)',
                  border: `1px solid ${driver.archetype.color}40`,
                  mb: 3,
                }}>
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                      Driver Archetype
                    </Typography>
                    
                    <Box sx={{ mb: 2 }}>
                      <Chip
                        label={driver.archetype.name}
                        sx={{
                          backgroundColor: driver.archetype.color,
                          color: 'white',
                          fontWeight: 600,
                          fontSize: '1rem',
                          padding: '4px 8px',
                          '& .MuiChip-label': {
                            padding: '0 12px',
                          }
                        }}
                      />
                    </Box>

                    {driver.archetype.description && (
                      <Typography 
                        variant="body2" 
                        color="text.secondary"
                        sx={{ 
                          fontStyle: 'italic',
                          maxWidth: 300,
                          margin: '0 auto'
                        }}
                      >
                        {driver.archetype.description}
                      </Typography>
                    )}
                    
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={() => router.push('/archetypes')}
                        sx={{
                          borderColor: driver.archetype.color,
                          color: driver.archetype.color,
                          '&:hover': {
                            borderColor: driver.archetype.color,
                            backgroundColor: `${driver.archetype.color}20`,
                          }
                        }}
                      >
                        Learn About Archetypes
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              )}

              {/* If no archetype available, show fallback */}
              {!driver.archetype && (
                <Card sx={{ 
                  background: 'rgba(0, 0, 0, 0.3)', 
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  mb: 3,
                }}>
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                      Driver Archetype
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary">
                      Archetype classification not available for this driver.
                    </Typography>
                    
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={() => router.push('/archetypes')}
                        sx={{
                          borderColor: 'rgba(255, 255, 255, 0.3)',
                          color: 'text.secondary',
                        }}
                      >
                        Learn About Archetypes
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              )}
            </Grid>
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
              Search Another Driver
            </Button>
          </Box>
        </Box>
      </Container>
    </GradientBackground>
  );
}