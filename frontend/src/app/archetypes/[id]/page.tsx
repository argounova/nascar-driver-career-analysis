// src/app/archetypes/[id]/page.tsx
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
  IconButton,
  Paper,
  Divider,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
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
  Person as PersonIcon,
  CheckCircle as ActiveIcon,
  Cancel as InactiveIcon,
} from '@mui/icons-material';
import { nascarColors } from '@/lib/theme';
import { GradientBackground } from '@/components/ui/BackgroundVariations';
import { nascarApi } from '@/lib/services/nascarApi';
import { queryKeys } from '@/lib/queryClient';
import { ArchetypeDetailResponse } from '@/types/api';


export default function IndividualArchetypePage() {
  const params = useParams();
  const router = useRouter();
  const archetypeId = params.id as string;

  // Fetch specific archetype data
  const {
    data: response,
    isLoading,
    error,
    isError,
  } = useQuery({
    queryKey: queryKeys.archetypes.detail(archetypeId),
    queryFn: () => nascarApi.archetypes.getArchetypeById(archetypeId),
    retry: 2,
    staleTime: 15 * 60 * 1000, // Cache for 15 minutes
  });

    // Debug logging
  React.useEffect(() => {
    if (response) {
      console.log('=== ARCHETYPE DETAIL DEBUG ===');
      console.log('Full response:', response);
      console.log('Archetype data:', response.archetype);
      console.log('Drivers array:', response.archetype.top_drivers);
      console.log('Representative drivers (raw):', response.archetype?.representative_drivers);
      console.log('==============================');
    }
  }, [response]);

  // Parse representative drivers string into array
  const representativeDrivers = React.useMemo(() => {
    if (!response?.archetype?.representative_drivers) return [];
    
    return response.archetype.representative_drivers
      .split(',')
      .map(driver => driver.trim())
      .filter(driver => driver.length > 0);
  }, [response?.archetype?.representative_drivers]);

  // Helper function to format percentages
  const formatPercentage = React.useCallback((value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  }, []);

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
              Loading archetype details...
            </Typography>
          </Box>
        </Container>
      </GradientBackground>
    );
  }

  // Error state
  if (isError || !response) {
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
              {error?.message || `Archetype "${archetypeId}" not found.`}
            </Alert>
            <Button
              variant="contained"
              onClick={() => router.push('/archetypes')}
              sx={{
                backgroundColor: nascarColors.primary,
                '&:hover': { backgroundColor: '#E55A2B' },
              }}
            >
              Back to Archetypes
            </Button>
          </Box>
        </Container>
      </GradientBackground>
    );
  }

  // Extract data from response
  const archetype = response.archetype;
  const drivers = response.drivers || [];
  const archetypeColor = archetype.color || nascarColors.primary;

  return (
    <GradientBackground>
      <Container maxWidth="lg">
        <Box sx={{ py: 4 }}>
          {/* Header Section */}
          <Box sx={{ mb: 4 }}>
            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
              <IconButton
                onClick={() => router.push('/archetypes')}
                sx={{ 
                  color: 'text.secondary',
                  '&:hover': { color: archetypeColor },
                }}
              >
                <BackIcon />
              </IconButton>
              <Typography variant="caption" color="text.secondary">
                Back to All Archetypes
              </Typography>
            </Stack>

            <Stack 
              direction={{ xs: 'column', md: 'row' }} 
              justifyContent="space-between" 
              alignItems={{ xs: 'flex-start', md: 'center' }}
              spacing={2}
            >
              <Box>
                <Chip
                  label={archetype.name}
                  sx={{
                    backgroundColor: archetypeColor,
                    color: 'white',
                    fontWeight: 600,
                    fontSize: '1.1rem',
                    px: 2,
                    py: 1,
                    mb: 2,
                  }}
                />
                
                <Typography 
                  variant="h2" 
                  sx={{ 
                    fontWeight: 800,
                    fontSize: { xs: '2rem', md: '3rem' },
                    background: `linear-gradient(135deg, ${archetypeColor} 0%, ${nascarColors.archetypes[1]} 100%)`,
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    mb: 1,
                  }}
                >
                  {archetype.name}
                </Typography>
                
                <Stack direction="row" spacing={2} alignItems="center">
                  <Chip
                    icon={<GroupIcon />}
                    label={`${archetype.driver_count} Drivers`}
                    color="primary"
                    variant="outlined"
                  />
                  <Typography variant="h6" color="text.secondary">
                    Machine Learning Classification
                  </Typography>
                </Stack>
              </Box>

              <Box sx={{ textAlign: { xs: 'left', md: 'right' } }}>
                <Typography variant="h4" color={archetypeColor} sx={{ fontWeight: 700 }}>
                  {archetype.characteristics.avg_wins_per_season.toFixed(1)}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Avg Wins/Season
                </Typography>
              </Box>
            </Stack>
          </Box>

          {/* Archetype Description */}
          <Paper sx={{ 
            p: 3, 
            mb: 4,
            background: 'rgba(0, 0, 0, 0.3)', 
            backdropFilter: 'blur(10px)',
            border: `1px solid ${archetypeColor}40`,
          }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Archetype Characteristics
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.7 }}>
              {archetype.description}
            </Typography>
          </Paper>

          <Grid container spacing={3}>
            {/* Performance Metrics */}
            <Grid size={{ xs: 12, md: 8 }}>
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid rgba(255, 255, 255, 0.1)`,
                mb: 3,
              }}>
                <CardContent>
                  <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                    Performance Profile
                  </Typography>

                  <Stack spacing={3}>
                    {/* Win Rate */}
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="body1">Average Win Rate</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {formatPercentage(archetype.characteristics.avg_win_rate)}
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={archetype.characteristics.avg_win_rate * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: archetypeColor,
                            borderRadius: 4,
                          },
                        }}
                      />
                    </Box>

                    {/* Top 5 Rate */}
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="body1">Average Top 5 Rate</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {formatPercentage(archetype.characteristics.avg_top5_rate)}
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={archetype.characteristics.avg_top5_rate * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: archetypeColor,
                            borderRadius: 4,
                            opacity: 0.8,
                          },
                        }}
                      />
                    </Box>

                    {/* Average Finish */}
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="body1">Average Finish Position</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {archetype.characteristics.avg_finish.toFixed(1)}
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={Math.max(0, (43 - archetype.characteristics.avg_finish) / 43 * 100)}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: archetypeColor,
                            borderRadius: 4,
                            opacity: 0.7,
                          },
                        }}
                      />
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                        Lower is better (1st place = best)
                      </Typography>
                    </Box>

                    {/* Career Length */}
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="body1">Average Career Length</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {archetype.characteristics.avg_seasons.toFixed(1)} seasons
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={Math.min(100, (archetype.characteristics.avg_seasons / 25 * 100))}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: archetypeColor,
                            borderRadius: 4,
                            opacity: 0.6,
                          },
                        }}
                      />
                    </Box>
                  </Stack>
                </CardContent>
              </Card>

              {/* Key Statistics Summary */}
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid rgba(255, 255, 255, 0.1)`,
              }}>
                <CardContent>
                  <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                    Key Statistics
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid size={{ xs: 6, md: 3 }}>
                      <Paper sx={{ 
                        p: 2, 
                        textAlign: 'center',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: `1px solid ${archetypeColor}40`,
                      }}>
                        <TrophyIcon sx={{ fontSize: 30, color: archetypeColor, mb: 1 }} />
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {archetype.characteristics.avg_wins_per_season.toFixed(1)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Wins/Season
                        </Typography>
                      </Paper>
                    </Grid>

                    <Grid size={{ xs: 6, md: 3 }}>
                      <Paper sx={{ 
                        p: 2, 
                        textAlign: 'center',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: `1px solid ${archetypeColor}40`,
                      }}>
                        <StarIcon sx={{ fontSize: 30, color: archetypeColor, mb: 1 }} />
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {formatPercentage(archetype.characteristics.avg_top5_rate)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Top 5 Rate
                        </Typography>
                      </Paper>
                    </Grid>

                    <Grid size={{ xs: 6, md: 3 }}>
                      <Paper sx={{ 
                        p: 2, 
                        textAlign: 'center',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: `1px solid ${archetypeColor}40`,
                      }}>
                        <SpeedIcon sx={{ fontSize: 30, color: archetypeColor, mb: 1 }} />
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {archetype.characteristics.avg_finish.toFixed(1)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Avg Finish
                        </Typography>
                      </Paper>
                    </Grid>

                    <Grid size={{ xs: 6, md: 3 }}>
                      <Paper sx={{ 
                        p: 2, 
                        textAlign: 'center',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: `1px solid ${archetypeColor}40`,
                      }}>
                        <TimelineIcon sx={{ fontSize: 30, color: archetypeColor, mb: 1 }} />
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {archetype.characteristics.avg_seasons.toFixed(1)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Avg Seasons
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Drivers List */}
            <Grid size={{ xs: 12, md: 4 }}>
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid rgba(255, 255, 255, 0.1)`,
                mb: 3,
              }}>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Representative Drivers
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Notable examples of this archetype:
                  </Typography>
                  
                  <Stack spacing={1}>
                    {(() => {
                      // representative_drivers is a string, so split it into an array
                      const driversString = archetype.representative_drivers || '';
                            
                      const drivers = driversString.split(',').map(d => d.trim()).filter(d => d.length > 0);
                      
                      return drivers.map((driver, index) => (
                        <Chip
                          key={index}
                          label={driver}
                          variant="outlined"
                          sx={{
                            borderColor: archetypeColor,
                            color: 'white',
                            justifyContent: 'flex-start',
                          }}
                        />
                      ));
                    })()}
                  </Stack>
                </CardContent>
              </Card>

              {/* All Drivers in Archetype */}
              <Card sx={{ 
                background: 'rgba(0, 0, 0, 0.3)', 
                backdropFilter: 'blur(10px)',
                border: `1px solid rgba(255, 255, 255, 0.1)`,
                maxHeight: 500,
                overflow: 'hidden',
              }}>
                <CardContent sx={{ pb: 0 }}>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    All Drivers ({archetype.driver_count})
                  </Typography>
                  <Stack justifyContent="space-between" sx={{   mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Sorted by career wins
                    </Typography>
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Typography variant="body2" color="text.secondary">
                        Active:
                      </Typography>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <ActiveIcon sx={{ fontSize: 16, color: '#4caf50' }} />
                        <Typography variant="caption" color="text.secondary">
                          Active
                        </Typography>
                      </Stack>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <InactiveIcon sx={{ fontSize: 16, color: '#757575' }} />
                        <Typography variant="caption" color="text.secondary">
                          Inactive
                        </Typography>
                      </Stack>
                    </Stack>
                  </Stack>

                </CardContent>
                
                <Box sx={{ maxHeight: 350, overflow: 'auto' }}>
                  <List dense>
                    {archetype.archetype_drivers.map((driver, index) => (
                      <ListItem key={driver.id} disablePadding>
                        <ListItemButton 
                          onClick={() => router.push(`/driver/${driver.id}`)}
                          sx={{
                            '&:hover': {
                              backgroundColor: `${archetypeColor}20`,
                            },
                          }}
                        >
                          <PersonIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 20 }} />
                          <ListItemText
                            primary={
                              <Stack direction="row" justifyContent="space-between" alignItems="center">
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                  {driver.name}
                                </Typography>
                                <Stack direction="row" spacing={1} alignItems="center">
                                  <Typography variant="caption" color="text.secondary">
                                    {driver.total_wins} wins
                                  </Typography>
                                  {driver.is_active ? (
                                    <ActiveIcon sx={{ fontSize: 14, color: 'success.main' }} />
                                  ) : (
                                    <InactiveIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
                                  )}
                                </Stack>
                              </Stack>
                            }
                            secondary={
                              <Typography variant="caption" color="text.secondary">
                                {driver.career_avg_finish.toFixed(1)} avg finish • {driver.total_seasons} seasons
                              </Typography>
                            }
                          />
                        </ListItemButton>
                      </ListItem>
                    ))}
                    {drivers.length > 20 && (
                      <ListItem>
                        <ListItemText
                          primary={
                            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                              +{drivers.length - 20} more drivers...
                            </Typography>
                          }
                        />
                      </ListItem>
                    )}
                  </List>
                </Box>
              </Card>
            </Grid>
          </Grid>

          {/* Navigation Buttons */}
          <Box sx={{ textAlign: 'center', mt: 6 }}>
            <Stack direction="row" spacing={2} justifyContent="center">
              <Button
                variant="outlined"
                size="large"
                onClick={() => router.push('/archetypes')}
                sx={{
                  borderColor: archetypeColor,
                  color: archetypeColor,
                  px: 4,
                  py: 1.5,
                  '&:hover': {
                    backgroundColor: `${archetypeColor}20`,
                    borderColor: archetypeColor,
                  },
                }}
              >
                All Archetypes
              </Button>
              <Button
                variant="contained"
                size="large"
                onClick={() => router.push('/')}
                sx={{
                  backgroundColor: nascarColors.primary,
                  px: 4,
                  py: 1.5,
                  '&:hover': {
                    backgroundColor: '#E55A2B',
                  },
                }}
              >
                Search Drivers
              </Button>
            </Stack>
          </Box>
        </Box>
      </Container>
    </GradientBackground>
  );
}