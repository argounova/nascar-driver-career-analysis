// src/app/about/page.tsx
'use client';

import React from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Stack,
  Chip,
  Divider,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  Psychology as BrainIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { nascarColors } from '@/lib/theme';
import { GradientBackground } from '@/components/ui/BackgroundVariations';

export default function AboutPage() {
  return (
    <GradientBackground>
      <Container maxWidth="md">
        <Box sx={{ py: 6 }}>
          {/* Header */}
          <Box sx={{ textAlign: 'center', mb: 6 }}>
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
              About NASCAR Analytics
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
              Advanced driver performance analysis powered by machine learning and decades of racing data
            </Typography>
          </Box>

          {/* Main Content */}
          <Stack spacing={4}>
            {/* Overview Card */}
            <Card sx={{ 
              background: 'rgba(0, 0, 0, 0.3)', 
              backdropFilter: 'blur(10px)',
              border: `1px solid rgba(255, 255, 255, 0.1)`,
            }}>
              <CardContent sx={{ p: 4 }}>
                <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
                  <AnalyticsIcon sx={{ color: nascarColors.primary, fontSize: 32 }} />
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    Project Overview
                  </Typography>
                </Stack>
                
                <Typography variant="body1" color="text.secondary" sx={{ mb: 3, lineHeight: 1.7 }}>
                  This NASCAR Driver Analytics platform combines comprehensive racing statistics with advanced machine learning 
                  techniques to provide deep insights into driver performance patterns, career trajectories, and racing archetypes.
                </Typography>

                <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                  Built using modern web technologies and powered by decades of NASCAR Cup Series data, 
                  our platform offers interactive visualizations and data-driven analysis to help fans, 
                  analysts, and researchers better understand the sport of NASCAR racing.
                </Typography>
              </CardContent>
            </Card>

            {/* Features Grid */}
            <Stack spacing={3}>
              <Typography variant="h5" sx={{ fontWeight: 600, textAlign: 'center' }}>
                Key Features
              </Typography>
              
              <Stack direction={{ xs: 'column', md: 'row' }} spacing={3}>
                <Card sx={{ 
                  flex: 1,
                  background: 'rgba(0, 0, 0, 0.3)', 
                  backdropFilter: 'blur(10px)',
                  border: `1px solid ${nascarColors.archetypes[1]}40`,
                }}>
                  <CardContent sx={{ p: 3 }}>
                    <Stack alignItems="center" spacing={2} sx={{ textAlign: 'center' }}>
                      <BrainIcon sx={{ color: nascarColors.archetypes[1], fontSize: 40 }} />
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Driver Archetypes
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Machine learning classification of drivers into distinct performance archetypes
                      </Typography>
                    </Stack>
                  </CardContent>
                </Card>

                <Card sx={{ 
                  flex: 1,
                  background: 'rgba(0, 0, 0, 0.3)', 
                  backdropFilter: 'blur(10px)',
                  border: `1px solid ${nascarColors.archetypes[2]}40`,
                }}>
                  <CardContent sx={{ p: 3 }}>
                    <Stack alignItems="center" spacing={2} sx={{ textAlign: 'center' }}>
                      <TimelineIcon sx={{ color: nascarColors.archetypes[2], fontSize: 40 }} />
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Career Analysis
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Comprehensive career statistics and performance trends over time
                      </Typography>
                    </Stack>
                  </CardContent>
                </Card>

                <Card sx={{ 
                  flex: 1,
                  background: 'rgba(0, 0, 0, 0.3)', 
                  backdropFilter: 'blur(10px)',
                  border: `1px solid ${nascarColors.primary}40`,
                }}>
                  <CardContent sx={{ p: 3 }}>
                    <Stack alignItems="center" spacing={2} sx={{ textAlign: 'center' }}>
                      <SpeedIcon sx={{ color: nascarColors.primary, fontSize: 40 }} />
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Performance Metrics
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Advanced statistics including win rates, average finish, and consistency measures
                      </Typography>
                    </Stack>
                  </CardContent>
                </Card>
              </Stack>
            </Stack>

            {/* Tech Stack */}
            <Card sx={{ 
              background: 'rgba(0, 0, 0, 0.3)', 
              backdropFilter: 'blur(10px)',
              border: `1px solid rgba(255, 255, 255, 0.1)`,
            }}>
              <CardContent sx={{ p: 4 }}>
                <Typography variant="h5" sx={{ fontWeight: 600, mb: 3 }}>
                  Technology Stack
                </Typography>
                
                <Typography variant="body1" color="text.secondary" sx={{ mb: 3, lineHeight: 1.7 }}>
                  This application is built with cutting-edge technologies to ensure performance, 
                  scalability, and an exceptional user experience.
                </Typography>

                <Divider sx={{ my: 3, borderColor: 'rgba(255, 255, 255, 0.1)' }} />

                <Stack spacing={3}>
                  <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1, color: nascarColors.primary }}>
                      Frontend
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
                      {['Next.js 15', 'React 18', 'TypeScript', 'Material-UI', 'React Query'].map((tech) => (
                        <Chip
                          key={tech}
                          label={tech}
                          size="small"
                          variant="outlined"
                          sx={{
                            borderColor: nascarColors.primary,
                            color: 'white',
                          }}
                        />
                      ))}
                    </Stack>
                  </Box>

                  <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1, color: nascarColors.archetypes[1] }}>
                      Backend & Analysis
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
                      {['Python', 'FastAPI', 'Pandas', 'Scikit-learn', 'Machine Learning'].map((tech) => (
                        <Chip
                          key={tech}
                          label={tech}
                          size="small"
                          variant="outlined"
                          sx={{
                            borderColor: nascarColors.archetypes[1],
                            color: 'white',
                          }}
                        />
                      ))}
                    </Stack>
                  </Box>

                  <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1, color: nascarColors.archetypes[2] }}>
                      Data Sources
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
                      {['NASCAR Cup Series', 'Historical Race Data', 'Driver Statistics', 'Performance Metrics'].map((source) => (
                        <Chip
                          key={source}
                          label={source}
                          size="small"
                          variant="outlined"
                          sx={{
                            borderColor: nascarColors.archetypes[2],
                            color: 'white',
                          }}
                        />
                      ))}
                    </Stack>
                  </Box>
                </Stack>
              </CardContent>
            </Card>

            {/* Coming Soon */}
            <Card sx={{ 
              background: 'rgba(0, 0, 0, 0.3)', 
              backdropFilter: 'blur(10px)',
              border: `1px solid ${nascarColors.warning}40`,
              textAlign: 'center',
            }}>
              <CardContent sx={{ p: 4 }}>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, color: nascarColors.warning }}>
                  🚧 More Content Coming Soon
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  This about page will be expanded with more detailed information about the project, 
                  methodology, data sources, and team behind NASCAR Analytics.
                </Typography>
              </CardContent>
            </Card>
          </Stack>
        </Box>
      </Container>
    </GradientBackground>
  );
}