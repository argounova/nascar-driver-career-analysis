// src/components/layout/NavBar.tsx
'use client';

import React from 'react';
import { useRouter, usePathname } from 'next/navigation';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  Stack,
  IconButton,
  useMediaQuery,
  useTheme,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Speed as NascarIcon,
  Menu as MenuIcon,
  Home as HomeIcon,
  Psychology as ArchetypeIcon,
  Person as DriverIcon,
  Info as AboutIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { nascarColors } from '@/lib/theme';

interface NavItem {
  label: string;
  path: string;
  icon?: React.ReactNode;
}

const navigation: NavItem[] = [
  { label: 'Home', path: '/', icon: <HomeIcon sx={{ fontSize: 20 }} /> },
  { label: 'Drivers', path: '/drivers', icon: <DriverIcon sx={{ fontSize: 20 }} /> },
  { label: 'Archetypes', path: '/archetypes', icon: <ArchetypeIcon sx={{ fontSize: 20 }} /> },
  { label: 'About', path: '/about', icon: <AboutIcon sx={{ fontSize: 20 }} /> },
];

export default function NavBar() {
  const router = useRouter();
  const pathname = usePathname();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = React.useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleNavigation = (path: string) => {
    router.push(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const isActivePath = (path: string) => {
    if (path === '/') {
      return pathname === '/';
    }
    return pathname.startsWith(path);
  };

  // Mobile drawer content
  const drawer = (
    <Box sx={{ width: 280, height: '100%', backgroundColor: 'rgba(0, 0, 0, 0.95)' }}>
      <Box sx={{ p: 2, borderBottom: `1px solid rgba(255, 255, 255, 0.1)` }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Stack direction="row" alignItems="center" spacing={1}>
            <NascarIcon sx={{ color: nascarColors.primary, fontSize: 28 }} />
            <Typography
              variant="h5"
              sx={{
                fontWeight: 700,
                color: 'white',
                background: `linear-gradient(45deg, ${nascarColors.primary}, #ff8659)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              NASCAR
            </Typography>
          </Stack>
          <IconButton
            onClick={handleDrawerToggle}
            sx={{ color: 'text.secondary' }}
          >
            <CloseIcon />
          </IconButton>
        </Stack>
      </Box>
      
      <List sx={{ px: 1, py: 2 }}>
        {navigation.map((item) => (
          <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              sx={{
                borderRadius: 2,
                py: 1.5,
                px: 2,
                backgroundColor: isActivePath(item.path) 
                  ? `${nascarColors.primary}20` 
                  : 'transparent',
                borderLeft: isActivePath(item.path) 
                  ? `4px solid ${nascarColors.primary}` 
                  : '4px solid transparent',
                '&:hover': {
                  backgroundColor: `${nascarColors.primary}10`,
                },
              }}
            >
              <Stack direction="row" alignItems="center" spacing={2}>
                <Box sx={{ 
                  color: isActivePath(item.path) ? nascarColors.primary : 'text.secondary',
                  display: 'flex',
                  alignItems: 'center',
                }}>
                  {item.icon}
                </Box>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{
                    fontWeight: isActivePath(item.path) ? 600 : 400,
                    color: isActivePath(item.path) ? nascarColors.primary : 'text.primary',
                  }}
                />
              </Stack>
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <>
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid rgba(255, 255, 255, 0.1)`,
          zIndex: theme.zIndex.appBar,
        }}
      >
        <Container maxWidth="lg">
          <Toolbar sx={{ px: { xs: 0, sm: 2 } }}>
            {/* Logo/Brand */}
            {/* <Stack
              direction="row"
              alignItems="center"
              spacing={1}
              onClick={() => handleNavigation('/')}
              sx={{ 
                cursor: 'pointer',
                flexGrow: isMobile ? 1 : 0,
                '&:hover': {
                  '& .nascar-text': {
                    background: `linear-gradient(45deg, #ff8659, ${nascarColors.primary})`,
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                  }
                }
              }}
            >
              <NascarIcon 
                sx={{ 
                  color: nascarColors.primary, 
                  fontSize: { xs: 28, md: 32 },
                  transition: 'transform 0.2s ease',
                  '&:hover': {
                    transform: 'scale(1.1)',
                  }
                }} 
              />
              <Typography
                variant="h5"
                className="nascar-text"
                sx={{
                  fontWeight: 700,
                  fontSize: { xs: '1.4rem', md: '1.5rem' },
                  color: 'white',
                  background: `linear-gradient(45deg, ${nascarColors.primary}, #ff8659)`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  transition: 'all 0.3s ease',
                  letterSpacing: '0.5px',
                }}
              >
                NASCAR
              </Typography>
            </Stack> */}

            {/* Desktop Navigation */}
            {!isMobile && (
              <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', ml: 4 }}>
                <Stack direction="row" spacing={1}>
                  {navigation.map((item) => (
                    <Button
                      key={item.path}
                      onClick={() => handleNavigation(item.path)}
                      startIcon={item.icon}
                      sx={{
                        color: isActivePath(item.path) ? nascarColors.primary : 'text.secondary',
                        fontWeight: isActivePath(item.path) ? 600 : 400,
                        px: 2,
                        py: 1,
                        borderRadius: 2,
                        textTransform: 'none',
                        fontSize: '0.95rem',
                        position: 'relative',
                        '&:hover': {
                          color: nascarColors.primary,
                          backgroundColor: `${nascarColors.primary}10`,
                        },
                        '&::after': isActivePath(item.path) ? {
                          content: '""',
                          position: 'absolute',
                          bottom: 0,
                          left: '50%',
                          transform: 'translateX(-50%)',
                          width: '60%',
                          height: '2px',
                          backgroundColor: nascarColors.primary,
                          borderRadius: '1px',
                        } : {},
                      }}
                    >
                      {item.label}
                    </Button>
                  ))}
                </Stack>
              </Box>
            )}

            {/* Mobile Menu Button */}
            {isMobile && (
              <IconButton
                color="inherit"
                aria-label="open drawer"
                edge="start"
                onClick={handleDrawerToggle}
                sx={{ 
                  color: 'text.secondary',
                  '&:hover': {
                    color: nascarColors.primary,
                  }
                }}
              >
                <MenuIcon />
              </IconButton>
            )}
          </Toolbar>
        </Container>
      </AppBar>

      {/* Mobile Drawer */}
      <Box component="nav">
        <Drawer
          variant="temporary"
          anchor="right"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: 280,
              backgroundColor: 'rgba(0, 0, 0, 0.95)',
              backdropFilter: 'blur(20px)',
              border: 'none',
            },
          }}
        >
          {drawer}
        </Drawer>
      </Box>
    </>
  );
}