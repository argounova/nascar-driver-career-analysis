// src/components/providers/QueryProvider.tsx
'use client';

import React, { ReactNode } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { queryClient, cacheUtils, logQueryError } from '@/lib/queryClient';

/**
 * React Query Provider Component
 * 
 * This component wraps your entire app and provides React Query functionality
 * to all child components. It's like a "power grid" that supplies React Query
 * capabilities throughout your NASCAR analytics application.
 * 
 * Key responsibilities:
 * - Provides the query client to all components
 * - Sets up development tools for debugging
 * - Handles app-level prefetching for better performance
 * - Manages connection monitoring
 */

interface QueryProviderProps {
  children: ReactNode;
}

export default function QueryProvider({ children }: QueryProviderProps) {
  /**
   * Prefetch critical data when the app starts
   * This loads important NASCAR data in the background for better UX
   */
  React.useEffect(() => {
    const prefetchCriticalData = async () => {
      try {
        // Check if backend is reachable first
        const isConnected = await cacheUtils.checkConnection();
        
        if (isConnected) {
          console.log('🏁 Backend connected, prefetching NASCAR data...');
          
          // Prefetch popular drivers for homepage (runs in background)
          cacheUtils.prefetchPopularDrivers().catch((error) => {
            logQueryError(error, 'prefetch-popular-drivers');
          });
          
          // Prefetch archetypes for better navigation experience
          cacheUtils.prefetchArchetypes().catch((error) => {
            logQueryError(error, 'prefetch-archetypes');
          });
          
          console.log('✅ NASCAR data prefetching initiated');
        } else {
          console.warn('⚠️ Backend not reachable, skipping prefetch');
        }
      } catch (error) {
        logQueryError(error, 'app-startup-prefetch');
      }
    };

    // Run prefetching after a short delay to not block initial render
    const timeoutId = setTimeout(prefetchCriticalData, 1000);
    
    return () => clearTimeout(timeoutId);
  }, []);

  /**
   * Monitor connection status and update cache accordingly
   * This helps provide feedback when the FastAPI backend is down
   */
  React.useEffect(() => {
    const handleOnline = () => {
      console.log('🌐 Network back online, checking backend...');
      cacheUtils.checkConnection().then((isConnected) => {
        if (isConnected) {
          console.log('✅ Backend reconnected');
          // Could show a "Connection restored" toast here
        }
      });
    };

    const handleOffline = () => {
      console.log('📴 Network went offline');
      // Could show an "Offline" indicator here
    };

    // Listen for browser online/offline events
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      
      {/* React Query DevTools - only shows in development */}
      <ReactQueryDevtools 
        initialIsOpen={false}
        buttonPosition="bottom-right"
        position="bottom"
      />
    </QueryClientProvider>
  );
}

/**
 * Higher Order Component for Query Error Boundary
 * 
 * This wraps components that heavily use React Query and provides
 * centralized error handling for query failures.
 */
interface QueryErrorBoundaryProps {
  children: ReactNode;
  fallback?: (error: Error, retry: () => void) => ReactNode;
  onError?: (error: Error) => void;
}

export function QueryErrorBoundary({ 
  children, 
  fallback,
  onError 
}: QueryErrorBoundaryProps) {
  const [hasError, setHasError] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  const retry = React.useCallback(() => {
    setHasError(false);
    setError(null);
    // Clear any stale cache that might be causing issues
    queryClient.invalidateQueries();
  }, []);

  // Reset error state when children change
  React.useEffect(() => {
    if (hasError) {
      setHasError(false);
      setError(null);
    }
  }, [children, hasError]);

  // Error boundary logic
  React.useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      // Only handle React Query related errors
      if (event.error?.message?.includes('query') || event.error?.userMessage) {
        setError(event.error);
        setHasError(true);
        onError?.(event.error);
        logQueryError(event.error, 'query-error-boundary');
      }
    };

    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, [onError]);

  if (hasError && error) {
    // Custom fallback UI or default error display
    if (fallback) {
      return <>{fallback(error, retry)}</>;
    }

    // Default error UI
    return (
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center', 
        background: '#1a1a1a', 
        color: '#fff',
        borderRadius: '8px',
        margin: '1rem'
      }}>
        <h3 style={{ color: '#FF6B35', marginBottom: '1rem' }}>
          NASCAR Data Error
        </h3>
        <p style={{ marginBottom: '1rem', color: '#b3b3b3' }}>
          {(error as any).userMessage || error.message || 'Failed to load NASCAR data'}
        </p>
        <button
          onClick={retry}
          style={{
            background: '#FF6B35',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: '600'
          }}
        >
          Try Again
        </button>
      </div>
    );
  }

  return <>{children}</>;
}

/**
 * Custom Hook for Connection Status
 * 
 * Provides real-time backend connection status to components.
 * Useful for showing connection indicators in your UI.
 */
export function useConnectionStatus() {
  const [isConnected, setIsConnected] = React.useState<boolean | null>(null);
  const [isChecking, setIsChecking] = React.useState(false);

  const checkConnection = React.useCallback(async () => {
    setIsChecking(true);
    try {
      const connected = await cacheUtils.checkConnection();
      setIsConnected(connected);
      return connected;
    } catch (error) {
      setIsConnected(false);
      return false;
    } finally {
      setIsChecking(false);
    }
  }, []);

  // Check connection on mount
  React.useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  // Periodic connection checks (every 30 seconds when page is visible)
  React.useEffect(() => {
    const interval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        checkConnection();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [checkConnection]);

  return {
    isConnected,
    isChecking,
    checkConnection,
  };
}

/**
 * Development Helper Component
 * 
 * Shows useful debugging info in development mode.
 * Can be added to your app during development to monitor React Query.
 */
export function QueryDebugInfo() {
  const [cacheStats, setCacheStats] = React.useState<any>(null);

  React.useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      const interval = setInterval(() => {
        const cache = queryClient.getQueryCache();
        const queries = cache.getAll();
        
        setCacheStats({
          totalQueries: queries.length,
          successfulQueries: queries.filter(q => q.state.status === 'success').length,
          errorQueries: queries.filter(q => q.state.status === 'error').length,
          loadingQueries: queries.filter(q => q.state.status === 'pending').length,
        });
      }, 2000);

      return () => clearInterval(interval);
    }
  }, []);

  if (process.env.NODE_ENV !== 'development' || !cacheStats) {
    return null;
  }

  return (
    <div style={{
      position: 'fixed',
      top: '10px',
      left: '10px',
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '8px',
      borderRadius: '4px',
      fontSize: '12px',
      fontFamily: 'monospace',
      zIndex: 9999,
    }}>
      <div>Queries: {cacheStats.totalQueries}</div>
      <div style={{ color: '#4ade80' }}>Success: {cacheStats.successfulQueries}</div>
      <div style={{ color: '#f87171' }}>Error: {cacheStats.errorQueries}</div>
      <div style={{ color: '#fbbf24' }}>Loading: {cacheStats.loadingQueries}</div>
    </div>
  );
}