// src/lib/queryClient.ts
import { QueryClient } from '@tanstack/react-query';

/**
 * React Query Client Configuration
 * 
 * This file sets up the React Query client that will manage all API state
 * throughout your application. React Query handles:
 * - Caching API responses (so you don't re-fetch the same data)
 * - Loading states (automatic loading/error/success states)
 * - Background refetching (keeps data fresh)
 * - Optimistic updates (for future features)
 * 
 * Think of it as a smart cache that sits between your components and API calls.
 */

/**
 * Create the global query client with NASCAR-optimized settings
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // How long data stays fresh before refetching (5 minutes)
      // NASCAR data doesn't change frequently, so this is reasonable
      staleTime: 5 * 60 * 1000,
      
      // How long unused data stays in cache (10 minutes)
      // Keeps recently viewed drivers cached for quick access
      gcTime: 10 * 60 * 1000,
      
      // Retry failed requests 2 times (network issues are common)
      retry: (failureCount, error: any) => {
        // Don't retry on client errors (400-499)
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false;
        }
        // Retry up to 2 times for network/server errors
        return failureCount < 2;
      },
      
      // Don't refetch when window regains focus (prevents unnecessary calls)
      refetchOnWindowFocus: false,
      
      // Don't refetch when component remounts (use cached data)
      refetchOnMount: false,
      
      // Do refetch when network reconnects (get latest data after offline)
      refetchOnReconnect: true,
      
      // Retry delay increases each time (100ms, 200ms, 400ms)
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
    
    mutations: {
      // Retry mutations once (for future features like favoriting drivers)
      retry: (failureCount, error: any) => {
        // Don't retry client errors for mutations
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false;
        }
        // Retry once for network/server errors
        return failureCount < 1;
      },
    },
  },
});

/**
 * Query Keys Factory
 * 
 * Centralized query keys ensure consistency and make cache invalidation easier.
 * Each type of data gets its own key structure for organized caching.
 */
export const queryKeys = {
  // Health check queries
  health: {
    all: ['health'] as const,
    status: () => [...queryKeys.health.all, 'status'] as const,
  },
  
  // Driver-related queries
  drivers: {
    all: ['drivers'] as const,
    
    // Driver search queries
    search: (query: string) => [...queryKeys.drivers.all, 'search', query] as const,
    popularDrivers: () => [...queryKeys.drivers.all, 'popular'] as const,
    
    // Individual driver queries
    detail: (driverId: string) => [...queryKeys.drivers.all, 'detail', driverId] as const,
    exists: (driverId: string) => [...queryKeys.drivers.all, 'exists', driverId] as const,
    
    // Driver listing queries
    list: (params?: Record<string, any>) => [...queryKeys.drivers.all, 'list', params] as const,
    active: () => [...queryKeys.drivers.all, 'active'] as const,
    withWins: (minWins: number) => [...queryKeys.drivers.all, 'withWins', minWins] as const,
  },
  
  // Archetype-related queries
  archetypes: {
    all: ['archetypes'] as const,
    list: () => [...queryKeys.archetypes.all, 'list'] as const,
    byName: (name: string) => [...queryKeys.archetypes.all, 'byName', name] as const,
    clusteringInfo: () => [...queryKeys.archetypes.all, 'clustering'] as const,
  },
} as const;

/**
 * Cache Utility Functions
 * 
 * Helper functions to manage the React Query cache programmatically.
 * Useful for clearing cache when data updates or handling special cases.
 */
export const cacheUtils = {
  /**
   * Clear all driver-related cache
   * Useful when you know driver data has been updated on the backend
   */
  clearDriverCache: () => {
    queryClient.removeQueries({ queryKey: queryKeys.drivers.all });
  },
  
  /**
   * Clear specific driver from cache
   * @param driverId - Driver to remove from cache
   */
  clearDriverDetail: (driverId: string) => {
    queryClient.removeQueries({ queryKey: queryKeys.drivers.detail(driverId) });
  },
  
  /**
   * Clear all search results cache
   * Useful when search functionality is updated
   */
  clearSearchCache: () => {
    queryClient.removeQueries({ 
      queryKey: queryKeys.drivers.all,
      predicate: (query) => {
        return query.queryKey.includes('search');
      }
    });
  },
  
  /**
   * Prefetch popular drivers for homepage
   * Call this when app starts to load data in background
   */
  prefetchPopularDrivers: async () => {
    await queryClient.prefetchQuery({
      queryKey: queryKeys.drivers.popularDrivers(),
      queryFn: async () => {
        const { driverSearchService } = await import('./services/nascarApi');
        return driverSearchService.getPopularDrivers();
      },
      staleTime: 10 * 60 * 1000, // Cache for 10 minutes
    });
  },
  
  /**
   * Prefetch all archetypes for better UX
   * Call this early to load archetype data in background
   */
  prefetchArchetypes: async () => {
    await queryClient.prefetchQuery({
      queryKey: queryKeys.archetypes.list(),
      queryFn: async () => {
        const { archetypesService } = await import('./services/nascarApi');
        return archetypesService.getAllArchetypes();
      },
      staleTime: 30 * 60 * 1000, // Cache for 30 minutes (archetypes change rarely)
    });
  },
  
  /**
   * Check if backend is reachable and update cache accordingly
   */
  checkConnection: async () => {
    try {
      await queryClient.fetchQuery({
        queryKey: queryKeys.health.status(),
        queryFn: async () => {
          const { healthService } = await import('./services/nascarApi');
          return healthService.checkHealth();
        },
        staleTime: 30 * 1000, // Check every 30 seconds
      });
      return true;
    } catch (error) {
      console.warn('Backend connection check failed:', error);
      return false;
    }
  },
};

/**
 * Global Error Handling
 * 
 * In React Query v5, error handling is done through the QueryCache
 * rather than default options. This provides global error logging.
 */
queryClient.getQueryCache().config = {
  onError: (error: any) => {
    // Log all query errors for debugging
    console.error('React Query Error:', {
      message: error.message,
      userMessage: error.userMessage,
      stack: error.stack,
    });
    
    // Could integrate with error reporting service here
    // Example: Sentry.captureException(error);
  },
};

queryClient.getMutationCache().config = {
  onError: (error: any) => {
    console.error('React Query Mutation Error:', error);
    
    // Could show global error toast here
    // Example: toast.error(error.userMessage || 'Something went wrong');
  },
};

/**
 * Development Tools
 * 
 * Helper functions for debugging React Query during development
 */
export const devTools = {
  /**
   * Log current cache state (development only)
   */
  logCacheState: () => {
    if (process.env.NODE_ENV === 'development') {
      console.log('React Query Cache State:', queryClient.getQueryCache().getAll());
    }
  },
  
  /**
   * Clear entire cache (development only)
   */
  clearAllCache: () => {
    if (process.env.NODE_ENV === 'development') {
      queryClient.clear();
      console.log('React Query cache cleared');
    }
  },
  
  /**
   * Get cache statistics
   */
  getCacheStats: () => {
    const cache = queryClient.getQueryCache();
    const queries = cache.getAll();
    
    return {
      totalQueries: queries.length,
      successfulQueries: queries.filter(q => q.state.status === 'success').length,
      errorQueries: queries.filter(q => q.state.status === 'error').length,
      loadingQueries: queries.filter(q => q.state.status === 'pending').length,
    };
  },
};

export default queryClient;

/**
 * Error Logging Utility
 * 
 * In React Query v5, error handling is done at the hook level or through
 * custom error boundaries. This utility provides consistent error logging
 * that can be used in your components.
 */
export const logQueryError = (error: any, context?: string) => {
  console.error(`React Query Error${context ? ` (${context})` : ''}:`, {
    message: error.message,
    userMessage: (error as any).userMessage,
    status: error?.response?.status,
    stack: error.stack,
  });
  
  // Could integrate with error reporting service here
  // Example: Sentry.captureException(error, { extra: { context } });
};