// src/types/api.ts
/**
 * TypeScript Type Definitions for NASCAR Analytics API
 * 
 * This file defines the exact structure of data that comes from your FastAPI backend.
 * It ensures type safety throughout your React app - TypeScript will catch errors
 * if you try to access properties that don't exist or use the wrong data types.
 * 
 * These types are based on your FastAPI backend's JSON response structures.
 */

/**
 * Driver Career Statistics
 * 
 * Represents the core performance metrics for a NASCAR driver's career.
 * This matches the structure from your backend's driver data export.
 */
export interface DriverCareerStats {
  total_wins: number;              // Career race wins
  total_races: number;             // Total races participated in
  career_avg_finish: number;       // Average finishing position across career
  career_top5_rate: number;        // Percentage of races finishing in top 5
  career_top10_rate: number;       // Percentage of races finishing in top 10
  career_win_rate: number;         // Percentage of races won
}

/**
 * Recent Performance Data
 * 
 * Represents a driver's performance in recent seasons.
 * Used to show current form vs. career averages.
 */
export interface RecentPerformance {
  recent_avg_finish: number;       // Average finish in recent seasons
  recent_wins: number;             // Wins in recent seasons
  seasons_analyzed: number;        // Number of recent seasons included
}

/**
 * Driver Archetype Information
 * 
 * Represents the machine learning cluster assignment for a driver.
 * This comes from your clustering analysis backend.
 */
export interface DriverArchetype {
  name: string;                    // Archetype name (e.g., "Elite Champions")
  color: string;                   // Hex color for visualization (#FF6B35)
  description?: string;            // Optional description of the archetype
}

/**
 * Individual Driver Data
 * 
 * Complete driver profile with all career information, stats, and archetype.
 * This is what you get when fetching a specific driver's details.
 */
export interface Driver {
  id: string;                      // URL-safe driver slug (e.g., "kyle-larson")
  name: string;                    // Full driver name (e.g., "Kyle Larson")
  is_active: boolean;              // Whether driver is currently racing
  career_span: string;             // Human-readable span (e.g., "2014-2025")
  first_season: number;            // First season in NASCAR
  last_season: number;             // Most recent season
  total_seasons: number;           // Number of seasons active
  career_stats: DriverCareerStats; // Career performance metrics
  recent_performance: RecentPerformance; // Recent performance data
  archetype?: DriverArchetype;     // Machine learning cluster assignment
}

/**
 * Driver Search Result
 * 
 * Simplified driver data for search autocomplete and results lists.
 * Contains just enough info for search suggestions without full details.
 */
export interface DriverSearchResult {
  id: string;                      // Driver slug for navigation
  name: string;                    // Full driver name
  total_wins: number;              // Career wins (for sorting)
  is_active: boolean;              // Current racing status
  career_span: string;             // Career timeframe
}

/**
 * Archetype Summary
 * 
 * Complete information about a driver archetype from clustering analysis.
 * Includes statistics and representative drivers for each cluster.
 */
export interface ArchetypeSummary {
  id: string;                      // Archetype slug (e.g., "elite-champions")
  name: string;                    // Archetype name (e.g., "Elite Champions")
  color: string;                   // Hex color for visualization (#FF6B35)
  driver_count: number;            // Number of drivers in this cluster
  characteristics: {               // Nested characteristics object
    avg_wins_per_season: number;  // Average wins per season for cluster
    avg_finish: number;            // Average finishing position
    avg_top5_rate: number;         // Average top-5 percentage
    avg_win_rate: number;          // Average win percentage
    avg_seasons: number;           // Average career length
  };
  representative_drivers: string;  // Comma-separated string of driver names
  top_drivers: string[];           // Array of top driver names
  description: string;             // Detailed description of the archetype
}

/**
 * API Response Wrappers
 * 
 * These represent the exact JSON structure returned by your FastAPI endpoints.
 * They include metadata like pagination, timestamps, and data source info.
 */

// Response for /api/drivers endpoint (driver listing with pagination)
export interface DriversListResponse {
  drivers: Driver[];               // Array of driver objects
  pagination: {
    total: number;                 // Total drivers available
    limit: number;                 // Results per page
    offset: number;                // Current page offset
    has_more: boolean;             // Whether more results exist
  };
  filters_applied: {
    active_only: boolean;          // Whether filtering to active drivers
    min_wins: number;              // Minimum wins filter applied
  };
}

// Response for /api/drivers/search endpoint (autocomplete)
export interface DriversSearchResponse {
  drivers: DriverSearchResult[];   // Matching drivers
  query: string;                   // Original search query
  total_matches: number;           // Total number of matches
}

// Response for /api/drivers/{driver-id} endpoint (individual driver)
export interface DriverDetailResponse {
  driver: Driver;                  // Complete driver information
  data_source: string;             // Source of the data
  last_updated: string;            // When data was last refreshed
}

// Response for /api/archetypes endpoint (all archetypes)
export interface ArchetypesResponse {
  archetypes: ArchetypeSummary[];  // All 6 driver archetypes
  clustering_info: {
    algorithm: string;             // Clustering algorithm used
    n_clusters: number;            // Number of clusters
    silhouette_score: number;      // Clustering quality metric
  };
  export_timestamp: string;        // When clustering was performed
}

// Response for /health endpoint (backend health check)
export interface HealthResponse {
  status: string;                  // "healthy" or error status
  timestamp?: string;              // Server timestamp
}

/**
 * API Error Response
 * 
 * Standard error format returned by FastAPI backend.
 * Helps with consistent error handling across the app.
 */
export interface ApiErrorResponse {
  detail: string;                  // Error message
  type?: string;                   // Error type/category
  code?: string;                   // Error code for programmatic handling
}

/**
 * Query Parameters
 * 
 * Type definitions for API endpoint parameters.
 * Ensures you pass the right data types to API calls.
 */

// Parameters for /api/drivers endpoint
export interface DriversQueryParams {
  limit?: number;                  // Number of results per page (default: 50)
  offset?: number;                 // Results to skip for pagination (default: 0)
  active_only?: boolean;           // Filter to active drivers only
  min_wins?: number;               // Minimum career wins required
}

// Parameters for /api/drivers/search endpoint
export interface DriverSearchParams {
  q: string;                       // Search query (driver name)
  limit?: number;                  // Max results to return (default: 10)
}

/**
 * Frontend-Only Types
 * 
 * These types are used internally in your React app but don't come from the API.
 * They help with state management and UI logic.
 */

// Loading states for API calls
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

// Search state management
export interface SearchState {
  query: string;                   // Current search input
  results: DriverSearchResult[];   // Search results
  loading: LoadingState;           // Loading status
  error: string | null;            // Error message if search failed
}

// Driver comparison state (for future comparison features)
export interface ComparisonState {
  selectedDrivers: Driver[];       // Drivers selected for comparison
  maxDrivers: number;              // Maximum drivers that can be compared
}