// src/lib/services/nascarApi.ts
import apiClient from '../api';
import {
  Driver,
  DriverDetailResponse,
  DriversListResponse,
  DriversSearchResponse,
  ArchetypesResponse,
  ArchetypeDetailResponse,
  HealthResponse,
  DriversQueryParams,
  DriverSearchParams,
  DriverSearchResult,
  ArchetypeSummary,
} from '../../types/api';

/**
 * NASCAR API Service Functions
 * 
 * This file contains all the functions that actually fetch data from your FastAPI backend.
 * Each function corresponds to a specific endpoint and returns properly typed data.
 * 
 * These functions handle:
 * - Making HTTP requests to your backend
 * - Error handling and user-friendly messages
 * - Type safety with our defined interfaces
 * - Consistent response formatting
 */

/**
 * Health Check Service
 * 
 * Checks if your FastAPI backend is running and responsive.
 * Useful for debugging connection issues and showing status in UI.
 */
export const healthService = {
  /**
   * Check backend health status
   * @returns Promise with health status
   */
  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await apiClient.get<HealthResponse>('/health');
      return response.data;
    } catch (error) {
      throw new Error('Backend health check failed');
    }
  },

  /**
   * Test connection to backend
   * @returns Promise<boolean> - true if backend is reachable
   */
  async isBackendReachable(): Promise<boolean> {
    try {
      await this.checkHealth();
      return true;
    } catch {
      return false;
    }
  },
};

/**
 * Driver Search Service
 * 
 * Handles driver search and autocomplete functionality.
 * This is what powers the search box on your homepage.
 */
export const driverSearchService = {
  /**
   * Search for drivers by name (autocomplete)
   * @param query - Search term (driver name)
   * @param limit - Maximum results to return (default: 10)
   * @returns Promise with search results
   */
  async searchDrivers(
    query: string,
    limit: number = 10
  ): Promise<DriverSearchResult[]> {
    try {
      // Validate input
      if (!query || query.trim().length < 2) {
        return []; // Return empty array for short queries
      }

      const params: DriverSearchParams = {
        q: query.trim(),
        limit,
      };

      const response = await apiClient.get<DriversSearchResponse>('/api/drivers/search', {
        params,
      });

      return response.data.drivers;
    } catch (error: any) {
      console.error('Driver search failed:', error);
      throw new Error(error.userMessage || 'Failed to search drivers');
    }
  },

  /**
   * Get popular drivers for suggestions
   * @returns Promise with top drivers by wins
   */
  async getPopularDrivers(): Promise<DriverSearchResult[]> {
    try {
      // Get top 8 drivers with most wins for homepage suggestions
      const params: DriversQueryParams = {
        limit: 8,
        offset: 0,
        min_wins: 10, // Only drivers with at least 10 wins
      };

      const response = await apiClient.get<DriversListResponse>('/api/drivers', {
        params,
      });

      // Convert Driver[] to DriverSearchResult[]
      return response.data.drivers.map(driver => ({
        id: driver.id,
        name: driver.name,
        total_wins: driver.career_stats.total_wins,
        is_active: driver.is_active,
        career_span: driver.career_span,
      }));
    } catch (error: any) {
      console.error('Failed to fetch popular drivers:', error);
      throw new Error(error.userMessage || 'Failed to load popular drivers');
    }
  },
};

/**
 * Driver Details Service
 * 
 * Handles fetching complete driver profiles and career information.
 */
export const driverDetailService = {
  /**
   * Get complete driver information by ID
   * @param driverId - Driver slug (e.g., "kyle-larson")
   * @returns Promise with complete driver data
   */
  async getDriverById(driverId: string): Promise<Driver> {
    try {
      // Validate input
      if (!driverId || !driverId.trim()) {
        throw new Error('Driver ID is required');
      }

      const response = await apiClient.get<DriverDetailResponse>(
        `/api/drivers/${encodeURIComponent(driverId.trim())}`
      );

      return response.data.driver;
    } catch (error: any) {
      console.error(`Failed to fetch driver ${driverId}:`, error);
      
      if (error.response?.status === 404) {
        throw new Error(`Driver "${driverId}" not found`);
      }
      
      throw new Error(error.userMessage || 'Failed to load driver information');
    }
  },

  /**
   * Check if a driver exists
   * @param driverId - Driver slug to check
   * @returns Promise<boolean> - true if driver exists
   */
  async driverExists(driverId: string): Promise<boolean> {
    try {
      await this.getDriverById(driverId);
      return true;
    } catch {
      return false;
    }
  },
};

/**
 * Drivers Listing Service
 * 
 * Handles fetching lists of drivers with filtering and pagination.
 */
export const driversListService = {
  /**
   * Get paginated list of drivers
   * @param params - Query parameters for filtering and pagination
   * @returns Promise with drivers list and pagination info
   */
  async getDrivers(params: DriversQueryParams = {}): Promise<DriversListResponse> {
    try {
      const response = await apiClient.get<DriversListResponse>('/api/drivers', {
        params,
      });

      return response.data;
    } catch (error: any) {
      console.error('Failed to fetch drivers list:', error);
      throw new Error(error.userMessage || 'Failed to load drivers');
    }
  },

  /**
   * Get all active drivers
   * @returns Promise with active drivers only
   */
  async getActiveDrivers(): Promise<Driver[]> {
    try {
      const response = await this.getDrivers({
        active_only: true,
        limit: 50, // Reasonable limit for active drivers
      });

      return response.drivers;
    } catch (error: any) {
      console.error('Failed to fetch active drivers:', error);
      throw new Error(error.userMessage || 'Failed to load active drivers');
    }
  },

  /**
   * Get drivers with minimum win count
   * @param minWins - Minimum career wins required
   * @returns Promise with filtered drivers
   */
  async getDriversWithMinWins(minWins: number): Promise<Driver[]> {
    try {
      const response = await this.getDrivers({
        min_wins: minWins,
        limit: 100, // Higher limit for winners
      });

      return response.drivers;
    } catch (error: any) {
      console.error(`Failed to fetch drivers with ${minWins}+ wins:`, error);
      throw new Error(error.userMessage || 'Failed to load drivers');
    }
  },
};

/**
 * Archetypes Service
 * 
 * Handles fetching driver archetype data from machine learning clustering.
 */
export const archetypesService = {
  /**
   * Get all driver archetypes from clustering analysis
   * @returns Promise with all archetype summaries
   */
  async getAllArchetypes(): Promise<ArchetypeSummary[]> {
    try {
      const response = await apiClient.get<ArchetypesResponse>('/api/archetypes');
      
      return response.data.archetypes;
    } catch (error: any) {
      console.error('Failed to fetch archetypes:', error);
      throw new Error(error.userMessage || 'Failed to load driver archetypes');
    }
  },

  /**
   * Get archetype by name
   * @param archetypeName - Name of the archetype to find
   * @returns Promise with specific archetype or null if not found
   */
  async getArchetypeByName(archetypeName: string): Promise<ArchetypeSummary | null> {
    try {
      const archetypes = await this.getAllArchetypes();
      
      return archetypes.find(
        archetype => archetype.name.toLowerCase() === archetypeName.toLowerCase()
      ) || null;
    } catch (error) {
      console.error(`Failed to find archetype "${archetypeName}":`, error);
      return null;
    }
  },

  /**
   * Get clustering information and metadata
   * @returns Promise with clustering analysis details
   */
  async getClusteringInfo(): Promise<ArchetypesResponse> {
    try {
      const response = await apiClient.get<ArchetypesResponse>('/api/archetypes');
      return response.data;
    } catch (error: any) {
      console.error('Failed to fetch clustering info:', error);
      throw new Error(error.userMessage || 'Failed to load clustering information');
    }
  },

  /**
   * Get detailed information for a specific archetype by ID
   * @param archetypeId - Archetype slug (e.g., 'mid-pack-drivers')
   * @returns Promise with full archetype details including driver list
   */
  async getArchetypeById(archetypeId: string): Promise<ArchetypeDetailResponse> {
    try {
      const response = await apiClient.get<ArchetypeDetailResponse>(`/api/archetypes/${archetypeId}`);
      
      return response.data;
    } catch (error: any) {
      console.error(`Failed to fetch archetype "${archetypeId}":`, error);
      throw new Error(error.userMessage || `Failed to load archetype "${archetypeId}"`);
    }
  },
};

/**
 * Combined NASCAR API Service
 * 
 * Main export that combines all services for easy importing.
 * This is what you'll import in your React components.
 */
export const nascarApi = {
  health: healthService,
  search: driverSearchService,
  drivers: driverDetailService,
  list: driversListService,
  archetypes: archetypesService,
};

/**
 * Convenience function for the most common operation: driver search
 * This is what your homepage search will use.
 */
export const searchDrivers = driverSearchService.searchDrivers;

/**
 * Convenience function for getting driver details
 * This is what your driver profile pages will use.
 */
export const getDriver = driverDetailService.getDriverById;