// src/lib/api.ts
import axios, { AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios';

/**
 * Base API Configuration
 * 
 * This file creates the foundation for all communication with your FastAPI backend.
 * It sets up axios with proper configuration, error handling, and request/response
 * interceptors that will be used throughout the application.
 */

// Extend axios config type to include our custom metadata
interface CustomAxiosRequestConfig extends InternalAxiosRequestConfig {
  metadata?: {
    startTime: number;
  };
}

// API Configuration
const API_CONFIG = {
  // Your FastAPI backend URL - update this if your backend runs on different port
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  
  // Request timeout (10 seconds)
  timeout: 10000,
  
  // Default headers for all requests
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
} as const;

/**
 * Create axios instance with base configuration
 * 
 * This creates a pre-configured axios instance that all API calls will use.
 * It includes the base URL to your FastAPI backend, timeouts, and default headers.
 */
const apiClient: AxiosInstance = axios.create(API_CONFIG);

/**
 * Request Interceptor
 * 
 * This runs before every API request is sent. It's useful for:
 * - Adding authentication tokens (when we add auth later)
 * - Logging requests during development
 * - Adding common headers
 */
apiClient.interceptors.request.use(
  (config: CustomAxiosRequestConfig) => {
    // Log API calls during development
    if (process.env.NODE_ENV === 'development') {
      console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    }
    
    // Add timestamp to requests (useful for tracking response times)
    config.metadata = { startTime: Date.now() };
    
    return config;
  },
  (error) => {
    console.error('❌ Request Error:', error);
    return Promise.reject(error);
  }
);

/**
 * Response Interceptor
 * 
 * This runs after every API response is received. It handles:
 * - Success logging
 * - Error logging and formatting
 * - Response time tracking
 * - Global error handling
 */
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // Calculate response time safely
    const config = response.config as CustomAxiosRequestConfig;
    const responseTime = config.metadata?.startTime 
      ? Date.now() - config.metadata.startTime 
      : 0;
    
    if (process.env.NODE_ENV === 'development') {
      console.log(`✅ API Response: ${response.config.url} (${responseTime}ms)`);
    }
    
    return response;
  },
  (error) => {
    // Enhanced error handling with specific NASCAR API context
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      console.error(`❌ API Error ${status}:`, {
        url: error.config?.url,
        method: error.config?.method,
        status,
        message: data?.detail || data?.message || 'Unknown error',
      });
      
      // Create user-friendly error messages
      switch (status) {
        case 404:
          error.userMessage = 'Driver or data not found';
          break;
        case 429:
          error.userMessage = 'Too many requests. Please wait a moment.';
          break;
        case 500:
          error.userMessage = 'Server error. Please try again later.';
          break;
        default:
          error.userMessage = data?.detail || 'Something went wrong';
      }
    } else if (error.request) {
      // Network error - can't reach the FastAPI backend
      console.error('❌ Network Error:', error.message);
      error.userMessage = 'Cannot connect to NASCAR analytics server. Is it running?';
    } else {
      // Something else went wrong
      console.error('❌ Unknown Error:', error.message);
      error.userMessage = 'An unexpected error occurred';
    }
    
    return Promise.reject(error);
  }
);

/**
 * Health Check Function
 * 
 * This function checks if your FastAPI backend is running and responsive.
 * Useful for debugging connection issues and showing connection status in the UI.
 */
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await apiClient.get('/health');
    console.log('🏥 Backend Health Check:', response.data);
    return response.status === 200;
  } catch (error) {
    console.error('🏥 Backend Health Check Failed:', error);
    return false;
  }
};

/**
 * Generic API Error Class
 * 
 * Custom error class for NASCAR API-specific errors.
 * Provides better error handling and user feedback.
 */
export class NascarApiError extends Error {
  public status?: number;
  public userMessage: string;
  
  constructor(message: string, status?: number, userMessage?: string) {
    super(message);
    this.name = 'NascarApiError';
    this.status = status;
    this.userMessage = userMessage || message;
  }
}

/**
 * Export the configured axios instance
 * 
 * This is what all other API functions will use to make requests.
 * It's pre-configured with your FastAPI backend URL, timeouts, and interceptors.
 */
export default apiClient;