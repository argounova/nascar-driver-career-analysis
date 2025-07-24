# backend/app/main.py
"""
NASCAR Driver Career Analysis FastAPI Backend

Main application file that creates the FastAPI app and includes all routers.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NASCAR Driver Career Analysis API",
    description="API for NASCAR driver performance analysis, clustering, and career predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data storage - will be loaded on startup
app_data = {
    "drivers": None,
    "archetypes": None,
    "predictions": None,
    "export_summary": None
}

def load_data():
    """Load JSON data files into memory."""
    data_dir = Path(__file__).parent / "data"
    
    try:
        # Load drivers data
        with open(data_dir / "drivers.json", "r") as f:
            app_data["drivers"] = json.load(f)
        logger.info(f"Loaded {len(app_data['drivers']['drivers'])} drivers")
        
        # Load archetypes data
        with open(data_dir / "archetypes.json", "r") as f:
            app_data["archetypes"] = json.load(f)
        logger.info(f"Loaded {len(app_data['archetypes']['archetypes'])} archetypes")
        
        # Load predictions data
        with open(data_dir / "predictions.json", "r") as f:
            app_data["predictions"] = json.load(f)
        logger.info("Loaded predictions data")
        
        # Load export summary
        with open(data_dir / "export_summary.json", "r") as f:
            app_data["export_summary"] = json.load(f)
        logger.info("Loaded export summary")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Please run the data export script first: python3 backend/scripts/export_data.py")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in data file: {e}")

@app.on_event("startup")
async def startup_event():
    """Load data when the application starts."""
    logger.info("Starting NASCAR API...")
    load_data()
    logger.info("NASCAR API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the application shuts down."""
    logger.info("Shutting down NASCAR API...")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NASCAR Driver Career Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
        "data_status": {
            "drivers_loaded": app_data["drivers"] is not None,
            "archetypes_loaded": app_data["archetypes"] is not None,
            "predictions_loaded": app_data["predictions"] is not None
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": app_data["export_summary"]["export_timestamp"] if app_data["export_summary"] else None
    }

# ============================================================================
# DRIVER ENDPOINTS
# ============================================================================

@app.get("/api/drivers")
async def get_drivers(
    limit: int = 50,
    offset: int = 0,
    active_only: bool = False,
    min_wins: int = 0
):
    """
    Get list of drivers with optional filtering.
    
    Args:
        limit: Maximum number of drivers to return
        offset: Number of drivers to skip (for pagination)
        active_only: Only return currently active drivers
        min_wins: Minimum career wins required
    """
    if app_data["drivers"] is None:
        raise HTTPException(status_code=503, detail="Driver data not loaded")
    
    drivers = app_data["drivers"]["drivers"]
    
    # Apply filters
    if active_only:
        drivers = [d for d in drivers if d["is_active"]]
    
    if min_wins > 0:
        drivers = [d for d in drivers if d["career_stats"]["total_wins"] >= min_wins]
    
    # Apply pagination
    total = len(drivers)
    drivers_page = drivers[offset:offset + limit]
    
    return {
        "drivers": drivers_page,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        },
        "filters_applied": {
            "active_only": active_only,
            "min_wins": min_wins
        }
    }

@app.get("/api/drivers/search")
async def search_drivers(q: str, limit: int = 10):
    """
    Search drivers by name for autocomplete.
    
    Args:
        q: Search query (partial driver name)
        limit: Maximum results to return
    """
    if app_data["drivers"] is None:
        raise HTTPException(status_code=503, detail="Driver data not loaded")
    
    if not q or len(q) < 2:
        return {"drivers": [], "query": q}
    
    query = q.lower()
    drivers = app_data["drivers"]["drivers"]
    
    # Search by name (case-insensitive)
    matches = []
    for driver in drivers:
        if query in driver["name"].lower():
            matches.append({
                "id": driver["id"],
                "name": driver["name"],
                "total_wins": driver["career_stats"]["total_wins"],
                "is_active": driver["is_active"],
                "career_span": driver["career_span"]
            })
    
    # Sort by total wins (most successful first) and limit results
    matches.sort(key=lambda x: x["total_wins"], reverse=True)
    matches = matches[:limit]
    
    return {
        "drivers": matches,
        "query": q,
        "total_matches": len(matches)
    }

@app.get("/api/drivers/{driver_id}")
async def get_driver(driver_id: str):
    """
    Get detailed information for a specific driver.
    
    Args:
        driver_id: Driver slug (e.g., 'kyle-larson')
    """
    if app_data["drivers"] is None:
        raise HTTPException(status_code=503, detail="Driver data not loaded")
    
    # Find driver by ID
    drivers = app_data["drivers"]["drivers"]
    driver = None
    for d in drivers:
        if d["id"] == driver_id:
            driver = d
            break
    
    if not driver:
        raise HTTPException(status_code=404, detail=f"Driver '{driver_id}' not found")
    
    # Add archetype information if available
    if app_data["archetypes"] and "driver_archetypes" in app_data["archetypes"]:
        driver_archetypes = app_data["archetypes"]["driver_archetypes"]

        if driver_id in driver_archetypes:
            # Get the driver's archetype assignment
            driver_archetype_info = driver_archetypes[driver_id]
            archetype_id = driver_archetype_info["archetype_id"]

            # Find the full archetype details
            archetypes = app_data["archetypes"]["archetypes"]
            for archetype in archetypes:
                if archetype["id"] == archetype_id:
                    # Assign complete archetype information
                    driver["archetype"] = {
                        "name": archetype["name"],
                        "color": archetype["color"],
                        "description": archetype.get("description", "") 
                    }
                    break

            # If archetype not found in main list, use fallback
            if "archetype" not in driver:
                driver["archetype"] = {
                    "name": driver_archetype_info["archetype_name"],
                    "color": "#666666",
                    "description": f"Driver classified as {driver_archetype_info['archetype_name']}"
                }
    
    return {
        "driver": driver,
        "data_source": app_data["drivers"]["data_source"],
        "last_updated": app_data["drivers"]["export_timestamp"]
    }

# ============================================================================
# ARCHETYPE ENDPOINTS
# ============================================================================

@app.get("/api/archetypes")
async def get_archetypes():
    """Get all driver archetypes from clustering analysis."""
    if app_data["archetypes"] is None:
        raise HTTPException(status_code=503, detail="Archetype data not loaded")
    
    return {
        "archetypes": app_data["archetypes"]["archetypes"],
        "clustering_info": app_data["archetypes"]["clustering_info"],
        "total_drivers_clustered": app_data["archetypes"]["clustering_info"]["total_drivers_clustered"],
        "last_updated": app_data["archetypes"]["export_timestamp"]
    }

@app.get("/api/archetypes/{archetype_id}")
async def get_archetype(archetype_id: str):
    """
    Get detailed information for a specific archetype.
    
    Args:
        archetype_id: Archetype slug (e.g., 'dominant-champions')
    """
    if app_data["archetypes"] is None:
        raise HTTPException(status_code=503, detail="Archetype data not loaded")
    
    # Find archetype by ID
    archetypes = app_data["archetypes"]["archetypes"]
    archetype = None
    for a in archetypes:
        if a["id"] == archetype_id:
            archetype = a
            break
    
    if not archetype:
        raise HTTPException(status_code=404, detail=f"Archetype '{archetype_id}' not found")
    
    # Get drivers in this archetype
    driver_archetypes = app_data["archetypes"]["driver_archetypes"]
    archetype_drivers = []
    
    if app_data["drivers"]:
        for driver in app_data["drivers"]["drivers"]:
            if driver["id"] in driver_archetypes:
                if driver_archetypes[driver["id"]]["archetype_id"] == archetype_id:
                    archetype_drivers.append({
                        "id": driver["id"],
                        "name": driver["name"],
                        "total_wins": driver["career_stats"]["total_wins"],
                        "career_avg_finish": driver["career_stats"]["career_avg_finish"],
                        "total_seasons": driver["total_seasons"],
                        "is_active": driver["is_active"]
                    })
    
    # Sort drivers by total wins
    archetype_drivers.sort(key=lambda x: x["total_wins"], reverse=True)
    
    return {
        "archetype": archetype,
        "drivers": archetype_drivers,
        "driver_count": len(archetype_drivers),
        "last_updated": app_data["archetypes"]["export_timestamp"]
    }

# ============================================================================
# PREDICTION ENDPOINTS (Placeholder for now)
# ============================================================================

@app.get("/api/predictions/{driver_id}")
async def get_driver_predictions(driver_id: str):
    """
    Get career predictions for a specific driver.
    
    Args:
        driver_id: Driver slug (e.g., 'kyle-larson')
    """
    if app_data["predictions"] is None:
        raise HTTPException(status_code=503, detail="Prediction data not loaded")
    
    # For now, return placeholder since LSTM integration comes later
    return {
        "driver_id": driver_id,
        "predictions_available": app_data["predictions"]["predictions_available"],
        "message": app_data["predictions"]["message"],
        "last_updated": app_data["predictions"]["export_timestamp"]
    }

# ============================================================================
# COMPARISON ENDPOINTS
# ============================================================================

@app.post("/api/drivers/compare")
async def compare_drivers(driver_ids: list[str]):
    """
    Compare multiple drivers side by side.
    
    Args:
        driver_ids: List of driver slugs to compare
    """
    if app_data["drivers"] is None:
        raise HTTPException(status_code=503, detail="Driver data not loaded")
    
    if len(driver_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 drivers required for comparison")
    
    if len(driver_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 drivers can be compared")
    
    # Find all requested drivers
    drivers = app_data["drivers"]["drivers"]
    comparison_drivers = []
    
    for driver_id in driver_ids:
        driver = None
        for d in drivers:
            if d["id"] == driver_id:
                driver = d
                break
        
        if driver:
            # Add archetype info
            if app_data["archetypes"] and "driver_archetypes" in app_data["archetypes"]:
                driver_archetypes = app_data["archetypes"]["driver_archetypes"]
                if driver_id in driver_archetypes:
                    driver["archetype"] = driver_archetypes[driver_id]
            
            comparison_drivers.append(driver)
        else:
            raise HTTPException(status_code=404, detail=f"Driver '{driver_id}' not found")
    
    return {
        "drivers": comparison_drivers,
        "comparison_metrics": [
            "total_wins",
            "career_avg_finish", 
            "career_top5_rate",
            "career_win_rate",
            "total_seasons"
        ],
        "total_drivers": len(comparison_drivers)
    }

# ============================================================================
# DATA STATUS ENDPOINTS
# ============================================================================

@app.get("/api/status")
async def get_data_status():
    """Get information about the loaded data and when it was last updated."""
    return {
        "api_status": "running",
        "data_loaded": {
            "drivers": app_data["drivers"] is not None,
            "archetypes": app_data["archetypes"] is not None,
            "predictions": app_data["predictions"] is not None
        },
        "export_summary": app_data["export_summary"],
        "stats": {
            "total_drivers": len(app_data["drivers"]["drivers"]) if app_data["drivers"] else 0,
            "total_archetypes": len(app_data["archetypes"]["archetypes"]) if app_data["archetypes"] else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)