"""
Local caching system for SatChange.

This module provides a disk-based LRU cache for satellite imagery tiles
to avoid redundant downloads from Google Earth Engine.
"""

import os
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import diskcache
from pathlib import Path

from .config import Config
from .utils import format_file_size

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Exception raised for cache-related errors."""
    pass


class ImageCache:
    """Disk-based LRU cache for satellite imagery tiles."""
    
    def __init__(self, config: Config):
        """Initialize image cache.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.cache_dir = self.config.get_cache_directory()
        self.max_size_bytes = self.config.get_cache_max_size_bytes()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize diskcache with LRU eviction policy
        self.cache = diskcache.Cache(
            self.cache_dir,
            size_limit=int(self.max_size_bytes),
            eviction_policy='least-recently-used'
        )
        
        logger.info(f"Initialized cache: {self.cache_dir} (max: {format_file_size(self.max_size_bytes)})")
    
    def _generate_key(self, center_lat: float, center_lon: float, 
                     pixel_size: int, date: datetime, bands: list) -> str:
        """Generate unique cache key from query parameters.
        
        Args:
            center_lat: Latitude of AOI center
            center_lon: Longitude of AOI center
            pixel_size: Number of pixels per side
            date: Acquisition date
            bands: List of band names
            
        Returns:
            SHA256 hash string as cache key
        """
        params = {
            'lat': round(center_lat, 6),
            'lon': round(center_lon, 6),
            'size': pixel_size,
            'date': date.isoformat(),
            'bands': sorted(bands)
        }
        
        key_string = json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, center_lat: float, center_lon: float, 
           pixel_size: int, date: datetime, bands: list) -> Optional[Dict[str, Any]]:
        """Retrieve cached image data.
        
        Args:
            center_lat: Latitude of AOI center
            center_lon: Longitude of AOI center
            pixel_size: Number of pixels per side
            date: Acquisition date
            bands: List of band names
            
        Returns:
            Cached data dictionary or None if not found
        """
        key = self._generate_key(center_lat, center_lon, pixel_size, date, bands)
        
        try:
            cached_data = self.cache.get(key)
            if cached_data is not None:
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return cached_data
            else:
                logger.debug(f"Cache miss for key: {key[:8]}...")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve from cache: {e}")
            return None
    
    def set(self, center_lat: float, center_lon: float, 
           pixel_size: int, date: datetime, bands: list, 
           data: Dict[str, Any]) -> bool:
        """Store image data in cache.
        
        Args:
            center_lat: Latitude of AOI center
            center_lon: Longitude of AOI center
            pixel_size: Number of pixels per side
            date: Acquisition date
            bands: List of band names
            data: Image data to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        key = self._generate_key(center_lat, center_lon, pixel_size, date, bands)
        
        try:
            # Copy data to avoid mutating caller's dict
            cache_data = dict(data)
            cache_data['cached_at'] = datetime.now().isoformat()
            cache_data['cache_key'] = key
            
            # Store in cache
            self.cache.set(key, cache_data)
            
            logger.debug(f"Cached data for key: {key[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in cache: {e}")
            return False
    
    def delete(self, center_lat: float, center_lon: float, 
              pixel_size: int, date: datetime, bands: list) -> bool:
        """Delete cached image data.
        
        Args:
            center_lat: Latitude of AOI center
            center_lon: Longitude of AOI center
            pixel_size: Number of pixels per side
            date: Acquisition date
            bands: List of band names
            
        Returns:
            True if successfully deleted, False otherwise
        """
        key = self._generate_key(center_lat, center_lon, pixel_size, date, bands)
        
        try:
            deleted = self.cache.delete(key)
            if deleted:
                logger.debug(f"Deleted cache entry for key: {key[:8]}...")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete from cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached data.
        
        Returns:
            True if successfully cleared, False otherwise
        """
        try:
            self.cache.clear()
            logger.info("Cleared all cached data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get basic cache statistics - diskcache.Cache.stats() returns (hits, misses) tuple
            cache_stats_tuple = self.cache.stats()
            cache_stats = {
                'hits': cache_stats_tuple[0] if isinstance(cache_stats_tuple, tuple) else cache_stats_tuple.get('hits', 0),
                'misses': cache_stats_tuple[1] if isinstance(cache_stats_tuple, tuple) else cache_stats_tuple.get('misses', 0)
            }
            
            # Get volume information
            volume = self.cache.volume()
            
            # Count total items
            total_items = len(self.cache)
            
            # Get directory size
            dir_size = self._get_directory_size()
            
            return {
                'total_items': total_items,
                'size_bytes': volume,
                'size_formatted': format_file_size(volume),
                'directory_size_bytes': dir_size,
                'directory_size_formatted': format_file_size(dir_size),
                'max_size_bytes': self.max_size_bytes,
                'max_size_formatted': format_file_size(self.max_size_bytes),
                'usage_percent': round((volume / self.max_size_bytes) * 100, 2) if self.max_size_bytes > 0 else 0.0,
                'hits': cache_stats.get('hits', 0),
                'misses': cache_stats.get('misses', 0),
                'hit_rate': self._calculate_hit_rate(cache_stats),
                'evictions': 0  # diskcache doesn't track evictions easily
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def _get_directory_size(self) -> int:
        """Get total size of cache directory.
        
        Returns:
            Size in bytes
        """
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size
        except Exception as e:
            logger.error(f"Failed to get directory size: {e}")
            return 0
    
    def _calculate_hit_rate(self, cache_stats: Dict[str, Any]) -> float:
        """Calculate cache hit rate.
        
        Args:
            cache_stats: Cache statistics from diskcache
            
        Returns:
            Hit rate as percentage (0-100)
        """
        hits = cache_stats.get('hits', 0)
        misses = cache_stats.get('misses', 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round((hits / total) * 100, 2)
    
    def cleanup(self) -> bool:
        """Clean up cache and remove old entries.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Remove expired entries (older than 30 days)
            cutoff_date = datetime.now().timestamp() - (30 * 24 * 60 * 60)
            
            removed_count = 0
            for key in list(self.cache.keys()):
                try:
                    data = self.cache.get(key)
                    if data and 'cached_at' in data:
                        cached_time = datetime.fromisoformat(data['cached_at']).timestamp()
                        if cached_time < cutoff_date:
                            self.cache.delete(key)
                            removed_count += 1
                except Exception:
                    # If we can't get the data, remove it
                    self.cache.delete(key)
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired cache entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            return False
    
    def get_cache_info(self, center_lat: float, center_lon: float, 
                      pixel_size: int, date: datetime, bands: list) -> Optional[Dict[str, Any]]:
        """Get information about a specific cache entry.
        
        Args:
            center_lat: Latitude of AOI center
            center_lon: Longitude of AOI center
            pixel_size: Number of pixels per side
            date: Acquisition date
            bands: List of band names
            
        Returns:
            Cache entry information or None if not found
        """
        key = self._generate_key(center_lat, center_lon, pixel_size, date, bands)
        
        try:
            data = self.cache.get(key)
            if data:
                return {
                    'key': key,
                    'cached_at': data.get('cached_at'),
                    'bands': data.get('bands', []),
                    'size_bytes': len(str(data)),  # Approximate size
                    'has_arrays': 'arrays' in data,
                    'has_metadata': 'metadata' in data
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return None
    
    def close(self) -> None:
        """Close cache connection."""
        try:
            self.cache.close()
            logger.info("Cache connection closed")
        except Exception as e:
            logger.error(f"Failed to close cache: {e}")


class CacheManager:
    """High-level cache manager for SatChange."""
    
    def __init__(self, config: Config):
        """Initialize cache manager.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.cache = ImageCache(config)
    
    def get_image_with_cache(self, center_lat: float, center_lon: float, 
                           pixel_size: int, date: datetime, bands: list,
                           download_func, *download_args, **download_kwargs) -> Tuple[Dict[str, Any], bool]:
        """Get image data from cache or download if not available.
        
        Args:
            center_lat: Latitude of AOI center
            center_lon: Longitude of AOI center
            pixel_size: Number of pixels per side
            date: Acquisition date
            bands: List of band names
            download_func: Function to download image if cache miss
            *download_args: Arguments for download function
            **download_kwargs: Keyword arguments for download function
            
        Returns:
            Tuple of (image_data, cache_hit)
        """
        # Try cache first
        cached_data = self.cache.get(center_lat, center_lon, pixel_size, date, bands)
        
        if cached_data is not None:
            logger.info("Cache hit - loading from disk")
            return cached_data, True
        
        # Cache miss - download from source
        logger.info("Cache miss - downloading from source...")
        
        try:
            # Download image data
            image_data = download_func(*download_args, **download_kwargs)
            
            # Store in cache
            self.cache.set(center_lat, center_lon, pixel_size, date, bands, image_data)
            
            logger.info("Download complete and cached")
            return image_data, False
            
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return self.cache.stats()
    
    def clear_cache(self) -> bool:
        """Clear all cached data.
        
        Returns:
            True if successful, False otherwise
        """
        return self.cache.clear()
    
    def cleanup_cache(self) -> bool:
        """Clean up old cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        return self.cache.cleanup()
    
    def close(self) -> None:
        """Close cache manager."""
        self.cache.close()