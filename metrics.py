"""Monitoring and metrics collection for the recommendation engine."""

import time
import logging
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Deque
from threading import Lock
import psutil
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for individual requests."""
    timestamp: float
    duration_ms: float
    status: str  # success, error, timeout
    book_title: str
    recommendations_count: int
    cache_hit: bool
    error_type: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    timestamp: float


class MetricsCollector:
    """Collects and aggregates metrics for monitoring."""
    
    def __init__(self, max_requests_history: int = 10000):
        self.max_requests_history = max_requests_history
        self.requests_history: Deque[RequestMetrics] = deque(maxlen=max_requests_history)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.system_metrics: Deque[SystemMetrics] = deque(maxlen=1000)
        
        # Thread safety
        self._lock = Lock()
        
        # Performance tracking
        self.response_times: Deque[float] = deque(maxlen=1000)
        
    @contextmanager
    def track_request(self, book_title: str):
        """Context manager to track request metrics."""
        start_time = time.time()
        request_metric = RequestMetrics(
            timestamp=start_time,
            duration_ms=0,
            status='pending',
            book_title=book_title,
            recommendations_count=0,
            cache_hit=False
        )
        
        try:
            yield request_metric
            request_metric.status = 'success'
            
        except Exception as e:
            request_metric.status = 'error'
            request_metric.error_type = type(e).__name__
            logger.error(f"Request failed for '{book_title}': {e}")
            raise
            
        finally:
            end_time = time.time()
            request_metric.duration_ms = (end_time - start_time) * 1000
            
            with self._lock:
                self.requests_history.append(request_metric)
                self.request_counts[request_metric.status] += 1
                self.response_times.append(request_metric.duration_ms)
                
                if request_metric.cache_hit:
                    self.cache_stats['hits'] += 1
                else:
                    self.cache_stats['misses'] += 1
                
                if request_metric.error_type:
                    self.error_counts[request_metric.error_type] += 1
    
    def record_system_metrics(self):
        """Record current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metric = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / (1024**2),
                disk_usage_percent=disk.percent,
                timestamp=time.time()
            )
            
            with self._lock:
                self.system_metrics.append(metric)
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def get_performance_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance statistics for the specified time window."""
        try:
            cutoff_time = time.time() - (window_minutes * 60)
            
            with self._lock:
                # Filter recent requests
                recent_requests = [
                    req for req in self.requests_history 
                    if req.timestamp > cutoff_time
                ]
                
                if not recent_requests:
                    return {
                        'requests_count': 0,
                        'avg_response_time_ms': 0,
                        'success_rate': 0,
                        'error_rate': 0
                    }
                
                # Calculate metrics
                total_requests = len(recent_requests)
                successful_requests = sum(1 for req in recent_requests if req.status == 'success')
                response_times = [req.duration_ms for req in recent_requests]
                
                # Cache metrics
                cache_hits = sum(1 for req in recent_requests if req.cache_hit)
                
                return {
                    'window_minutes': window_minutes,
                    'requests_count': total_requests,
                    'successful_requests': successful_requests,
                    'avg_response_time_ms': np.mean(response_times) if response_times else 0,
                    'p95_response_time_ms': np.percentile(response_times, 95) if response_times else 0,
                    'p99_response_time_ms': np.percentile(response_times, 99) if response_times else 0,
                    'min_response_time_ms': min(response_times) if response_times else 0,
                    'max_response_time_ms': max(response_times) if response_times else 0,
                    'success_rate': (successful_requests / total_requests) * 100,
                    'error_rate': ((total_requests - successful_requests) / total_requests) * 100,
                    'cache_hit_rate': (cache_hits / total_requests) * 100 if total_requests > 0 else 0,
                    'requests_per_minute': total_requests / window_minutes
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate performance stats: {e}")
            return {'error': str(e)}
    
    def get_error_breakdown(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get error breakdown for the specified time window."""
        try:
            cutoff_time = time.time() - (window_minutes * 60)
            
            with self._lock:
                recent_requests = [
                    req for req in self.requests_history 
                    if req.timestamp > cutoff_time and req.error_type
                ]
                
                error_breakdown = defaultdict(int)
                for req in recent_requests:
                    error_breakdown[req.error_type] += 1
                
                return dict(error_breakdown)
                
        except Exception as e:
            logger.error(f"Failed to get error breakdown: {e}")
            return {'error': str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        try:
            self.record_system_metrics()
            
            with self._lock:
                if not self.system_metrics:
                    return {'status': 'no_data'}
                
                latest = self.system_metrics[-1]
                
                # Calculate health status
                health_issues = []
                if latest.cpu_percent > 80:
                    health_issues.append('high_cpu')
                if latest.memory_percent > 85:
                    health_issues.append('high_memory')
                if latest.disk_usage_percent > 90:
                    health_issues.append('high_disk')
                
                status = 'unhealthy' if health_issues else 'healthy'
                
                return {
                    'status': status,
                    'issues': health_issues,
                    'cpu_percent': latest.cpu_percent,
                    'memory_percent': latest.memory_percent,
                    'memory_mb': latest.memory_mb,
                    'disk_usage_percent': latest.disk_usage_percent,
                    'timestamp': latest.timestamp
                }
                
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_popular_books(self, limit: int = 10, window_minutes: int = 1440) -> Dict[str, Any]:
        """Get most requested books in the time window."""
        try:
            cutoff_time = time.time() - (window_minutes * 60)
            
            with self._lock:
                recent_requests = [
                    req for req in self.requests_history 
                    if req.timestamp > cutoff_time
                ]
                
                book_counts = defaultdict(int)
                for req in recent_requests:
                    book_counts[req.book_title] += 1
                
                # Sort by popularity
                popular_books = sorted(
                    book_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:limit]
                
                return {
                    'window_minutes': window_minutes,
                    'popular_books': [
                        {'title': title, 'request_count': count} 
                        for title, count in popular_books
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get popular books: {e}")
            return {'error': str(e)}
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self._lock:
            self.requests_history.clear()
            self.request_counts.clear()
            self.error_counts.clear()
            self.cache_stats = {'hits': 0, 'misses': 0}
            self.system_metrics.clear()
            self.response_times.clear()
        
        logger.info("Metrics reset successfully")


# Global metrics collector instance
metrics_collector = MetricsCollector()