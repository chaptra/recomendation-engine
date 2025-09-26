"""Input validation and sanitization utilities."""

import re
import logging
from typing import Dict, Any, Optional

from exceptions import InvalidInputError

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    # Regex patterns for validation
    TITLE_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-\.,;:!?\'"()&]+$')
    SQL_INJECTION_PATTERN = re.compile(r'(\'|(\\x27)|(\\x2D\\x2D)|(%27)|(%2D%2D))', re.IGNORECASE)
    
    @staticmethod
    def validate_book_title(title: str) -> str:
        """
        Validate and sanitize book title input.
        
        Args:
            title: Raw book title input
            
        Returns:
            Sanitized title
            
        Raises:
            InvalidInputError: If title is invalid
        """
        if not title or not isinstance(title, str):
            raise InvalidInputError("Title must be a non-empty string")
        
        # Remove leading/trailing whitespace
        title = title.strip()
        
        # Check length
        if len(title) < 1:
            raise InvalidInputError("Title cannot be empty")
        if len(title) > 500:
            raise InvalidInputError("Title too long (max 500 characters)")
        
        # Check for SQL injection attempts
        if InputValidator.SQL_INJECTION_PATTERN.search(title):
            logger.warning(f"Potential SQL injection attempt detected: {title}")
            raise InvalidInputError("Invalid characters in title")
        
        # Allow basic punctuation and alphanumeric characters
        if not InputValidator.TITLE_PATTERN.match(title):
            # Remove invalid characters instead of rejecting
            title = re.sub(r'[^a-zA-Z0-9\s\-\.,;:!?\'"()&]', '', title)
            if not title.strip():
                raise InvalidInputError("Title contains no valid characters")
        
        return title.strip()
    
    @staticmethod
    def validate_limit(limit: Any) -> int:
        """
        Validate recommendation limit parameter.
        
        Args:
            limit: Raw limit input
            
        Returns:
            Validated limit as integer
            
        Raises:
            InvalidInputError: If limit is invalid
        """
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            raise InvalidInputError("Limit must be a valid integer")
        
        if limit < 1:
            raise InvalidInputError("Limit must be at least 1")
        if limit > 100:  # Reasonable upper bound
            raise InvalidInputError("Limit cannot exceed 100")
        
        return limit
    
    @staticmethod
    def validate_similarity_threshold(threshold: Any) -> float:
        """
        Validate similarity threshold parameter.
        
        Args:
            threshold: Raw threshold input
            
        Returns:
            Validated threshold as float
            
        Raises:
            InvalidInputError: If threshold is invalid
        """
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            raise InvalidInputError("Threshold must be a valid number")
        
        if not 0.0 <= threshold <= 1.0:
            raise InvalidInputError("Threshold must be between 0.0 and 1.0")
        
        return threshold
    
    @staticmethod
    def sanitize_text_for_search(text: str) -> str:
        """
        Sanitize text for safe database searching.
        
        Args:
            text: Raw text input
            
        Returns:
            Sanitized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove SQL injection patterns
        text = InputValidator.SQL_INJECTION_PATTERN.sub('', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Limit length
        if len(text) > 1000:
            text = text[:1000]
        
        return text.strip()
    
    @staticmethod
    def validate_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete request data.
        
        Args:
            data: Request data dictionary
            
        Returns:
            Validated and sanitized data
            
        Raises:
            InvalidInputError: If any field is invalid
        """
        validated = {}
        
        # Required field: title
        if 'title' not in data:
            raise InvalidInputError("Missing required field: title")
        validated['title'] = InputValidator.validate_book_title(data['title'])
        
        # Optional field: limit
        if 'limit' in data:
            validated['limit'] = InputValidator.validate_limit(data['limit'])
        else:
            validated['limit'] = 10  # Default
        
        # Optional field: threshold
        if 'threshold' in data:
            validated['threshold'] = InputValidator.validate_similarity_threshold(data['threshold'])
        else:
            validated['threshold'] = 0.1  # Default
        
        # Optional field: include_metadata
        validated['include_metadata'] = bool(data.get('include_metadata', True))
        
        return validated


class SecurityValidator:
    """Security-focused validation utilities."""
    
    @staticmethod
    def check_rate_limit_headers(headers: Dict[str, str]) -> bool:
        """
        Check if request should be rate limited based on headers.
        
        Args:
            headers: Request headers
            
        Returns:
            True if request should be allowed
        """
        # Check for common bot patterns
        user_agent = headers.get('User-Agent', '').lower()
        suspicious_agents = ['bot', 'crawler', 'spider', 'scraper']
        
        if any(agent in user_agent for agent in suspicious_agents):
            logger.warning(f"Suspicious user agent detected: {user_agent}")
            return False
        
        return True
    
    @staticmethod
    def validate_api_key(api_key: Optional[str]) -> bool:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid
        """
        if not api_key:
            return False
        
        # Basic format validation (customize as needed)
        if len(api_key) < 20 or len(api_key) > 100:
            return False
        
        # Check for valid characters only
        if not re.match(r'^[a-zA-Z0-9\-_]+$', api_key):
            return False
        
        return True