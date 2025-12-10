from datetime import datetime, timedelta

def is_gemini_file_expired(uploaded_at: datetime) -> bool:
    """
    Check if a Gemini Files API upload has expired (48 hour limit)
    
    Args:
        uploaded_at: The datetime when the file was uploaded to Gemini
    
    Returns:
        True if the file is expired or will expire soon (47h buffer), False otherwise
    """
    if not uploaded_at:
        return True
    
    # Use 47 hours as threshold (1 hour safety buffer before actual 48h expiry)
    expiry_threshold = timedelta(hours=47)
    time_elapsed = datetime.now() - uploaded_at
    
    return time_elapsed >= expiry_threshold
