from fastapi import HTTPException, status
from core.database import messages_collection
from datetime import datetime

class UserController:
    """Controller for user-related operations"""
    
    @staticmethod
    async def get_user_media(user_id: str):
        """Get all media files (attachments) for a user across all conversations"""
        try:
            # Find all messages by this user that have non-empty attachments
            messages_cursor = messages_collection.find({
                "user_id": user_id,
                "attachments": {"$exists": True, "$ne": []}
            }).sort("timestamp", -1)
            
            messages = await messages_cursor.to_list(length=1000)
            
            media_list = []
            for msg in messages:
                # Robust check: Ensure attachments exists and is a list (not None)
                attachments = msg.get("attachments")
                if attachments and isinstance(attachments, list):
                    for attachment in attachments:
                        # Enrich attachment with message context
                        media_item = attachment.copy()
                        media_item["message_id"] = str(msg["_id"])
                        media_item["conversation_id"] = msg.get("conversation_id")
                        media_item["timestamp"] = msg.get("timestamp")
                        if isinstance(media_item["timestamp"], datetime):
                            media_item["timestamp"] = media_item["timestamp"].isoformat()
                            
                        media_list.append(media_item)
            
            return media_list
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
