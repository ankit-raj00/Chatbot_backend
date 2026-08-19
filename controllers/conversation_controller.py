from fastapi import HTTPException, status
from core.database import conversations_collection, messages_collection
from models.conversation import ConversationCreate
from datetime import datetime
from bson import ObjectId

class ConversationController:
    """Controller for conversation operations"""
    
    @staticmethod
    async def get_user_conversations(user_id: str):
        """Get all conversations for a user"""
        try:
            conversations_cursor = conversations_collection.find({
                "user_id": user_id
            }).sort("updated_at", -1)
            conversations_list = await conversations_cursor.to_list(length=100)
            
            # Convert ObjectId to string
            for conv in conversations_list:
                conv["_id"] = str(conv["_id"])
                conv["created_at"] = conv["created_at"].isoformat()
                conv["updated_at"] = conv["updated_at"].isoformat()
            
            return conversations_list
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def create_conversation(user_id: str, conversation: ConversationCreate):
        """Create a new conversation"""
        try:
            new_conv = {
                "user_id": user_id,
                "title": conversation.title,
                "mcp_server_id": conversation.mcp_server_id,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            result = await conversations_collection.insert_one(new_conv)
            new_conv["_id"] = str(result.inserted_id)
            new_conv["created_at"] = new_conv["created_at"].isoformat()
            new_conv["updated_at"] = new_conv["updated_at"].isoformat()
            return new_conv
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def get_conversation_messages(conversation_id: str, user_id: str):
        """Get all messages for a conversation"""
        try:
            # Verify conversation belongs to user
            conv = await conversations_collection.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })
            if not conv:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            
            messages_cursor = messages_collection.find({
                "conversation_id": conversation_id,
                "user_id": user_id,
                # Responses replaced via Retry are tombstoned rather than
                # deleted (see ChatService.retry) so a failed regeneration
                # can't destroy the original — but the user should only ever
                # see the surviving one.
                "superseded": {"$ne": True}
            }).sort("timestamp", 1)
            messages_list = await messages_cursor.to_list(length=1000)
            
            # Convert ObjectId and datetime to string
            for msg in messages_list:
                msg["_id"] = str(msg["_id"])
                msg["timestamp"] = msg["timestamp"].isoformat()
                if "tool_steps" in msg:
                    msg["toolSteps"] = msg["tool_steps"]
            
            return messages_list
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def rename_conversation(conversation_id: str, user_id: str, title: str):
        """Rename a conversation. Scoped by user_id in the update filter itself
        so another user's conversation can never be touched, matching how
        submit_message_feedback scopes its update."""
        try:
            clean = (title or "").strip()
            if not clean:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Title cannot be empty"
                )

            result = await conversations_collection.update_one(
                {"_id": ObjectId(conversation_id), "user_id": user_id},
                {"$set": {"title": clean, "updated_at": datetime.now()}}
            )
            if result.matched_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            return {"_id": conversation_id, "title": clean}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    @staticmethod
    async def submit_message_feedback(
        conversation_id: str, message_id: str, user_id: str,
        rating: str | None, reason: str | None,
    ):
        """Set (or clear, when rating is None) the thumbs up/down on one
        assistant message.

        Feedback lives as flat fields on the message document rather than in
        its own collection — it's strictly 1:1 with a message and is always
        read alongside one, matching how cost_usd/stop_reason are already
        stored. Clearing writes an explicit None instead of $unset so the
        export query below can filter on rating values directly.
        """
        try:
            conv = await conversations_collection.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })
            if not conv:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )

            # role: "model" is how assistant messages are stored (see
            # ChatService Step 8) — scoping the update to it means a user
            # can't rate their own prompt.
            result = await messages_collection.update_one(
                {
                    "_id": ObjectId(message_id),
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "role": "model",
                },
                {"$set": {
                    "rating": rating,
                    "rating_reason": reason,
                    "rated_at": datetime.now(),
                }}
            )
            if result.matched_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Message not found"
                )
            return {"rating": rating, "reason": reason}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    @staticmethod
    async def delete_conversation(conversation_id: str, user_id: str):
        """Delete a conversation and all its messages"""
        try:
            # Verify conversation belongs to user
            conv = await conversations_collection.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })
            if not conv:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            
            # Find all messages with attachments
            messages_cursor = messages_collection.find({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "attachments": {"$exists": True, "$ne": []}
            })
            
            # Delete associated files from Cloudinary
            from utils.cloudinary_handler import CloudinaryHandler
            cloudinary_handler = CloudinaryHandler()
            
            async for msg in messages_cursor:
                if msg.get("attachments"):
                    for attachment in msg["attachments"]:
                        if "cloudinary_public_id" in attachment:
                            try:
                                print(f"Deleting Cloudinary file: {attachment['cloudinary_public_id']}")
                                await cloudinary_handler.delete_file(attachment['cloudinary_public_id'])
                            except Exception as e:
                                print(f"Error deleting Cloudinary file {attachment['cloudinary_public_id']}: {e}")
            
            # Delete conversation and messages
            await conversations_collection.delete_one({"_id": ObjectId(conversation_id)})
            await messages_collection.delete_many({
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            return {"message": "Conversation and associated media deleted"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
