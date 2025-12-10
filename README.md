# Gemini MCP Chat Backend - Modular Architecture

## Project Structure

```
backend/
├── backend.py              # Main FastAPI application
├── database.py             # MongoDB connection
├── auth.py                 # JWT & password hashing utilities
├── middleware.py           # Authentication middleware
├── mcp_server.py          # Local MCP tools
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
│
├── models/                # Data models
│   ├── user.py           # User model & schemas
│   ├── conversation.py   # Conversation model
│   └── message.py        # Message model
│
├── controllers/           # Business logic
│   ├── auth_controller.py        # Authentication logic
│   ├── conversation_controller.py # Conversation operations
│   ├── chat_controller.py        # Gemini + MCP chat logic
│   └── mcp_controller.py         # MCP operations
│
└── routes/                # API routes
    ├── auth_routes.py            # /auth/* endpoints
    ├── conversation_routes.py    # /api/conversations/* endpoints
    ├── chat_routes.py            # /chat endpoint
    └── mcp_routes.py             # /mcp/* endpoints
```

## Features

- ✅ Modular MVC architecture
- ✅ Separate controllers for business logic
- ✅ Clean route definitions
- ✅ JWT authentication
- ✅ User-specific chat history
- ✅ MongoDB integration
- ✅ Gemini + MCP integration

## API Endpoints

### Authentication (`/auth`)
- `POST /auth/signup` - Register new user
- `POST /auth/login` - Login & get JWT token
- `GET /auth/me` - Get current user (protected)

### Conversations (`/api/conversations`)
- `GET /api/conversations` - Get user's conversations (protected)
- `POST /api/conversations` - Create conversation (protected)
- `GET /api/conversations/:id/messages` - Get messages (protected)
- `DELETE /api/conversations/:id` - Delete conversation (protected)

### Chat (`/chat`)
- `POST /chat` - Send message (protected)

### MCP (`/mcp`)
- `POST /mcp/connect` - List MCP tools

## Running the Backend

```bash
cd backend
pip install -r requirements.txt
python backend.py
```

Server runs on `http://localhost:8000`
API docs available at `http://localhost:8000/docs`
