with open(".env.sample", "a", encoding="utf-8") as f:
    f.write("\n# --- CORS ----------------------------------------------------\nALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,https://agentx.ankitrajai.in\n")
print("Updated .env.sample")
