import src.api.routes
from src.api.app import app
from src.config import SERVER_HOST, SERVER_PORT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
