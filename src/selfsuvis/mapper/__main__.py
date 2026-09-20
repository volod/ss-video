import uvicorn

from selfsuvis.mapper.main import app
from selfsuvis.pipeline.core.env import env_int, load_layered_env

load_layered_env(anchor_file=__file__)

if __name__ == "__main__":
    port = env_int("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
