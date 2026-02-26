from main_app import app
from fastapi.openapi.utils import get_openapi
import yaml

schema = get_openapi(title=app.title, version=app.version, routes=app.routes)

with open("api_doc.yaml", "w", encoding="utf-8") as f:
    yaml.dump(schema, f, sort_keys=False)

print("api_doc.yaml generated")
