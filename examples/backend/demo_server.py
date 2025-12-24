from tanstack_pydantic_ai import create_app
from .demo_agent import DemoAgent

app = create_app(DemoAgent())
