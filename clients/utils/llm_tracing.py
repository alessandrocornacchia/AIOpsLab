import os
import sys
import base64
from langfuse import Langfuse
import openlit

# Langfuse + Autogen guide: https://langfuse.com/integrations/frameworks/autogen
#   The above resulted in not being able to turn off console logs.
#   Openlit does not allow us to turn of console logs unless we export 'OTEL_EXPORTER_OTLP_ENDPOINT'.
#   We can set it by following this guide: https://docs.openlit.io/latest/connections/langfuse

def initLangFuse():
    print("Initializing Langfuse")

    langfuse = Langfuse(
        blocked_instrumentation_scopes=["autogen SingleThreadedAgentRuntime"]
    )

    # Verify connection
    if not langfuse.auth_check():
        print("Authentication failed. Please check your Langfuse credentials and host.")
        sys.exit(1)

    print("Langfuse client is authenticated and ready!")

    # Configure authHeader
    LANGFUSE_PUBLIC_KEY = os.environ['LANGFUSE_PUBLIC_KEY']
    LANGFUSE_SECRET_KEY = os.environ['LANGFUSE_SECRET_KEY']
    LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    # Initialize OpenLIT instrumentation.
    # The disable_batch flag is set to true to process traces immediately.
    openlit.init(disable_batch=True)
