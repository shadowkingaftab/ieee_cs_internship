from mcp.server.fastmcp import FastMCP
from classifier import classify_prompt
from router import get_route, get_route_explanation

# Create an MCP server
mcp = FastMCP("Edge AI Classifier")

@mcp.tool()
def classify(text: str) -> dict:
    """Classify the intent of a user prompt and return confidence scores."""
    result = classify_prompt(text)
    
    # Extract needed information from the dictionary
    intent = result["intent"]
    confidence = result["confidence"]
    
    # Also get the route recommendation to provide full context
    route_recommendation = get_route(intent, confidence)
    
    return {
        "intent": intent,
        "confidence": confidence,
        "latency_ms": result["latency_ms"],
        "route_recommendation": route_recommendation,
        "all_scores": result["all_scores"]
    }

@mcp.tool()
def route(intent: str, confidence: float) -> str:
    """Route a task based on intent and confidence. Returns the recommended route (e.g. Edge ODA, Hybrid, Cloud LLM)."""
    return get_route(intent, confidence)

@mcp.tool()
def explain_route(intent: str, confidence: float, route: str) -> str:
    """Get an explanation for why a specific route was chosen."""
    return get_route_explanation(intent, confidence, route)

if __name__ == "__main__":
    mcp.run()
