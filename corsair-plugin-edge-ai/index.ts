import { createPlugin } from "@corsairdev/corsair";

export default createPlugin({
  name: "Edge AI Intent Classifier",
  description: "Classify user prompts and route to ODA/Hybrid/Cloud.",
  actions: {
    classify: {
      description: "Classify the intent of a user prompt.",
      params: {
        text: { type: "string", description: "User prompt to classify." },
      },
      async handler({ text }, { corsair }) {
        // Call the MCP server
        const response = await fetch("http://localhost:8080/mcp", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            method: "tools/classify",
            params: { text },
          }),
        });
        const { intent, confidence, route_recommendation } = await response.json();

        // Log the classification for auditing
        await corsair.logger.info(`Classified intent: ${intent} (confidence: ${confidence})`);

        return { intent, confidence, route_recommendation };
      },
    },
    route: {
      description: "Route a task based on intent and confidence.",
      params: {
        intent: { type: "string" },
        confidence: { type: "number" },
      },
      async handler({ intent, confidence }, { corsair }) {
        // Call the MCP server
        const route = await fetch("http://localhost:8080/mcp", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            method: "tools/route",
            params: { intent, confidence },
          }),
        }).then((res) => res.json());

        // Require approval for Cloud LLM routes
        if (route === "Cloud LLM") {
          await corsair.permissions.requestApproval({
            action: "route_to_cloud",
            description: `Routing to Cloud LLM for intent: ${intent}`,
          });
        }

        return { route };
      },
    },
  },
  // Define permission modes for your actions
  permissions: {
    classify: { defaultMode: "open" }, // Safe to run automatically
    route: {
      defaultMode: "cautious", // Require approval for Cloud routes
      rules: [
        {
          if: { route: "Cloud LLM" },
          then: { mode: "strict" }, // Block or require approval
        },
      ],
    },
  },
});
