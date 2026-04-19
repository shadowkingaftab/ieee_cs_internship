import re

def split_steps(user_input):
    """
    Intelligent heuristic decomposition without heavy NLP models.
    """
    text = user_input.strip()
    
    # 1. Check for explicit numbering
    if re.search(r'\b\d+[\.\)]\s', text):
        raw_steps = re.split(r'\b\d+[\.\)]\s', text)
    else:
        # 2. Split on major sentence boundaries or conjunctions (avoiding simple commas)
        # We split by periods, semicolons, text newlines, or explicit coordinating adverbs
        # We ensure it looks like a real sentence boundary
        delimiters = r'\n|(?<=[a-z])\.\s+|(?<=[a-z]);\s+|(?<=[a-z])\s+(?:and then|then|furthermore|afterwards|next|finally)\s+'
        raw_steps = re.split(delimiters, text, flags=re.IGNORECASE)
    
    steps = []
    for s in raw_steps:
        # Clean up the parsed string
        clean_s = s.strip(" \t\n\r-.*,")
        if len(clean_s) > 10: # Ensure it's a substantive step
            clean_s = clean_s[0].upper() + clean_s[1:]
            steps.append(clean_s)
    
    # 3. If the user provided a very short prompt, synthesize logical orchestration steps
    # to demonstrate the system's pipeline capabilities.
    if len(steps) < 2:
        core_intent = steps[0] if steps else text
        steps = [
            f"Analyze requirements and scope for: {core_intent}",
            "Design system architecture and necessary data models",
            f"Implement core functionality for {core_intent.lower()}",
            "Configure unit testing and automated CI/CD pipelines",
            "Deploy infrastructure and launch services"
        ]
        
    return steps[:8] # Cap at 8 meaningful steps

def route_step(step):
    """
    Routes a step to the appropriate execution layer.
    """
    step_lower = step.lower()
    
    # 🟢 On-device AI (Deep Green): Execution
    device_keywords = [
        "deploy", "execute", "run", "display", "save", "log", "email", 
        "notification", "download", "fetch", "store", "cache", "update", "launch",
        "install", "configure", "pipeline"
    ]
    
    # 🔴 Cloud LLM (Deep Red): Reasoning / Generation
    llm_keywords = [
        "write", "generate", "analyze", "develop", "code", "design", "create", 
        "elaborate", "summary", "explain", "model", "predict", "train", "infer", 
        "evaluate", "translate", "summarize", "architect", "brainstorm", "draft", "scope"
    ]
    
    # 🟡 Hybrid (Amber/Yellow): Planning / Coordination
    hybrid_keywords = [
        "plan", "organize", "manage", "setup", "arrange", "coordinate", 
        "orchestrate", "prepare", "integrate", "test", "verify", "monitor", "review"
    ]

    llm_score = sum(2 for w in llm_keywords if w in step_lower)
    device_score = sum(2 for w in device_keywords if w in step_lower)
    hybrid_score = sum(1.5 for w in hybrid_keywords if w in step_lower) # slightly lower weight

    if llm_score > device_score and llm_score >= hybrid_score:
        return "🔴 Cloud LLM"
    elif device_score > llm_score and device_score >= hybrid_score:
        return "🟢 On-device AI"
    elif hybrid_score > llm_score and hybrid_score > device_score:
        return "🟡 Hybrid"
    else:
        # Strict fallbacks ensuring logical flow
        if "design" in step_lower or "analyze" in step_lower: return "🔴 Cloud LLM"
        if "test" in step_lower or "integrate" in step_lower: return "🟡 Hybrid"
        return "🟢 On-device AI"

def process_prompt(user_input):
    steps = split_steps(user_input)
    results = []
    for step in steps:
        route = route_step(step)
        results.append((step, route))
    return results
