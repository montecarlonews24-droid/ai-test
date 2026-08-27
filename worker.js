import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// Runs the model off the main thread so the UI stays responsive
// while the model is generating.
new webllm.WebWorkerMLCEngineHandler();
