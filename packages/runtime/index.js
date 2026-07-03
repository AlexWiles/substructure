import { platform, arch } from "os";
import { createRequire } from "module";

const PLATFORMS = {
  "darwin-arm64": { pkg: "@substructure.ai/runtime-darwin-arm64", suffix: "darwin-arm64" },
  "darwin-x64": { pkg: "@substructure.ai/runtime-darwin-x64", suffix: "darwin-x64" },
  "linux-arm64": { pkg: "@substructure.ai/runtime-linux-arm64-gnu", suffix: "linux-arm64-gnu" },
  "linux-x64": { pkg: "@substructure.ai/runtime-linux-x64-gnu", suffix: "linux-x64-gnu" },
};

function loadBinding() {
  const require = createRequire(import.meta.url);
  const key = `${platform()}-${arch()}`;
  const entry = PLATFORMS[key];
  const localFile = entry ? `substructure-napi.${entry.suffix}.node` : `substructure-napi.${key}.node`;
  const errors = [];

  // Local build output first (development)
  try {
    return require(`./${localFile}`);
  } catch (e) {
    errors.push(e);
  }

  // Platform-specific npm package (production)
  if (entry) {
    try {
      return require(entry.pkg);
    } catch (e) {
      errors.push(e);
    }
  }

  throw new Error(
    `Could not load substructure-napi for ${key}.\n` +
    `Run "napi build --platform" or install the platform package.\n` +
    errors.map((e) => e.message).join("\n"),
  );
}

const binding = loadBinding();
const NativeRuntime = binding.EmbeddedRuntime;

export class EmbeddedRuntime {
  constructor(options) {
    this._native = new NativeRuntime(options);
  }

  registerWorker(tenantId, agentIds, callback) {
    return this._native.registerWorker(tenantId, agentIds, callback);
  }

  submitPayload(sessionId, agentId, payloadJson, identityJson, turnId) {
    return this._native.submitPayload(sessionId, agentId, payloadJson, identityJson, turnId);
  }

  settleEffect(sessionId, tenantId, kind, id, attempt, resultJson, responseJson, errorMessage, retryable) {
    return this._native.settleEffect(
      sessionId, tenantId, kind, id, attempt, resultJson, responseJson, errorMessage, retryable,
    );
  }

  async *streamSession(tenantId, sessionId, turnId, sequenceAfter) {
    let resolve;
    let done = false;
    const buffer = [];

    const finished = this._native.streamSession(
      tenantId, sessionId, turnId, sequenceAfter,
      (json) => {
        buffer.push(json);
        resolve?.();
      },
    ).then(() => { done = true; resolve?.(); });

    while (!done || buffer.length > 0) {
      if (buffer.length > 0) {
        yield buffer.shift();
      } else {
        await new Promise((r) => { resolve = r; });
      }
    }

    await finished;
  }

  shutdown() {
    return this._native.shutdown();
  }
}
