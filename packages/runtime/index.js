import { platform, arch } from "os";
import { createRequire } from "module";

const PLATFORMS = {
  "darwin-arm64": "@substructure.ai/runtime-darwin-arm64",
  "darwin-x64": "@substructure.ai/runtime-darwin-x64",
  "linux-arm64": "@substructure.ai/runtime-linux-arm64",
  "linux-x64": "@substructure.ai/runtime-linux-x64",
};

function loadBinding() {
  const require = createRequire(import.meta.url);
  const key = `${platform()}-${arch()}`;
  const localFile = `substructure-napi.${key}.node`;
  const errors = [];

  // Local build output first (development)
  try {
    return require(`./${localFile}`);
  } catch (e) {
    errors.push(e);
  }

  // Platform-specific npm package (production)
  const pkg = PLATFORMS[key];
  if (pkg) {
    try {
      return require(pkg);
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

  async *submitPayload(sessionId, agentId, payloadJson, identityJson, turnId) {
    let resolve;
    let done = false;
    const buffer = [];

    const finished = this._native.submitPayload(
      sessionId, agentId, payloadJson, identityJson, turnId,
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
