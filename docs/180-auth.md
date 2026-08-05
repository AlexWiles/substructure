---
title: Authentication
group: Operations
---

Every request into the engine has a caller, and every caller belongs to a
tenant. A session records one owner when it is created, and the engine checks
every later input against that owner. In the other direction, the engine signs
each decision it sends your worker.

## Callers

| Caller | Who | Credential |
| --- | --- | --- |
| Frontend | An end-user client. | An HS256 bearer JWT. |
| Machine | Your backend, or an operator. | An API key (`SUBSTRUCTURE_API_KEY`). |
| System | The engine itself. | Internal. It never goes over the wire. |

System has the most privilege, then Machine, then Frontend.

## Client tokens

A client calls the client API with `Authorization: Bearer <jwt>`. The engine
verifies the token as HS256 against the configured issuer and audience, using
`CLIENT_TOKEN_HS256_SECRET`. It reads the session owner from the `sub` claim.

Your backend creates these tokens through the machine API, using its own API
key:

```http
POST /api/machine/client-tokens
Authorization: Bearer <SUBSTRUCTURE_API_KEY>

{ "identity": { "id": "user_42" }, "ttl_seconds": 600 }
```

`identity.id` is required. It becomes the session owner. The response holds the
token and its expiry, in Unix seconds:

```json
{ "token": "<jwt>", "expires_at": 1784000000 }
```

No request carries a `tenant_id`. The API key that authenticates sets the
tenant, and your backend sets the identity. A client chooses neither.

## Session identity

The engine fixes a session's owner when it creates the session. It checks every
later input against that owner: the tenant must match, and a Frontend caller
must own the session.

The worker receives that owner as `DecisionRequest.identity`. It holds the id
and the metadata, but not the tenant:

```typescript
type WorkerIdentity = { id?: string; metadata?: Record<string, string> }
```

The engine sets this once and vouches for it, so the worker reads it and does
not verify it. It is the owner, not the caller of this request.

## Patterns

### One session per user

Authenticate the user in your own app, create a token that carries their id, and
give it to the browser. That user owns every session the browser opens, and the
worker reads that id on each decision.

### Limit by identity

Read `identity` on each decision to give an owner only their own data, or to
choose tools by tenant.

```javascript
function decide({ trigger, proposed, identity }) {
    if (trigger.type === "tool.execute" && trigger.name === "list_files") {
        return { actions: [{ type: "tool.result", result: filesFor(identity.id) }] };
    }
    return proposed;
}
```

## Worker signing

When a worker has a signing secret, the engine signs each decision it POSTs:

```
X-Substructure-Signature: sha256=<hex HMAC-SHA256 of the body>
```

Your worker computes the same HMAC with the same secret and refuses a request
that does not match. When you run the engine yourself, name the variable that
holds the secret with `signing_secret_env` on the agent's `[agent.<id>]` section.
The engine calls a worker that names no variable without a signature, because a
secret only the engine knows would prove nothing. In the cloud, the deployment
creates the secret. Worker responses are not signed. The engine trusts the
connection it opened.

## Rules

- A Frontend caller acts only on a session it owns.
- A Frontend caller ends only client-handled tool calls, never LLM calls.
- A worker decision needs a Machine or System caller.
- To resume an interrupt, a caller needs at least the privilege of the caller
  that raised it.

## Next

- [Cloud](./170-cloud.md): API keys and signing secrets for a hosted app.
- [Client-side tools](./90-client-tools.md): why only the owner can end them.
- [Protocol](./150-protocol.md): `SessionOwner`, `identity`, and the signature.
