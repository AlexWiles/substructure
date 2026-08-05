---
title: Authentication
group: Running it
---

Every request into the engine has a caller. Every caller belongs to a tenant.

A session records one owner when it is created. The engine checks every later
input against that owner. In the other direction, the engine signs each decision
it sends your worker.

## Callers

| Caller | Who | Credential |
| --- | --- | --- |
| Frontend | An end-user client. | An HS256 bearer JWT. |
| Machine | Your backend, or an operator. | An API key. |
| System | The engine itself. | Internal. |

System has the most privilege, then Machine, then Frontend.

## Client tokens

A client calls the client API with `Authorization: Bearer <jwt>`.

The engine verifies the token as HS256 against the configured issuer and
audience, using `CLIENT_TOKEN_HS256_SECRET`. It reads the session owner from the
`sub` claim.

Your backend creates these tokens through the machine API, using its own API
key.

```http
POST /api/machine/client-tokens
Authorization: Bearer <SUBSTRUCTURE_API_KEY>

{ "identity": { "id": "user_42" }, "ttl_seconds": 600 }
```

`identity.id` is required. It becomes the session owner. The response holds the
token and its expiry in Unix seconds.

```json
{ "token": "<jwt>", "expires_at": 1784000000 }
```

No request carries a `tenant_id`. The API key sets the tenant. Your backend sets
the identity. A client chooses neither.

Never send an API key to a browser.

## Session identity

The engine fixes a session's owner when it creates the session. It checks every
later input against that owner. The tenant must match, and a Frontend caller
must own the session.

The worker receives that owner as `DecisionRequest.identity`. It holds the id
and the metadata, not the tenant.

```typescript
type WorkerIdentity = { id?: string; metadata?: Record<string, string> }
```

The engine sets this once and vouches for it. Read it without verifying it. It
is the owner, not the caller of this request.

## Patterns

### One session per user

Authenticate the user in your own app. Create a token that carries their id and
give it to the browser. That user owns every session the browser opens.

### Limit by identity

Read `identity` on each decision to give an owner only their own data.

```javascript
function decide({ trigger, proposed, identity }) {
    if (trigger.type === "tool.execute" && trigger.name === "list_files") {
        return { actions: [{ type: "tool.result", result: filesFor(identity.id) }] };
    }
    return proposed;
}
```

## Worker signing

When an agent has a signing secret, the engine signs each decision it POSTs.

```
X-Substructure-Signature: sha256=<hex HMAC-SHA256 of the body>
```

Your worker computes the same HMAC and refuses a request that does not match.
See [Workers](./50-workers.md#verify-the-signature).

| Where the engine runs | Where the secret comes from |
| --- | --- |
| The cloud | The deployment creates one per agent. Read it with `subs agents show <id>`. |
| Your own machine | `signing_secret_env` on the agent names the variable. |

An agent that names no variable on a local engine gets unsigned requests.

Worker responses are not signed. The engine trusts the connection it opened.

## Rules

- A Frontend caller acts only on a session it owns.
- A Frontend caller ends only client-handled tool calls.
- A worker decision needs a Machine or System caller.
- To resume an interrupt, a caller needs at least the privilege of the caller
  that raised it.

## Next

- [Cloud](./170-cloud.md): API keys and signing secrets for a hosted project.
- [Self-hosting](./180-self-hosting.md): setting these up yourself.
- [REST API](./250-api.md): the endpoints these tokens reach.
