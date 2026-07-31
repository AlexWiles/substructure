---
title: Authentication
group: Operations
---

Every request into the engine resolves to a caller scoped to a tenant. A
session records one owner when it is created, and every later input is checked
against it. In the other direction, the engine signs the decisions it sends
your worker.

## Callers

| Caller | Who | Credential |
| --- | --- | --- |
| Frontend | An end-user client. | An HS256 bearer JWT. |
| Machine | Your backend, or an operator. | An API key (`SUBSTRUCTURE_API_KEY`). |
| System | The engine itself. | Internal; never over the wire. |

Privilege runs System, then Machine, then Frontend.

## Client tokens

A client reaches the client API with `Authorization: Bearer <jwt>`. The engine
verifies the token as HS256 against the configured issuer and audience, using
`CLIENT_TOKEN_HS256_SECRET`, and takes the session owner from its `sub` claim.

Your backend mints these tokens through the machine API, authenticating with
its own API key:

```http
POST /api/machine/client-tokens
Authorization: Bearer <SUBSTRUCTURE_API_KEY>

{ "identity": { "id": "user_42" }, "ttl_seconds": 600 }
```

`identity.id` is required and becomes the session owner. The response is the
token and its expiry, in Unix seconds:

```json
{ "token": "<jwt>", "expires_at": 1784000000 }
```

No request carries a `tenant_id`. The tenant is fixed by the authenticating
API key, and the identity is set by your backend, so a client chooses neither.

## Session identity

A session's owner is fixed when the session is created, and every later input
is checked against it: the tenant must match, and a Frontend caller must own
the session.

The worker receives that owner as `DecisionRequest.identity`, with the id and
metadata but not the tenant:

```typescript
type WorkerIdentity = { id?: string; metadata?: Record<string, string> }
```

It is engine-attested and set once, so the worker reads it rather than
verifying it, and sees the owner, not the per-request caller.

## Patterns

### Per-user sessions

Authenticate the user in your own app, mint a token carrying their id, and hand
it to the browser. Every session it opens is owned by that user, and the worker
reads that id on each decision.

### Scope by identity

Read `identity` on each decision to serve an owner only their own data, or to
pick tools by tenant.

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

Your worker recomputes the HMAC with the same secret and rejects a mismatch.
Locally, name the variable holding it with `signing_secret_env` on the agent's
`[agent.<id>]` section; a worker that names none is called unsigned, because
signing with a secret only the engine knows proves nothing to anybody. In the
cloud the deployment mints it. Worker responses are not signed; the engine
trusts the connection it opened.

## Rules

- A Frontend caller acts only on a session it owns.
- A Frontend caller settles only client-handled tools, never LLM calls.
- Worker decisions require a Machine or System caller.
- Resuming an interrupt needs privilege at least its origin's.

## Next

- [Cloud](./170-cloud.md): API keys and signing secrets for a hosted app.
- [Client-side tools](./90-client-tools.md): why only the owner settles them.
- [Protocol](./150-protocol.md): `SessionOwner`, `identity`, and the signature.
