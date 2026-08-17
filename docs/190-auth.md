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
| Admin | A person who logged in. | A user token. |
| ApiKey | Your backend, or a worker. | An API key. |
| System | The engine itself. | Internal. |

System has the most privilege, then ApiKey and Admin, then Frontend.

ApiKey and Admin differ in who holds the credential: a program holds a key, a
person logs in. Only a worker answers a decision the engine hands out, so only
ApiKey may. An admin administers a session and does not run the model for it.

## Client tokens

A client calls the client API with `Authorization: Bearer <jwt>`.

The engine verifies the token as HS256 against the configured issuer and
audience, using `CLIENT_TOKEN_HS256_SECRET`. The `sub` claim names the person;
the issuer is stamped here, never read from the token.

Your backend creates these tokens through the machine API, using its own API
key.

```http
POST /api/machine/client-tokens
Authorization: Bearer <SUBSTRUCTURE_API_KEY>

{ "identity": { "id": "user_42", "visibility": "private" }, "ttl_seconds": 600 }
```

`identity.id` is required. It names the person the session runs for, under the
`app` issuer — the one your application vouches for. You choose the id and we
sign it, so a browser holding the token cannot rename itself.

`identity.visibility` says whether anyone but that person can read the session,
and it decides whether their own credentials may answer there. Only you know
what surface the token is for: a chat window is `private`, an agent embedded in
a shared team inbox is not. It defaults to `shared`, which reaches no personal
credential, so a mistake is a refusal rather than one user's mailbox answering
in front of their colleagues.

The response holds the token and its expiry in Unix seconds.

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

The worker receives that owner as `DecisionRequest.identity`. It holds who the
session is for and the metadata, not the tenant.

```typescript
type Subject = {
    /** Which source named this person: "slack", "app", "operator", "cli", … */
    issuer: string
    /** That source's own name for them. Never parsed. */
    id: string
}
type WorkerIdentity = {
    /** Absent ⇒ nobody is behind this session: a schedule, a key, the engine. */
    subject?: Subject
    /** Whether anyone else can read the conversation. */
    visibility: "shared" | "private"
    metadata?: Record<string, string>
}
```

The engine sets this once and vouches for it. Read it without verifying it. It
is who the session is for, not the caller of this request.

The issuer is half the name. An id is only unique within the source that minted
it, so your application's `bob` and a workspace's `bob` are two people, and
comparing ids alone would conflate them.

## Patterns

### One session per user

Authenticate the user in your own app. Create a token that carries their id and
give it to the browser. That user owns every session the browser opens.

### Limit by identity

Read `identity` on each decision to give an owner only their own data. Check the
issuer as well as the id: an operator is not the end user with that name.

```javascript
function decide({ trigger, proposed, identity }) {
    if (trigger.type === "tool.execute" && trigger.name === "list_files") {
        if (identity.subject?.issuer !== "app") {
            return { actions: [{ type: "tool.error", error: "not an end user" }] };
        }
        const text = filesFor(identity.subject.id);
        return { actions: [{ type: "tool.result", result: { content: [{ type: "text", text: text }] } }] };
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
| The cloud | The deployment creates one per agent. Read it with `subs agents secret <id>`. |
| Your own machine | `signing_secret_env` on the agent names the variable. |

An agent that names no variable on a local engine gets unsigned requests.

Worker responses are not signed. The engine trusts the connection it opened.

## Rules

- A Frontend caller acts only on a session it owns.
- A Frontend caller ends only client-handled tool calls.
- A worker decision needs an ApiKey or System caller. An Admin caller cannot
  submit one, and cannot answer an `llm.execute`.
- Cancelling a session needs any caller but Frontend.
- A session records the kind of owner as well as the name. An end user opens
  only a session an end user owns, so an admin and a user with the same name
  are different owners.
- A session an operator starts is owned by that credential, and named by it.
- To resume an interrupt, a caller needs at least the privilege of the caller
  that raised it.

## Next

- [Cloud](./170-cloud.md): API keys and signing secrets for a hosted project.
- [Self-hosting](./180-self-hosting.md): setting these up yourself.
- [REST API](./250-api.md): the endpoints these tokens reach.
