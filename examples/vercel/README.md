# vercel

Deploys the worker as a single Vercel serverless function. One file
under `api/`, no framework. Each request runs one agent decision and
returns; the backend keeps the durable state, so the function can scale
to zero between turns.

Use this shape when you want to host the worker on Vercel without
pulling in Next.js.

## Deploy

```sh
pnpm install
pnpm deploy
```

Set the signing secret on the deployment, then point a Substructure
backend at `https://<your-deploy>.vercel.app/api/agent`:

```sh
vercel env add SIGNING_SECRET
```

## Trigger a turn

`client.ts` submits a turn against the backend:

```sh
export SUBSTRUCTURE_URL=https://api.substructure.ai
export SUBSTRUCTURE_API_KEY=...
pnpm client
```
