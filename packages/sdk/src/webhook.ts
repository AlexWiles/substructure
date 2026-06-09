export class WebhookVerificationError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "WebhookVerificationError";
    }
}

export interface VerifyOptions {
    /** Tolerance in seconds (default: 300 = 5 minutes) */
    tolerance?: number;
}

/** Returns the parsed JSON body if valid, throws WebhookVerificationError otherwise. */
export async function verifyWebhookSignature<T = unknown>(
    req: Request,
    secret: string,
    options?: VerifyOptions,
): Promise<T> {
    const timestamp = req.headers.get("x-substructure-timestamp");
    const signature = req.headers.get("x-substructure-signature");

    if (!timestamp || !signature) {
        throw new WebhookVerificationError("Missing signature headers");
    }

    const ts = parseInt(timestamp, 10);
    if (isNaN(ts)) {
        throw new WebhookVerificationError("Invalid timestamp");
    }

    const tolerance = options?.tolerance ?? 300;
    const now = Math.floor(Date.now() / 1000);
    if (Math.abs(now - ts) > tolerance) {
        throw new WebhookVerificationError("Timestamp outside tolerance window");
    }

    const match = signature.match(/^v1=([0-9a-f]+)$/);
    if (!match) {
        throw new WebhookVerificationError("Invalid signature format");
    }
    const receivedSig = match[1];

    const body = await req.text();
    const signingPayload = `${timestamp}.${body}`;

    const key = await crypto.subtle.importKey(
        "raw",
        new TextEncoder().encode(secret),
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["sign"],
    );
    const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signingPayload));
    const expectedSig = Array.from(new Uint8Array(mac), (b) => b.toString(16).padStart(2, "0")).join("");

    if (!timingSafeEqual(receivedSig, expectedSig)) {
        throw new WebhookVerificationError("Signature mismatch");
    }

    return JSON.parse(body) as T;
}

function timingSafeEqual(a: string, b: string): boolean {
    if (a.length !== b.length) return false;
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        result |= a.charCodeAt(i) ^ b.charCodeAt(i);
    }
    return result === 0;
}
