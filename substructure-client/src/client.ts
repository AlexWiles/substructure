import type {
  SubmitRequest,
  SubmitResponse,
  RegisterRequest,
  RegisterResponse,
} from "./types";

export interface ClientOptions {
  baseUrl: string;
}

export class Client {
  private baseUrl: string;

  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
  }

  async submit(request: SubmitRequest): Promise<SubmitResponse> {
    const resp = await fetch(`${this.baseUrl}/workers/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    return (await resp.json()) as SubmitResponse;
  }

  async register(request: RegisterRequest): Promise<RegisterResponse> {
    const resp = await fetch(`${this.baseUrl}/workers/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    return (await resp.json()) as RegisterResponse;
  }
}
