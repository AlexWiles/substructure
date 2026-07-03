import type { BackendClientOptions } from "./backend-client";
import { BackendClient } from "./backend-client";
import type { FrontendClientOptions } from "./frontend-client";
import { FrontendClient } from "./frontend-client";

class BackendNamespace {
    client(options: BackendClientOptions): BackendClient {
        return new BackendClient(options);
    }
}

class FrontendNamespace {
    client(options: FrontendClientOptions): FrontendClient {
        return new FrontendClient(options);
    }
}

export class Substructure {
    readonly backend: BackendNamespace;
    readonly frontend: FrontendNamespace;

    constructor() {
        this.backend = new BackendNamespace();
        this.frontend = new FrontendNamespace();
    }
}
