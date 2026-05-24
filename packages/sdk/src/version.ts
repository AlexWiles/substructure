declare const __SDK_VERSION__: string | undefined;

export const SDK_VERSION: string = typeof __SDK_VERSION__ !== "undefined" ? __SDK_VERSION__ : "0.0.0-dev";
