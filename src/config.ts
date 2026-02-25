export const config = {
    GEMINI_API_KEY: process.env.GEMINI_API_KEY ?? (() => { throw new Error("GEMINI_API_KEY is not set in .env file"); })(),
    PACKAGE_NAME: process.env.PACKAGE_NAME ?? (() => { throw new Error("PACKAGE_NAME is not set in .env file"); })(),
    MENTRAOS_API_KEY: process.env.MENTRAOS_API_KEY ?? (() => { throw new Error("MENTRAOS_API_KEY is not set in .env file"); })(),
    BACKEND_URL: process.env.BACKEND_URL ?? (() => { throw new Error("BACKEND_URL is not set in .env file"); })(),
    BACKEND_AUTH_TOKEN: process.env.BACKEND_AUTH_TOKEN ?? (() => { throw new Error("BACKEND_AUTH_TOKEN is not set in .env file"); })(),
    PORT: parseInt(process.env.PORT || '3000'),
} as const;