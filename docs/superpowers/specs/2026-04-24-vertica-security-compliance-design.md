# Vertica MCP Security Compliance Design

## Goal Description
To establish a production-grade security posture for the Vertica MCP Server by enforcing password-less authentication at all connection boundaries. This addresses two connection hops:
1. AI System to MCP Server
2. MCP Server to Vertica Database

## 1. AI ↔ MCP Authentication (JWT Validation)
Replaces the static API key mechanism with dynamic JSON Web Token (JWT) validation, sourced from an external Identity Provider (e.g., Auth0, Clerk).

### Architecture
- **Dependency Addition:** Integrate `PyJWT` and `cryptography` to handle secure JWT decoding and RSA signature validation.
- **Middleware Refactor:** Overhaul the `AuthMiddleware` in `server.py` to validate JWTs instead of a static Bearer token.
- **JWKS Caching:** The middleware will dynamically fetch and securely cache the Identity Provider's JSON Web Key Set (JWKS) to verify token signatures without introducing significant latency.
- **Error Handling:** Standardized HTTP 401 Unauthorized responses for missing, expired, or invalid tokens.

### Configuration (Environment Variables)
- `JWT_ISSUER`: The Identity Provider's issuer URL (e.g., `https://my-tenant.auth0.com/`).
- `JWT_AUDIENCE`: The expected audience for the token.
- `MCP_API_KEY`: Will be deprecated/removed in favor of JWT configuration.

## 2. MCP ↔ Vertica Database Authentication (Multi-Auth Strategy)
Enhances the `connection.py` module to fully support, validate, and document Vertica's native password-less authentication mechanisms. 

### Supported Authentication Modes
Configured via `VERTICA_AUTH_MODE`.
- `oauth`: Passes a JWT/OAuth token (`VERTICA_OAUTH_TOKEN`) directly to Vertica.
- `mtls`: Mutual TLS authentication using client certificates (`VERTICA_SSL_CERT`, `VERTICA_SSL_KEY`).
- `kerberos`: Enterprise GSSAPI authentication (`VERTICA_KERBEROS_SERVICE`, `VERTICA_KERBEROS_HOST`).
- `basic`: Username/Password authentication (retained for local development and backward compatibility).

### Architecture
- **Validation:** The `VerticaConnectionPool` connection logic will ensure that if a secure mode (like `mtls` or `oauth`) is selected, the required credentials (e.g., certificates or tokens) are strictly present before attempting a connection.
- **Documentation:** The `.env.example` file and project README will be thoroughly documented to explain setup requirements for each of these security modes.

## Verification Plan
1. **AI ↔ MCP:** Send requests with valid, expired, and invalid JWTs to ensure proper authorization enforcement.
2. **MCP ↔ Vertica:** Ensure connection initialization gracefully handles the various `VERTICA_AUTH_MODE` configurations, rejecting startup if required secure parameters (like mTLS certs) are missing when selected.
