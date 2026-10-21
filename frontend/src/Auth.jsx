/**
 * Sign-in gate in front of the dashboard (FR-21; backend: src/serving/auth.py).
 *
 * Asks /auth/session first. Sign-in off on the backend -> the dashboard, as
 * before sign-in existed. Signed in -> the dashboard, with the account in
 * the masthead. Otherwise -> the login page. If the backend can't be reached
 * at all the dashboard still renders, so IR-07's labelled example data
 * shows instead of a login page that couldn't work anyway.
 */
import { useCallback, useEffect, useState } from "react";
import App from "./App.jsx";
import { AUTH_REQUIRED_EVENT, fetchSession, loginUrl, passwordLogin, signOut } from "./api.js";

export const AUTH_ERRORS = {
  not_allowed: "That account isn't on this dashboard's access list. Ask the owner to add your email address.",
  unverified: "That account has no verified email address, so it can't be checked against the access list.",
  failed: "Sign-in didn't complete. Please try again.",
};

/** The backend's callback redirects here with ?auth_error=<code> on refusal. */
export function readAuthError(search = window.location.search) {
  const code = new URLSearchParams(search).get("auth_error");
  return code ? AUTH_ERRORS[code] ?? AUTH_ERRORS.failed : null;
}

export default function AuthGate() {
  const [session, setSession] = useState(null); // null until /auth/session answers
  const [error] = useState(() => readAuthError());

  const load = useCallback(async () => {
    const r = await fetchSession();
    setSession(r.ok ? r.data : { unreachable: true });
  }, []);

  useEffect(() => {
    load();
    // Drop ?auth_error so a reload doesn't repeat the message.
    const url = new URL(window.location.href);
    if (url.searchParams.has("auth_error")) {
      url.searchParams.delete("auth_error");
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    }
  }, [load]);

  // A 401 from any dashboard call means the session ended mid-use.
  useEffect(() => {
    window.addEventListener(AUTH_REQUIRED_EVENT, load);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, load);
  }, [load]);

  const handleSignOut = async () => {
    await signOut();
    load();
  };

  if (session === null) return <div className="login" aria-busy="true" />;
  if (session.unreachable || !session.auth_enabled) return <App />;
  if (session.user) return <App user={session.user} onSignOut={handleSignOut} />;
  return (
    <LoginPage
      providers={session.providers}
      passwordLogin={session.password_login}
      error={error}
      onSignedIn={load}
    />
  );
}

export function LoginPage({ providers, passwordLogin: showPassword = false, error, onSignedIn }) {
  // undefined until the form is used; then the form's own result (null
  // included) replaces any ?auth_error message from an earlier attempt.
  const [formError, setFormError] = useState(undefined);
  const shownError = formError === undefined ? error : formError;
  return (
    <main className="login">
      <section className="glass login-card" aria-labelledby="login-title">
        <h1 id="login-title">Energy Agent Telemetry</h1>
        <p className="login-sub">Sign in to view the continuous-training dashboard.</p>
        {shownError && <p className="login-error" role="alert">{shownError}</p>}
        {showPassword && <PasswordForm onSignedIn={onSignedIn} onError={setFormError} />}
        {showPassword && providers.length > 0 && (
          <div className="login-divider" role="separator"><span>or</span></div>
        )}
        {providers.length > 0 && (
          <div className="login-providers">
            {providers.map((p) => (
              <a key={p.id} className="login-provider" href={loginUrl(p.id)}>
                <ProviderIcon id={p.id} />
                <span>Continue with {p.label}</span>
              </a>
            ))}
          </div>
        )}
        <p className="login-note">
          Access is limited to approved accounts.
          {showPassword && " Password accounts are created by the dashboard's owner."}
        </p>
      </section>
    </main>
  );
}

const PASSWORD_ERRORS = {
  unreachable: "Couldn't reach the server. Check your connection and try again.",
  timeout: "The server took too long to answer. Try again.",
};

function PasswordForm({ onSignedIn, onError }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    onError(null);
    const r = await passwordLogin(email.trim(), password);
    setBusy(false);
    if (r.ok) {
      onSignedIn();
      return;
    }
    setPassword("");
    onError(r.detail ?? PASSWORD_ERRORS[r.reason] ?? AUTH_ERRORS.failed);
  };

  return (
    <form className="login-form" onSubmit={submit}>
      <label className="login-field">
        <span>Email</span>
        <input type="email" autoComplete="username" required value={email}
               onChange={(e) => setEmail(e.target.value)} />
      </label>
      <label className="login-field">
        <span>Password</span>
        <input type="password" autoComplete="current-password" required value={password}
               onChange={(e) => setPassword(e.target.value)} />
      </label>
      <button type="submit" className="btn" disabled={busy}>
        {busy ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}

/** Each provider's own mark, as their sign-in branding guidelines ask for. */
function ProviderIcon({ id }) {
  if (id === "google") {
    return (
      <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
        <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z" />
        <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z" />
        <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z" />
        <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z" />
      </svg>
    );
  }
  if (id === "microsoft") {
    return (
      <svg width="18" height="18" viewBox="0 0 21 21" aria-hidden="true">
        <rect x="1" y="1" width="9" height="9" fill="#F25022" />
        <rect x="11" y="1" width="9" height="9" fill="#7FBA00" />
        <rect x="1" y="11" width="9" height="9" fill="#00A4EF" />
        <rect x="11" y="11" width="9" height="9" fill="#FFB900" />
      </svg>
    );
  }
  if (id === "github") {
    return (
      <svg width="18" height="18" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
      </svg>
    );
  }
  return null;
}
