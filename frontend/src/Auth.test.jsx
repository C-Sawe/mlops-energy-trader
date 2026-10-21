/**
 * The sign-in gate (Auth.jsx, FR-21): which of login page / dashboard renders for
 * each /auth/session answer, the refusal messages, sign-out, and the
 * return to the login page when a session ends mid-use. The dashboard
 * itself is stubbed: its own rendering is App.render.test.jsx's job.
 */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, fetchSession: vi.fn(), signOut: vi.fn(), passwordLogin: vi.fn() };
});
vi.mock("./App.jsx", () => ({
  default: ({ user, onSignOut }) => (
    <div>
      dashboard for {user ? user.email : "anyone"}
      {onSignOut && <button onClick={onSignOut}>Sign out</button>}
    </div>
  ),
}));

import * as api from "./api.js";
import AuthGate, { AUTH_ERRORS, readAuthError } from "./Auth.jsx";

const ok = (data) => ({ ok: true, data });
const PROVIDERS = [
  { id: "google", label: "Google" },
  { id: "microsoft", label: "Microsoft" },
  { id: "github", label: "GitHub" },
];
const USER = { email: "caleb@example.com", name: "Caleb", provider: "google" };
const signedOut = ok({ auth_enabled: true, providers: PROVIDERS, password_login: false, user: null });
const signedIn = ok({ auth_enabled: true, providers: PROVIDERS, password_login: false, user: USER });
const passwordOnly = ok({ auth_enabled: true, providers: [], password_login: true, user: null });

async function renderGate() {
  render(<AuthGate />);
  await act(async () => {}); // let /auth/session resolve
}

beforeEach(() => {
  window.history.replaceState(null, "", "/");
});
afterEach(() => {
  vi.clearAllMocks();
});

describe("AuthGate", () => {
  it("shows the dashboard directly when sign-in is off", async () => {
    api.fetchSession.mockResolvedValue(ok({ auth_enabled: false, providers: [], user: null }));
    await renderGate();
    expect(screen.getByText("dashboard for anyone")).toBeTruthy();
    expect(screen.queryByText("Sign out")).toBeNull();
  });

  it("falls back to the dashboard when the backend is unreachable (IR-07)", async () => {
    api.fetchSession.mockResolvedValue({ ok: false, reason: "unreachable" });
    await renderGate();
    expect(screen.getByText("dashboard for anyone")).toBeTruthy();
  });

  it("shows a sign-in link for each configured provider", async () => {
    api.fetchSession.mockResolvedValue(signedOut);
    await renderGate();
    const links = screen.getAllByRole("link");
    expect(links.map((a) => a.textContent)).toEqual([
      "Continue with Google", "Continue with Microsoft", "Continue with GitHub",
    ]);
    expect(links[0].getAttribute("href")).toBe(api.loginUrl("google"));
    expect(screen.queryByText(/dashboard for/)).toBeNull();
  });

  it("shows the dashboard with the account when signed in, and signs out", async () => {
    api.fetchSession.mockResolvedValue(signedIn);
    api.signOut.mockResolvedValue({ ok: true, data: null });
    await renderGate();
    expect(screen.getByText("dashboard for caleb@example.com")).toBeTruthy();

    api.fetchSession.mockResolvedValue(signedOut);
    await act(async () => { fireEvent.click(screen.getByText("Sign out")); });
    expect(api.signOut).toHaveBeenCalledOnce();
    expect(screen.getByText("Continue with Google")).toBeTruthy();
  });

  it("returns to the login page when a dashboard call reports the session ended", async () => {
    api.fetchSession.mockResolvedValue(signedIn);
    await renderGate();

    api.fetchSession.mockResolvedValue(signedOut);
    await act(async () => { window.dispatchEvent(new Event(api.AUTH_REQUIRED_EVENT)); });
    expect(screen.getByText("Continue with GitHub")).toBeTruthy();
  });

  it("explains a refused sign-in and clears it from the URL", async () => {
    window.history.replaceState(null, "", "/?auth_error=not_allowed");
    api.fetchSession.mockResolvedValue(signedOut);
    await renderGate();
    expect(screen.getByRole("alert").textContent).toBe(AUTH_ERRORS.not_allowed);
    expect(window.location.search).toBe("");
  });
});

describe("email and password", () => {
  const fill = (email, password) => {
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: email } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: password } });
  };

  it("shows the form only when the backend enables it", async () => {
    api.fetchSession.mockResolvedValue(signedOut);
    await renderGate();
    expect(screen.queryByLabelText("Password")).toBeNull();
  });

  it("shows the form without a divider or buttons when it is the only method", async () => {
    api.fetchSession.mockResolvedValue(passwordOnly);
    await renderGate();
    expect(screen.getByLabelText("Password")).toBeTruthy();
    expect(screen.queryByRole("separator")).toBeNull();
    expect(screen.queryAllByRole("link")).toHaveLength(0);
  });

  it("shows the form above the provider buttons when both are on", async () => {
    api.fetchSession.mockResolvedValue(ok({ ...signedOut.data, password_login: true }));
    await renderGate();
    expect(screen.getByRole("separator").textContent).toBe("or");
    expect(screen.getAllByRole("link")).toHaveLength(3);
  });

  it("signs in and loads the dashboard", async () => {
    api.fetchSession.mockResolvedValue(passwordOnly);
    api.passwordLogin.mockResolvedValue(ok({ email: "caleb@example.com", name: null, provider: "password" }));
    await renderGate();

    fill(" caleb@example.com ", "correct horse battery staple");
    api.fetchSession.mockResolvedValue(signedIn);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Sign in" })); });

    expect(api.passwordLogin).toHaveBeenCalledWith("caleb@example.com", "correct horse battery staple");
    expect(screen.getByText("dashboard for caleb@example.com")).toBeTruthy();
  });

  it("shows the server's message and clears the password on a refusal", async () => {
    api.fetchSession.mockResolvedValue(passwordOnly);
    api.passwordLogin.mockResolvedValue({ ok: false, reason: "rejected", detail: "Incorrect email or password." });
    await renderGate();

    fill("caleb@example.com", "wrong");
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Sign in" })); });

    expect(screen.getByRole("alert").textContent).toBe("Incorrect email or password.");
    expect(screen.getByLabelText("Password").value).toBe("");
    expect(screen.getByLabelText("Email").value).toBe("caleb@example.com");
  });

  it("replaces an earlier sign-in error once the form is used", async () => {
    window.history.replaceState(null, "", "/?auth_error=not_allowed");
    api.fetchSession.mockResolvedValue(passwordOnly);
    api.passwordLogin.mockResolvedValue({ ok: false, reason: "unreachable" });
    await renderGate();
    expect(screen.getByRole("alert").textContent).toBe(AUTH_ERRORS.not_allowed);

    fill("caleb@example.com", "anything");
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Sign in" })); });
    expect(screen.getByRole("alert").textContent).toMatch(/Couldn't reach the server/);
  });
});

describe("readAuthError", () => {
  it("maps known codes and treats unknown ones as a failed sign-in", () => {
    expect(readAuthError("")).toBeNull();
    expect(readAuthError("?auth_error=unverified")).toBe(AUTH_ERRORS.unverified);
    expect(readAuthError("?auth_error=something_new")).toBe(AUTH_ERRORS.failed);
  });
});
