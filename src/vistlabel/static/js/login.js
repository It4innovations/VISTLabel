/* Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
 * SPDX-License-Identifier: MIT */
// Sign-in form: posts JSON to /api/v1/auth/login and follows the ?next= target.

const assetQuery = new URL(import.meta.url).search;
const [{ apiFetch }, { setUpThemeToggle }] = await Promise.all([
  import(`./session.js${assetQuery}`),
  import(`./theme.js${assetQuery}`),
]);

setUpThemeToggle("theme-toggle");

const form = document.getElementById("login-form");
const error = document.getElementById("login-error");
const button = document.getElementById("login-button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.textContent = "";
  button.disabled = true;
  try {
    const response = await apiFetch("/api/v1/auth/login", {
      method: "POST",
      allowUnauthenticated: true,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: document.getElementById("username").value,
        password: document.getElementById("password").value,
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(
        typeof payload.detail === "string" ? payload.detail : `Sign-in failed (${response.status})`
      );
    }
    const next = form.dataset.next || "/";
    window.location.href = next.startsWith("/") && !next.startsWith("//") ? next : "/";
  } catch (err) {
    error.textContent = err.message;
    button.disabled = false;
  }
});
