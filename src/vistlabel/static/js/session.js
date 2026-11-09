/* Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
 * SPDX-License-Identifier: MIT */
// Session helpers shared by every page: apiFetch() is fetch() for the
// application's own API -- it carries the same-origin marker header the
// backend requires on state-changing requests and sends an expired
// session to the sign-in page -- and setUpLogout() wires the Sign out
// button.

export const CSRF_HEADER = "X-Requested-With";
export const CSRF_HEADER_VALUE = "VISTLabel";

export function redirectToLogin() {
  const here = window.location.pathname + window.location.search;
  window.location.href = "/login?next=" + encodeURIComponent(here);
}

export async function apiFetch(path, init = {}) {
  const headers = new Headers(init.headers || {});
  headers.set(CSRF_HEADER, CSRF_HEADER_VALUE);
  const response = await fetch(path, { ...init, headers });
  if (response.status === 401 && !init.allowUnauthenticated) {
    redirectToLogin();
  }
  return response;
}

export function setUpLogout(buttonId = "logout-button") {
  const button = document.getElementById(buttonId);
  if (!button) return;
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await apiFetch("/api/v1/auth/logout", { method: "POST", allowUnauthenticated: true });
    } finally {
      window.location.href = "/login";
    }
  });
}
