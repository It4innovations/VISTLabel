/* Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
 * SPDX-License-Identifier: MIT */
// Text search produces temporary proposals. Only explicit acceptance creates
// an object; a model's suggested class never becomes a label automatically.
export function setUpConceptSearch({ getContext, request, preview, accept, reportStatus = () => {} }) {
  const text = document.getElementById("concept-text");
  const search = document.getElementById("concept-search-button");
  const status = document.getElementById("concept-status");
  const results = document.getElementById("concept-results");
  const select = document.getElementById("concept-proposal");
  const label = document.getElementById("concept-label");
  const acceptButton = document.getElementById("concept-accept-button");
  const dismiss = document.getElementById("concept-dismiss-button");
  let generation = 0;
  let response = null;
  let searching = false;
  let accepting = false;
  let searchStartedAt = 0;

  function updateCapabilities() {
    const backend = getContext().backend;
    const reason = !backend?.available
      ? (backend?.unavailable_reason || "Select an available backend")
      : !backend.capabilities.includes("text_prompts")
        ? "Select SAM 3 to search by description" : "";
    search.disabled = searching || accepting || Boolean(reason);
    text.disabled = accepting || Boolean(reason);
    search.textContent = searching
      ? `Searching… ${Math.floor((performance.now() - searchStartedAt) / 1000)}s`
      : "Find in this frame";
    search.setAttribute("aria-busy", String(searching));
    search.title = searching
      ? "SAM 3 is processing the request. The first search also loads the model and video."
      : reason;
    if (reason && !response && !searching && !accepting) status.textContent = reason;
  }

  function invalidate() {
    generation += 1;
    response = null;
    results.hidden = true;
    preview(null);
    status.textContent = "";
    updateCapabilities();
  }

  async function showProposal() {
    const proposal = response?.proposals[Number(select.value)];
    label.value = proposal?.suggested_label_name.slice(0, 200) || "";
    await preview(proposal || null);
  }

  async function showResults() {
    select.replaceChildren();
    response.proposals.forEach((proposal, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `Instance ${index + 1} — score ${proposal.score.toFixed(2)}`;
      select.appendChild(option);
    });
    results.hidden = response.proposals.length === 0;
    status.textContent = response.proposals.length
      ? `${response.proposals.length} match(es). Review the mask and confirm its class.`
      : "No matching objects. Try another description.";
    await showProposal();
  }

  async function runSearch() {
    if (search.disabled) return;
    const concept = text.value.trim();
    if (!concept) { status.textContent = "Enter an object description first."; return; }
    invalidate();
    const token = generation;
    const context = getContext();
    searchStartedAt = performance.now();
    searching = true;
    updateCapabilities();
    status.textContent = "Searching this frame… The first search may take a while to load SAM 3 and prepare the video.";
    reportStatus(status.textContent);
    const progressTimer = setInterval(updateCapabilities, 1000);
    try {
      const found = await request(context, concept);
      if (token !== generation) return;
      response = found;
      await showResults();
      if (token === generation) reportStatus(status.textContent);
    } catch (error) {
      if (token === generation) {
        status.textContent = error.detail || error.message;
        reportStatus(status.textContent);
      }
    } finally {
      clearInterval(progressTimer);
      searching = false;
      updateCapabilities();
    }
  }

  search.addEventListener("click", runSearch);
  text.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); runSearch(); }
  });
  select.addEventListener("change", () => showProposal().catch((error) => {
    status.textContent = error.message;
  }));
  dismiss.addEventListener("click", invalidate);
  acceptButton.addEventListener("click", async () => {
    if (accepting || !response) return;
    const name = label.value.trim();
    if (!name) { status.textContent = "Confirm a class label first."; label.focus(); return; }
    const token = generation;
    const proposal = response.proposals[Number(select.value)];
    const acceptedResponse = response;
    accepting = true;
    for (const control of [acceptButton, dismiss, select, label]) control.disabled = true;
    updateCapabilities();
    try {
      await accept(proposal, name, acceptedResponse);
      if (token !== generation) return;
      response.proposals = response.proposals.filter((entry) => entry !== proposal);
      await showResults();
      if (token !== generation) return;
      if (!response.proposals.length) status.textContent = "All matching instances accepted.";
    } catch (error) {
      if (token === generation) status.textContent = error.detail || error.message;
    } finally {
      accepting = false;
      for (const control of [acceptButton, dismiss, select, label]) control.disabled = false;
      updateCapabilities();
    }
  });
  return { invalidate, updateCapabilities, isAccepting: () => accepting };
}
