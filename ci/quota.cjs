// Reservation artifacts deliberately overcount: failures/cancellations still
// consume their entire allocation. Never delete them to force another run.
module.exports = async function ({ github, context, core }) {
  const fs = require('fs');
  const budget = Number(process.env.CI_DAILY_BUDGET || '3');
  if (!Number.isInteger(budget) || budget < 0 || budget > 20) {
    throw new Error('GEMINI_CI_DAILY_BUDGET must be an integer between 0 and 20.');
  }
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  const artifacts = await github.paginate(github.rest.actions.listArtifactsForRepo, {
    ...context.repo, per_page: 100,
  });
  const reservations = artifacts.filter(a => /^gemini-reservation-\d+-\d+$/.test(a.name)
    && Date.parse(a.created_at) >= cutoff);
  const reserved = reservations.length * 3;
  const allowed = reserved + 3 <= budget;
  core.setOutput('allowed', String(allowed));
  core.info(`CI requests reserved in last 24h: ${reserved}; budget: ${budget}; requested: 3.`);
  if (allowed) {
    fs.mkdirSync('ci_output', { recursive: true });
    fs.writeFileSync('ci_output/quota-reservation.json', JSON.stringify({
      run_id: context.runId, attempt: process.env.GITHUB_RUN_ATTEMPT,
      requested_gemini_calls: 3, created_at: new Date().toISOString(),
    }, null, 2));
  }
};
