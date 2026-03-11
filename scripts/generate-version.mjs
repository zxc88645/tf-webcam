import { writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

async function inferCommit() {
  const sha =
    process.env.GITHUB_SHA ||
    process.env.COMMIT_SHA ||
    process.env.VERCEL_GIT_COMMIT_SHA;
  if (sha) return sha.slice(0, 7);
  try {
    const { stdout } = await execFileAsync("git", [
      "rev-parse",
      "--short",
      "HEAD",
    ]);
    const v = String(stdout || "").trim();
    return v || "dev";
  } catch {
    return "dev";
  }
}

const builtAt = new Date().toISOString();
const commit = await inferCommit();

const version = {
  commit,
  builtAt,
};

await writeFile(
  new URL("../public/version.json", import.meta.url),
  JSON.stringify(version, null, 2) + "\n",
  "utf8",
);
