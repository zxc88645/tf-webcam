import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

function inferBaseFromGithubRepo(): string {
  const repo = process.env.GITHUB_REPOSITORY; // owner/name
  if (!repo) return "/";
  const name = repo.split("/")[1];
  if (!name) return "/";
  return `/${name}/`;
}

export default defineConfig(() => {
  return {
    base: inferBaseFromGithubRepo(),
    plugins: [react()],
    build: {
      outDir: "dist",
      emptyOutDir: true,
    },
  };
});
