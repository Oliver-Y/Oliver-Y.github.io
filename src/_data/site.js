const { execSync } = require("child_process");

// `ver 001` in the faceplate is the git commit count, so it bumps itself
// on every deploy. Falls back to 000 outside a git checkout (e.g. a fresh
// clone on a build server that fetched without history).
function version() {
  try {
    const count = execSync("git rev-list --count HEAD", {
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString()
      .trim();
    return String(count).padStart(3, "0");
  } catch {
    return "000";
  }
}

module.exports = {
  title: "Oliver Ye",
  url: "https://oliverye.dev",
  version: version(),
  sections: [
    { key: "gadgets", label: "gadgets", title: "Gadgets & Gizmos" },
    { key: "kitchen", label: "kitchen", title: "Kitchen & Quotes" },
    { key: "essays", label: "essays", title: "Essays" },
  ],
  links: {
    github: "https://github.com/oliverye",
    email: "mailto:hello@oliverye.dev",
  },
};
