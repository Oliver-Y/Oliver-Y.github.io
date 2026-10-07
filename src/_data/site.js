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
  url: "https://oliver-y.github.io",
  version: version(),
  sections: [
    { key: "notes", label: "notes", title: "Technical Notes" },
    { key: "gadgets", label: "gadgets", title: "Gadgets & Gizmos" },
    { key: "quotes", label: "quotes", title: "Quotes" },
    { key: "writing", label: "writing", title: "Writing & Recipes" },
    { key: "film", label: "film", title: "Film" },
  ],
  links: {
    github: "https://github.com/Oliver-Y",
    email: "mailto:olivermax700@gmail.com",
    linkedin: "https://www.linkedin.com/in/oliver-ye-50018a16a/",
  },
};
