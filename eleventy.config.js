module.exports = function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/img");
  eleventyConfig.addPassthroughCopy("src/fonts");
  eleventyConfig.addPassthroughCopy("src/files");
  // Serve the KaTeX CSS/fonts from the same katex the markdown plugin renders with;
  // a version mismatch breaks layout (class names changed in 0.18).
  const path = require("path");
  const katexDist = path.relative(
    __dirname,
    path.join(
      path.dirname(
        require.resolve("katex/package.json", {
          paths: [path.dirname(require.resolve("@vscode/markdown-it-katex"))],
        })
      ),
      "dist"
    )
  );
  eleventyConfig.addPassthroughCopy({
    [`./${katexDist}/katex.min.css`]: "css/katex.min.css",
  });
  eleventyConfig.addPassthroughCopy({
    [`./${katexDist}/fonts`]: "css/fonts",
  });

  const md = require("markdown-it")({ html: true });
  md.use(require("@vscode/markdown-it-katex").default);
  eleventyConfig.setLibrary("md", md);

  eleventyConfig.addFilter("date", (value, format) => {
    const d = value === "now" ? new Date() : new Date(value);
    const month = d.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
    if (format === "year") return String(d.getUTCFullYear());
    return `${month} ${d.getUTCFullYear()}`;
  });

  // Newest first, and drop anything still marked draft.
  const live = (collection, tag) =>
    collection
      .getFilteredByTag(tag)
      .filter((post) => !post.data.draft)
      .reverse();

  eleventyConfig.addCollection("gadgets", (c) => live(c, "gadgets"));
  eleventyConfig.addCollection("quotes", (c) => live(c, "quotes"));
  eleventyConfig.addCollection("notes", (c) => live(c, "notes"));
  eleventyConfig.addCollection("writing", (c) => live(c, "writing"));

  // Everything across all three sections, newest first — used by the feed.
  eleventyConfig.addCollection("everything", (c) =>
    ["gadgets", "quotes", "notes", "writing"]
      .flatMap((tag) => live(c, tag))
      .sort((a, b) => b.date - a.date)
  );

  eleventyConfig.addFilter("rfc3339", (value) =>
    (value === "now" ? new Date() : new Date(value)).toISOString()
  );

  // The masthead lamp lights only when something was posted recently —
  // it reports state rather than decorating.
  const FRESH_DAYS = 30;
  eleventyConfig.addFilter("isFresh", (date) => {
    if (!date) return false;
    const age = (Date.now() - new Date(date).getTime()) / 86400000;
    return age >= 0 && age <= FRESH_DAYS;
  });

  // Newest first; `date` is YYYY-MM so string order is date order.
  eleventyConfig.addFilter("byYear", (photos, year) =>
    (year ? photos.filter((p) => p.year === year) : [...photos]).sort((a, b) =>
      b.date.localeCompare(a.date)
    )
  );

  eleventyConfig.addFilter("take", (array, n) => (array || []).slice(0, n));

  // Human label for the section a post lives in.
  const SECTION_TITLES = {
    gadgets: "Gadgets & Gizmos",
    kitchen: "Kitchen & Quotes",
    essays: "Essays",
    writing: "Writing & Recipes",
  };
  eleventyConfig.addFilter("sectionTitle", (key) => SECTION_TITLES[key] || key);

  return {
    // Root by default (repo named <user>.github.io). If the site lives at
    // <user>.github.io/<repo>/ instead, set a PATH_PREFIX of "/<repo>/".
    pathPrefix: process.env.PATH_PREFIX || "/",
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      data: "_data",
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
  };
};
