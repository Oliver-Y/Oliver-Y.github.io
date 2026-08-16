module.exports = function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/img");

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
  eleventyConfig.addCollection("kitchen", (c) => live(c, "kitchen"));
  eleventyConfig.addCollection("essays", (c) => live(c, "essays"));

  // Everything across all three sections, newest first — used by the feed.
  eleventyConfig.addCollection("everything", (c) =>
    ["gadgets", "kitchen", "essays"]
      .flatMap((tag) => live(c, tag))
      .sort((a, b) => b.date - a.date)
  );

  eleventyConfig.addFilter("rfc3339", (value) =>
    (value === "now" ? new Date() : new Date(value)).toISOString()
  );

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
