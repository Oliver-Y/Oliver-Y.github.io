const film = require("./film.json");

module.exports = [...new Set(film.map((p) => p.year))].sort().reverse();
