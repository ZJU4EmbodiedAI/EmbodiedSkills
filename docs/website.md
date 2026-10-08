# Project website

The GitHub Pages site lives in `site/` and uses plain HTML, CSS and JavaScript. It has no build step, package dependencies, analytics or externally hosted runtime assets.

Preview it locally from the repository root:

```bash
python -m http.server 8765 --directory site
```

Open `http://localhost:8765`. A local HTTP server is required for JavaScript module loading.

The interactive examples in `site/evidence.js` contain published candidate scores, outcomes and references to recorded observations. The two MetaWorld players replay saved trajectory frames; the moving paths and latent grids in the explanatory diagrams are illustrations. The benchmark figures and values follow the current paper. Update the values in both `site/index.html` and `site/app.js` when revising results.

The site is published from the root of the `gh-pages` branch. After committing changes under `site/`, publish with:

```bash
git subtree split --prefix site -b site-publish
git push origin site-publish:gh-pages
git branch -D site-publish
```

GitHub Pages should use **Deploy from a branch → gh-pages → / (root)**. Its address is <https://zju4embodiedai.github.io/EmbodiedSkills/>.

## Visual material

All paper figures and robot observations are from the EmbodiedSkills project. Source images were converted to WebP; original experiment records and private filesystem paths are not part of the website. Images shown after CLIPort candidates are immediate post-action observations, while the displayed outcome labels describe the complete recorded branch. The website never presents an animated schematic as a generated future observation.

Manrope is self-hosted under the SIL Open Font License. Its license is included in `site/assets/fonts/OFL.txt`; the source is the [Google Fonts Manrope directory](https://github.com/google/fonts/tree/main/ofl/manrope). The interface, diagrams and motion design were implemented specifically for this project.
