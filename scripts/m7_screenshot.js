// M7: screenshot a page with Playwright (Chromium). Called by scripts/m7_app_screenshot.py.
//
//   node scripts/m7_screenshot.js <url> <out.png> <width> <height> [plotly_js_file]
//
// With plotly_js_file, requests for the plotly CDN script are answered from that local
// file, so a page that loads plotly from a CDN renders without network access. The script
// waits until every Plotly chart on the page has drawn before taking the picture.
const { chromium } = require("playwright");
const fs = require("fs");

(async () => {
  const [url, out, width, height, plotlyFile] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: Number(width), height: Number(height) },
    deviceScaleFactor: 1,
  });
  if (plotlyFile) {
    const body = fs.readFileSync(plotlyFile);
    await page.route(/cdn\.plot\.ly\/plotly-.*\.js/, (route) =>
      route.fulfill({ status: 200, contentType: "application/javascript", body }),
    );
  }
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(url, { waitUntil: "load", timeout: 120000 });
  await page.waitForSelector(".js-plotly-plot .main-svg", { timeout: 120000 });
  // Let late charts and fonts settle.
  await page.waitForTimeout(4000);
  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
  if (errors.length) {
    console.error("page errors:\n" + errors.join("\n"));
    process.exit(2);
  }
  console.log("saved " + out);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
