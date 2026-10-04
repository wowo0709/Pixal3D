const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

const origin = process.env.VIEWER_URL || 'http://127.0.0.1:8765';
const output = process.env.VIEWER_TEST_OUTPUT || path.resolve(__dirname, '../../temp/result-viewer-browser');

async function orbits(page) {
  return page.locator('model-viewer').evaluateAll(elements => elements.map(element => {
    const orbit = element.getCameraOrbit();
    const target = element.getCameraTarget();
    return { theta: orbit.theta, phi: orbit.phi, radius: orbit.radius,
      target: [target.x, target.y, target.z], fov: element.getFieldOfView() };
  }));
}

async function drag(page, index) {
  const box = await page.locator('model-viewer').nth(index).boundingBox();
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 95, y + 28, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(800);
}

function close(a, b, tolerance = 0.025) {
  return Math.abs(a - b) < tolerance;
}

async function main() {
  await fs.mkdir(output, { recursive: true });
  const response = await fetch(origin);
  assert.equal(response.status, 200, 'The result viewer page must be served');
  const catalog = await (await fetch(origin + '/api/cases')).json();
  assert(catalog.cases.length > 0, 'Register at least one completed case');
  const firstCase = catalog.cases[0];
  const resultCount = firstCase.results.length;
  const inputCounts = firstCase.results.map(result => result.inputs.length);
  const browser = await chromium.launch({ headless: true,
    executablePath: process.env.VIEWER_BROWSER_EXECUTABLE, args: [
    '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-sandbox',
  ] });
  try {
    const context = await browser.newContext({ viewport: { width: 1600, height: 1100 }, acceptDownloads: true });
    const external = [];
    await context.route('**/*', route => {
      if (route.request().url().startsWith(origin + '/')) return route.continue();
      external.push(route.request().url());
      return route.abort();
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(origin);
    await page.locator('h1').waitFor({ timeout: 5000 });
    assert.match(await page.locator('h1').innerText(), /Multi-view comparison/);
    if (process.argv.includes('--smoke')) return;
    await page.waitForFunction(count => document.querySelectorAll('model-viewer').length === count &&
      [...document.querySelectorAll('model-viewer')].every(element => element.loaded), resultCount, { timeout: 120000 });
    await page.waitForTimeout(1500);
    assert.equal(await page.locator('.result-card[data-state="ready"]').count(), resultCount);
    assert.deepEqual(await page.locator('.input-strip').evaluateAll(strips => strips.map(strip => strip.querySelectorAll('button').length)), inputCounts);
    const materialCounts = await page.locator('model-viewer').evaluateAll(elements => elements.map(element =>
      element.model.materials.filter(material => material.pbrMetallicRoughness.baseColorTexture.texture !== null).length));
    assert(materialCounts.every(count => count > 0), 'Each GLB must display its actual base-color texture');
    await page.screenshot({ path: path.join(output, 'viewer-desktop.png'), fullPage: true });

    const start = await orbits(page);
    await drag(page, 0);
    const synced = await orbits(page);
    assert(Math.abs(synced[0].theta - start[0].theta) > 0.1, 'Dragging must orbit the first model');
    assert(synced.every(orbit => close(orbit.theta, synced[0].theta) && close(orbit.phi, synced[0].phi)), 'Orbits must synchronize');
    const box = await page.locator('model-viewer').first().boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -180);
    await page.waitForTimeout(900);
    const zoomed = await orbits(page);
    assert(Math.abs(zoomed[0].radius - synced[0].radius) > 0.01, 'Wheel must zoom');
    assert(zoomed.every(orbit => close(orbit.radius, zoomed[0].radius)), 'Zoom must synchronize');

    let independentOrbit = false;
    if (resultCount > 1) {
      await page.locator('#sync-camera').uncheck();
      const beforeIndependent = await orbits(page);
      await drag(page, 1);
      const independent = await orbits(page);
      assert(Math.abs(independent[1].theta - beforeIndependent[1].theta) > 0.1);
      assert(independent.every((orbit, index) => index === 1 || close(orbit.theta, beforeIndependent[index].theta, 0.001)));
      await page.locator('#sync-camera').check();
      await page.waitForTimeout(500);
      const aligned = await orbits(page);
      assert(aligned.every(orbit => close(orbit.theta, aligned[1].theta)), 'Re-enabling sync must align cameras');
      independentOrbit = true;
    }
    const resynced = await orbits(page);
    await page.keyboard.down('Shift');
    await drag(page, 0);
    await page.keyboard.up('Shift');
    const panned = await orbits(page);
    assert(panned[0].target.some((value, i) => Math.abs(value - resynced[0].target[i]) > 0.001), 'Shift-drag must pan');
    assert(panned.every(orbit => orbit.target.every((value, i) => close(value, panned[0].target[i], 0.005))), 'Pan targets must synchronize');
    await page.locator('#reset-camera').click();
    await page.waitForTimeout(500);
    const reset = await orbits(page);
    assert(reset.every(orbit => close(orbit.theta, start[0].theta) && close(orbit.radius, start[0].radius)));

    const inputResult = inputCounts.indexOf(Math.max(...inputCounts));
    const inputIndex = inputCounts[inputResult] - 1;
    await page.locator('.input-strip').nth(inputResult).locator('button').nth(inputIndex).click();
    await page.locator('#lightbox[open]').waitFor();
    assert.equal(await page.locator('#lightbox-image').getAttribute('src'), firstCase.results[inputResult].inputs[inputIndex].url);
    await page.keyboard.press('Escape');
    await page.locator('[data-mode="normal"]').click();
    assert.equal(await page.locator('.render-pane:not([hidden])').count(), resultCount);
    assert((await page.locator('.render-pane img').evaluateAll(images => images.map(image => image.src))).every(src => src.endsWith('/normal.png')));
    await page.locator('.render-pane button').first().click();
    await page.locator('#lightbox[open]').waitFor();
    await page.locator('#lightbox-close').click();
    await page.locator('[data-mode="base_color"]').click();
    assert((await page.locator('.render-pane img').evaluateAll(images => images.map(image => image.src))).every(src => src.endsWith('/base_color.png')));
    await page.locator('[data-mode="3d"]').click();

    const downloadPromise = page.waitForEvent('download');
    await page.locator('.glb-download').first().click();
    const download = await downloadPromise;
    const downloaded = path.join(output, download.suggestedFilename());
    await download.saveAs(downloaded);
    assert.equal((await fs.readFile(downloaded)).subarray(0, 4).toString(), 'glTF');
    await page.locator('details summary').first().click();
    assert.match(await page.locator('details').first().innerText(), /SS64/);
    await page.locator('details summary').first().click();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(900);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile layout must not overflow horizontally');
    const paintedPixels = [];
    for (let index = 0; index < resultCount; index++) {
      const viewer = page.locator('model-viewer').nth(index);
      await viewer.scrollIntoViewIfNeeded();
      await page.waitForTimeout(600);
      paintedPixels.push(await viewer.evaluate(async element => {
        // model-viewer switches between shared WebGL and per-model 2D canvases.
        // Its public snapshot API reads the canvas currently being displayed.
        const image = await createImageBitmap(await element.toBlob());
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        canvas.getContext('2d').drawImage(image, 0, 0);
        image.close();
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        let count = 0;
        for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) count++;
        return count;
      }));
      await viewer.screenshot({ path: path.join(output, `viewer-mobile-model-${index + 1}.png`) });
    }
    assert(paintedPixels.every(count => count > 1000), 'Every model must render when scrolled into view on mobile');
    await page.evaluate(() => scrollTo(0, 0));
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(output, 'viewer-mobile.png'), fullPage: true });

    const failedPage = await context.newPage();
    await failedPage.route('**' + firstCase.results[0].files.model, route => route.abort());
    await failedPage.goto(origin);
    await failedPage.locator('.result-card[data-state="error"]').waitFor({ timeout: 30000 });
    assert.match(await failedPage.locator('.result-card[data-state="error"] .load-status').innerText(), /failed/i);
    assert.equal(external.length, 0, 'No external CDN requests should be needed');
    assert.deepEqual(errors, []);
    // Real selection changes must load the corresponding GLBs and keep
    // the original nonzero anchor IDs visible after every switch.
    const checkedCases = [];
    await page.setViewportSize({ width: 1600, height: 1100 });
    for (const item of catalog.cases) {
      await page.locator('#case-select').selectOption(item.id);
      await page.waitForFunction(count => document.querySelectorAll('model-viewer').length === count &&
        [...document.querySelectorAll('model-viewer')].every(element => element.loaded), item.results.length, { timeout: 120000 });
      await page.waitForTimeout(250);
      const first = item.results[0];
      assert((await page.locator('#common-settings').innerText()).includes(`Anchor view ${first.anchor_view}`));
      const visibleInputs = await page.locator('.input-strip').evaluateAll(strips => strips.map(strip =>
        [...strip.querySelectorAll('img')].map(image => image.getAttribute('src'))));
      assert.deepEqual(visibleInputs, item.results.map(result => result.inputs.map(input => input.url)));
      const textured = await page.locator('model-viewer').evaluateAll(elements => elements.map(element =>
        element.model.materials.some(material => material.pbrMetallicRoughness.baseColorTexture.texture !== null)));
      assert(textured.every(Boolean), `Missing textures after selecting ${item.id}`);
      for (let index = 0; index < item.results.length; index++) {
        await page.locator('details summary').nth(index).click();
        assert((await page.locator('details').nth(index).innerText()).includes(`Anchor: input view ${first.anchor_view}`));
        await page.locator('details summary').nth(index).click();
      }
      checkedCases.push({ id: item.id, anchor: first.anchor_view, models: item.results.length, textured });
    }
    assert.deepEqual(errors, []);
    assert.equal(external.length, 0);
    const report = { passed: true, models: resultCount, textured_models: materialCounts.length,
      input_counts: inputCounts, orbit_sync: true, zoom_sync: true, independent_orbit: independentOrbit,
      pan_sync: true, reset: true, image_enlargement: true, render_tabs: true, glb_download: true,
      mobile_layout: true, mobile_rendered_pixels: paintedPixels,
      model_error_visible: true, cases: checkedCases, external_requests: external, page_errors: errors,
      renderer: 'Chromium software WebGL / SwiftShader', completed_utc: new Date().toISOString() };
    await fs.writeFile(path.join(output, 'browser_verification.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
