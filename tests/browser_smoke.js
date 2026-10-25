const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/LAPTOP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const pageErrors = [];
  page.on('pageerror', e => pageErrors.push(e.message));
  await page.goto('http://127.0.0.1:8765/', { waitUntil: 'networkidle' });
  await page.addStyleTag({ content: '*,*::before,*::after{animation:none!important;transition:none!important}' });
  await page.getByRole('heading', { name: /Từ tín hiệu sớm/ }).waitFor();
  await page.screenshot({ path: 'D:/Bao cao/docs/screenshots/overview.png', fullPage: true });
  await page.getByRole('button', { name: /Dự báo nguy cơ/ }).click();
  await page.getByRole('button', { name: /Chạy CART/ }).click();
  await page.locator('#prediction-result').waitFor({ state: 'visible' });
  assert.match(await page.locator('#prediction-result').innerText(), /Nút lá/);
  await page.screenshot({ path: 'D:/Bao cao/docs/screenshots/prediction.png', fullPage: true });
  await page.locator('#f-clicks_day_0_28').fill('999999999');
  await page.getByRole('button', { name: /Chạy CART/ }).click();
  await page.locator('#form-error').getByText(/observed training range/).waitFor();
  const invalidInput = await page.locator('#form-error').innerText();
  await page.locator('#f-clicks_day_0_28').fill(String((await page.locator('#f-clicks_day_0_28').getAttribute('max'))));
  await page.getByRole('button', { name: /Đánh giá mô hình/ }).click();
  await page.locator('#confusion').waitFor();
  assert.match(await page.locator('#eval-summary').innerText(), /Recall/);
  await page.screenshot({ path: 'D:/Bao cao/docs/screenshots/evaluation.png', fullPage: true });
  await page.getByRole('button', { name: /Dữ liệu & giới hạn/ }).click();
  assert.match(await page.locator('#feature-chips').innerText(), /VLE/);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: 'D:/Bao cao/docs/screenshots/dataset-mobile.png', fullPage: true });
  assert.deepEqual(pageErrors, []);
  console.log(JSON.stringify({ pages: 4, prediction: 'passed', decision_path: 'passed', invalid_input: invalidInput, metrics: 'passed', dataset: 'passed', mobile_viewport: '390x844', page_errors: pageErrors, screenshots: ['docs/screenshots/overview.png', 'docs/screenshots/prediction.png', 'docs/screenshots/evaluation.png', 'docs/screenshots/dataset-mobile.png'] }, null, 2));
  await browser.close();
})().catch(e => { console.error(e); process.exitCode = 1; });


