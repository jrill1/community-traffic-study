const { test, expect } = require('@playwright/test');
const path = require('path');

const URL = `file://${path.resolve(__dirname, '../dashboard.html')}`;

// Helpers
async function getStatCards(page) {
  const cards = page.locator('#stats .stat');
  const events = await cards.nth(0).locator('.v').textContent();
  const pct    = await cards.nth(1).locator('.v').textContent();
  return { events: events.trim(), pct: pct.trim() };
}

async function getStatCardsB(page) {
  const cards = page.locator('#statsB .stat');
  const events = await cards.nth(0).locator('.v').textContent();
  const pct    = await cards.nth(1).locator('.v').textContent();
  return { events: events.trim(), pct: pct.trim() };
}

async function selectDate(page, isoDate) {
  await page.evaluate((date) => {
    const picker = document.getElementById('datePicker');
    picker.value = date;
    picker.dispatchEvent(new Event('change'));
  }, isoDate);
}

test.describe('Dashboard stats', () => {
  test('Single date — Jun 30: 326 events, 31% over limit', async ({ page }) => {
    await page.goto(URL);
    await selectDate(page, '2026-06-30');
    const { events, pct } = await getStatCards(page);
    expect(events).toBe('326');
    expect(pct).toBe('31%');
  });

  test('Compare date mode — Jun 29 vs Jun 30: left 353/26%, right 326/31%', async ({ page }) => {
    await page.goto(URL);
    await selectDate(page, '2026-06-30');
    await page.locator('#compareBtn').click();
    await page.locator('#dateLabel').click();
    // Left = Jun 29, Right = Jun 30
    const left  = await getStatCards(page);
    const right = await getStatCardsB(page);
    expect(left.events).toBe('353');
    expect(left.pct).toBe('26%');
    expect(right.events).toBe('326');
    expect(right.pct).toBe('31%');
  });

  test('Span mode — Jun 29–30: 679 events, 28% over limit', async ({ page }) => {
    await page.goto(URL);
    await selectDate(page, '2026-06-30');
    await page.locator('#spanToggle').click();
    const { events, pct } = await getStatCards(page);
    expect(events).toBe('679');
    expect(pct).toBe('28%');
  });

  test('Compare spans — Jun 27–28 vs Jun 29–30: left 571/24%, right 679/28%', async ({ page }) => {
    await page.goto(URL);
    await selectDate(page, '2026-06-30');
    await page.locator('#spanToggle').click();
    await page.locator('#compareBtn').click();
    await page.locator('#dateLabel').click();
    const left  = await getStatCards(page);
    const right = await getStatCardsB(page);
    expect(left.events).toBe('571');
    expect(left.pct).toBe('24%');
    expect(right.events).toBe('679');
    expect(right.pct).toBe('28%');
  });
});
