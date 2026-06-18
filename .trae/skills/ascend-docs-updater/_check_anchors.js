const fs = require('fs');
const path = require('path');

function docId(text) {
  let slug = text.toLowerCase();
  slug = slug.replace(/\./g, '-');
  slug = slug.replace(/[^a-z0-9\s-]/g, '');
  slug = slug.replace(/\s+/g, '-');
  slug = slug.replace(/-+/g, '-');
  slug = slug.replace(/^-|-$/g, '');
  return slug;
}

const dir = 'c:/Users/xujianzhao/Desktop/sglang/docs_new/docs/hardware-platforms/ascend-npus/best_practice';
const files = fs.readdirSync(dir).filter(f => f.endsWith('.mdx'));
files.sort();

let totalOk = 0, totalBroken = 0;
const dupes = [];

files.forEach(f => {
  const content = fs.readFileSync(path.join(dir, f), 'utf8');
  const headings = {};
  const anchors = [];

  content.split('\n').forEach((line, i) => {
    const h = line.replace(/\r$/, '').match(/^### (.+)/);
    if (h) {
      const id = docId(h[1]);
      headings[id] = {text: h[1], line: i + 1};
    }
    const a = [...line.matchAll(/#([a-z0-9][a-z0-9-]*)(?=[\s\)\]])/g)];
    a.forEach(m => anchors.push({anchor: m[1], line: i + 1}));
  });

  // Check heading duplicates
  const seen = new Set();
  Object.values(headings).forEach(h => {
    if (seen.has(h.text)) return;
    seen.add(h.text);
  });

  let broken = 0;
  anchors.forEach(a => {
    if (!headings[a.anchor]) broken++;
  });

  if (broken > 0) {
    console.log('❌ ' + f + ': ' + broken + ' broken anchors');
    anchors.filter(a => !headings[a.anchor]).forEach(a => {
      console.log('   L' + a.line + ': #' + a.anchor);
    });
  } else {
    totalOk++;
  }
  totalBroken += broken;
});

console.log('\n✅ ' + totalOk + ' files clean, ' + totalBroken + ' total broken anchors');
