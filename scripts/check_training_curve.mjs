import fs from 'node:fs';
const text=fs.readFileSync(new URL('./build/loss_curve.csv',import.meta.url),'utf8');
const lines=text.split('\n');lines.pop();
if(lines.shift()!=='timestamp,kind,step,nll,lr')throw Error('CSV header mismatch');
const last={train:0,validation:0},counts={train:0,validation:0},latest={};
for(const line of lines){if(!line.trim())continue;const [timestamp,kind,stepText,nllText,lrText,...extra]=line.trim().split(',');const step=Number(stepText),nll=Number(nllText),lr=Number(lrText);if(extra.length||!Object.hasOwn(last,kind)||!Number.isInteger(step)||step<=last[kind]||!Number.isFinite(nll)||nll<0||!Number.isFinite(lr)||lr<=0||!Number.isFinite(Date.parse(timestamp)))throw Error('Invalid or nonmonotonic curve row: '+line);last[kind]=step;counts[kind]++;latest[kind]={step,nll,lr,timestamp};}
console.log(JSON.stringify({valid:true,counts,latest,note:'Read-only snapshot; ignores unfinished trailing line; does not prove trainer liveness.'},null,2));
