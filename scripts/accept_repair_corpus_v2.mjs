import fs from 'node:fs';import crypto from 'node:crypto';import {execFileSync} from 'node:child_process';
const ROOT='D:/TaoVm';
const v=process.argv[2]||'20260909_v2';
const base=ROOT+'/build/repair_export_'+v+'/train';
const binFile=base+'.bin';
const receipt=ROOT+'/build/repair_export_'+v+'/train.bin.accepted';
if(fs.existsSync(receipt))throw Error('already accepted; do not repeat');
if(!fs.existsSync(binFile))throw Error('train.bin missing');
if(!fs.existsSync(binFile+'.manifest.tsv'))throw Error('manifest missing');
const metaBytes=fs.readFileSync(binFile+'.manifest.tsv');
const meta=new Map(metaBytes.toString('utf8').trim().split(/\r?\n/).map(l=>{const p=l.indexOf('\t');return [l.slice(0,p),l.slice(p+1)];}));
if(meta.get('status')!=='complete')throw Error('not committed');
for(const [file,key]of [[ROOT+'/build/formal_tokenizer.bbp','tokenizer_sha256'],[ROOT+'/build/bpe_pilot_validation.bin','validation_sha256'],[ROOT+'/build/bpe_pilot_test.bin','test_sha256']]){
  const h=crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  if(h!==meta.get(key))throw Error('frozen dependency changed '+file);
}
const composition=execFileSync('D:/nodejs/node.exe',[ROOT+'/summarize_repair_corpus.mjs',base],{encoding:'utf8'});
const result=JSON.parse(composition);
const docs=Number(meta.get('accepted')),supTok=Number(meta.get('supervised'));
if(docs<=2015||supTok<=398141)throw Error('filter reduced below baseline');
const validation=execFileSync(ROOT+'/build/validate_repair_corpus.exe',[binFile],{encoding:'utf8'});
if(!validation.startsWith('PASS ')||!validation.includes('SHA256='+meta.get('bin_sha256')))throw Error('native validation');
fs.writeFileSync(ROOT+'/build/repair_corpus_'+v+'_composition.json',composition,{flag:'wx'});
fs.writeFileSync(ROOT+'/build/repair_corpus_'+v+'_validation.log',validation,{flag:'wx'});
const hash=crypto.createHash('sha256').update(metaBytes).digest('hex');
fs.writeFileSync(receipt+'.partial','TAO_REPAIR_CORPUS_ACCEPT_V1\ndataset_sha256 '+meta.get('bin_sha256')+'\ntokenizer_sha256 '+meta.get('tokenizer_sha256')+'\nmanifest_sha256 '+hash+'\n',{flag:'wx'});
fs.renameSync(receipt+'.partial',receipt);
console.log(JSON.stringify({accepted:receipt,docs:result.docs,supervised:result.supervised,validation:validation.slice(0,400)},null,2));
