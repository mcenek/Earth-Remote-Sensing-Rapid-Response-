'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const code=fs.readFileSync('dist/GTM_Model0.js','utf8');
const context=vm.createContext({window:{},document:{},console,URLSearchParams,location:{search:""}});
vm.runInContext(code.slice(0,code.search(/^try\s*\{/m))+'\nthis.api={validateGrid,validateExperiment,allowedAsset,hasPred};',context);
const a=context.api;
assert.throws(()=>a.allowedAsset('../outside.json'));
assert.throws(()=>a.allowedAsset('https://untrusted.example/image.png'));
assert.throws(()=>a.validateGrid({width:-2,height:-2}));
assert.throws(()=>a.validateGrid({width:2,height:2,probability:[0,0,0,2]}));
assert.throws(()=>a.validateGrid({width:2,height:2,valid:[1,1,1,7]}));
assert.throws(()=>a.validateGrid({width:2,height:2,truth:[1,0,null]}));
assert.doesNotThrow(()=>a.validateGrid({width:2,height:2,probability:[0,1,null,.2],truth:[0,1,null,0],valid:[1,1,0,1]}));
assert.equal(a.hasPred({grid:{width:1,height:1,truth:[1]}}),false);
assert.equal(a.hasPred({grid:{width:1,height:1,probability:[.5]}}),true);
if (process.argv.includes('--source-only')) { console.log('PASS: source-only import contract, path validation, and no-data handling; omitted asset bundles were not checked'); process.exit(0); }
const data=JSON.parse(fs.readFileSync('dist/GTM_Model0_data.js','utf8').replace(/^window.GTM_DATA\s*=\s*/,'').replace(/;\s*$/,''));
let scenes=0,grids=0;
for(const e of data.experiments){a.validateExperiment(e);if(e.reportUrl)assert(fs.existsSync(path.join('dist',e.reportUrl)),e.reportUrl);for(const s of e.scenes){scenes++;for(const k of ['rgb','emit','truthImage','predictionImage','disagreementImage','grid'])if(typeof s[k]==='string'&&!s[k].startsWith('data:')){assert(fs.existsSync(path.join('dist',s[k])),s[k]);if(k==='grid'){a.validateGrid(JSON.parse(fs.readFileSync(path.join('dist',s[k]),'utf8')));grids++;}}}}
for(const m of fs.readFileSync('dist/index.html','utf8').matchAll(/(?:src|href)="([^"#]+)"/g))assert(fs.existsSync(path.join('dist',m[1])),m[1]);
console.log(`PASS: import contract, no-data handling, ${data.experiments.length} experiments, ${scenes} scenes, ${grids} grids, linked static assets`);
