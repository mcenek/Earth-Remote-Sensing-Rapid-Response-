'use strict';
fetch('/api/experiments').then(r=>{if(!r.ok)throw Error('Local catalog unavailable');return r.json();}).then(d=>{window.GTM_DATA=d;const s=document.createElement('script');s.src='GTM_Model0.js';document.body.append(s);}).catch(e=>{document.getElementById('message').textContent='Start the local viewer launcher. '+e.message;});
