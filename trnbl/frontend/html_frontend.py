import json

HTML_FRONTEND: str = json.loads(
	'<!doctypehtml><title>trnbl dashboard</title><script src=https://cdn.plot.ly/plotly-latest.min.js></script><script src=https://unpkg.com/interactjs/dist/interact.min.js></script><script src=https://cdn.jsdelivr.net/npm/ag-grid-community/dist/ag-grid-community.min.js></script><script src=https://unpkg.com/feather-icons></script><style>*,svg{font-family:Roboto-mono,monospace}.draggable:hover{z-index:999;border:1px solid #00f;transition-delay:.5s}.draggable{transition:z-index .5s}.draggable:not(:hover){transition-delay:.5s}.ag-icon{font-family:agGridAlpine!important}option{font-size:2em!important}.rootContainer{height:100%}.plotContainer,.runsManifestBox{background-color:#f0f0f0;border:1px solid #000;border-radius:10px;padding:3px}.plotSettings{margin-left:2px}.headerButton{cursor:pointer;color:#000;text-align:center;background-color:#d8d8d8;border:1px solid #3c3c3c;border-radius:5px;height:35px;padding:0 10px;font-size:16px;line-height:30px;transition:background-color .1s,border-color .1s,color .1s;display:inline-block}.headerButton:hover,.headerButton:focus{color:#000;background-color:#797979;border-color:#213244}.headerButton svg{vertical-align:middle}hr{border:none;border-top:1px solid #ccc;margin-top:10px}.axis-toggle-container{margin-top:10px}.switch{width:3em;height:1.5em;display:inline-block;position:relative}.switch input{opacity:0;width:0;height:0}.slider{cursor:pointer;background-color:#ccc;-webkit-transition:all .4s;transition:all .4s;position:absolute;inset:0}.slider:before{content:"";background-color:#fff;width:1em;height:1em;-webkit-transition:all .4s;transition:all .4s;position:absolute;bottom:4px;left:4px}input:checked+.slider{background-color:#2196f3}input:focus+.slider{box-shadow:0 0 1px #2196f3}input:checked+.slider:before{-webkit-transform:translate(1.5em);-ms-transform:translate(1.5em);transform:translate(1.5em)}.slider.round{border-radius:34px}.slider.round:before{border-radius:50%}</style><body><header id=mainHeader><h1 id=projectH1>trnbl Dashboard</h1><button class=headerButton id=refreshButton><i data-feather=refresh-cw></i><i data-feather=bar-chart-2></i> Refresh Data</button><button class=headerButton id=saveLayoutButton><i data-feather=save></i><i data-feather=layout></i> Save Current Layout</button><button class=headerButton id=resetLayoutButton><i data-feather=refresh-ccw></i><i data-feather=layout></i> Reset Layout</button><button class=headerButton id=downloadLayoutButton><i data-feather=download></i><i data-feather=layout></i> Download Current Layout</button><div class=headerButton><input id=gridSnapCheckbox type=checkbox><label for=gridSnapCheckbox><i data-feather=grid></i> Grid Snap</label></div><button class=headerButton id=resetColumnStateButton><i data-feather=refresh-ccw></i><i data-feather=columns></i> Reset Column State</button><hr></header><div class=rootContainer id=rootContainerDiv><div id=invisHbar style=width:100em;height:100em></div></div><script>document.addEventListener(`DOMContentLoaded`,init)</script><script>var createColumnDefs=(a=>{let i=0,h=1,k=`open`,j=null,l=`Config`,g=Date;var b=[{headerName:`view/hide`,field:`selected`,width:30,checkboxSelection:!0,headerCheckboxSelection:!0,headerCheckboxSelectionFilteredOnly:!0}];const c={filter:`agDateColumnFilter`,filterParams:{comparator:((a,b)=>{const c=new g(b);const d=new g(a);if(c<d){return -h}else if(c>d){return h};return i}),browserDatePicker:!1,inRangeInclusive:!0}};const d=[{headerName:`Name`,children:[{field:`id.syllabic`,headerName:`Syllabic ID`,columnGroupShow:j},{field:`id.run`,headerName:`Full Run ID`,columnGroupShow:k}],marryChildren:!0},{headerName:`Timing`,children:[{field:`timing.start`,headerName:`Start`,columnGroupShow:j,...c},{field:`timing.final`,headerName:`End`,columnGroupShow:k,...c},{field:`timing.duration`,headerName:`Duration (ms)`,columnGroupShow:k}],marryChildren:!0},{headerName:`Final Metrics`,children:[]},{headerName:l,children:[{field:`config`,headerName:l,cellRenderer:fancyCellRenderer}],marryChildren:!0}];const e=new Set();a.forEach(a=>{Object.keys(a.final_metrics).forEach(a=>e.add(a))});var f=i;e.forEach(a=>{d[2].children.push({field:`final_metrics.${a}`,headerName:a,columnGroupShow:f===h?j:k});f+=h});b=b.concat(d);return b});var headerButtons=(async()=>{let c=`click`;const a=document.getElementById(`projectH1`);a.textContent=DATA_MANAGER.projectName+ ` trnbl Dashboard`;const b=document.getElementById(`gridSnapCheckbox`);b.checked=LAYOUT_MANAGER.do_snap;b.addEventListener(`change`,(()=>{LAYOUT_MANAGER.updateSnap(b.checked)}));document.getElementById(`saveLayoutButton`).addEventListener(c,async()=>{await LAYOUT_MANAGER.saveLayout()});document.getElementById(`downloadLayoutButton`).addEventListener(c,async()=>{const a=LAYOUT_MANAGER.get_local_storage_key();const b=JSON.stringify(LAYOUT_MANAGER,null,`\\\\t`);const c=new Blob([b],{type:`application/json`});const d=URL.createObjectURL(c);const e=document.createElement(`a`);e.href=d;e.download=a+ `.json`;document.body.appendChild(e);e.click();e.remove();URL.revokeObjectURL(d)});document.getElementById(`resetLayoutButton`).addEventListener(c,async()=>{const a=LAYOUT_MANAGER.get_local_storage_key();IO_MANAGER.deleteJsonLocal(a);location.reload();createNotification(`Layout resetting...`,`info`)});document.getElementById(`refreshButton`).addEventListener(c,async()=>{createNotification(`Not yet implemented!`,`error`)});document.getElementById(`resetColumnStateButton`).addEventListener(c,async()=>{GRID_API.resetColumnState()})});var createRunsManifestTable=(a=>{let e=10;const b=document.createElement(`div`);b.id=`runsManifest`;b.classList.add(`runsManifestBox`,`ag-theme-alpine`);document.getElementById(`rootContainerDiv`).appendChild(b);const c=LAYOUT_MANAGER.layout[b.id];if(c){b.style.cssText=`position: absolute; width: ${c.width}px; height: ${c.height}px; left: ${c.x}px; top: ${c.y}px; margin-bottom: 20px; ${DEFAULT_STYLE}`};LAYOUT_MANAGER.makeElementDraggable(b);const d={columnDefs:createColumnDefs(a),rowData:a,pagination:!0,enableCellTextSelection:!0,enableBrowserTooltips:!0,rowSelection:`multiple`,pagination:!0,paginationPageSize:e,paginationPageSizeSelector:[1,2,5,e,25,50,100,500,1000],defaultColDef:{resizable:!0,filter:!0,floatingFilter:!0,menuTabs:[]},domLayout:`autoHeight`,onFirstDataRendered:(a=>{adjustTableHeight(b)}),onPaginationChanged:(a=>{adjustTableHeight(b)}),initialState:LAYOUT_MANAGER.grid_state};GRID_API=agGrid.createGrid(b,d)});var init=(async()=>{await DATA_MANAGER.loadManifest();LAYOUT_MANAGER=new LayoutManager(DATA_MANAGER.projectName);LAYOUT_MANAGER.loadLayout(do_update=!1);await headerButtons();await PLOT_MANAGER.createAllPlots();await DATA_MANAGER.loadRuns();await createRunsManifestTable(DATA_MANAGER.summaryManifest);await PLOT_MANAGER.populateAllPlots();try{feather.replace();const a=document.querySelectorAll(`i[data-feather]`);a.forEach(a=>{a.innerHTML=``})}catch(a){console.error(`Feather icons not found`);displayNotification(`Feather icons not found, keeping text fallback`,`error`)}console.log(`init complete`);feather.replace()});var adjustTableHeight=(a=>{const b=a.querySelector(`.ag-center-cols-viewport`).offsetHeight;const c=a.querySelector(`.ag-header`).offsetHeight;const d=a.querySelector(`.ag-paging-panel`).offsetHeight;const e=b+ c+ d+ 50;a.style.minHeight=`${e}px`});var createNotification=((a,b=e)=>{let e=`info`;const c=`[${b}]: ${a}`;switch(b){case e:console.log(c);break;case `warning`:console.warn(c);break;case `error`:console.error(c);break;default:console.log(c)}const d=document.createElement(`div`);d.textContent=a;d.style.cssText=`position: absolute; top: 0; right: 0; padding: 10px; border-radius: 10px; background-color: ${NOTIFICATION_COLORS[b]};`;document.body.appendChild(d);setTimeout(()=>{d.remove()},3000)});var isISODate=(a=>{const b=/^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(.\\d+)?(Z|[+-]\\d{2}:\\d{2})?$/;return b.test(a)});var fancyCellRenderer=(a=>{let d=null;var b;if(a.value===undefined){return c}else{b=a.value};var c=document.createElement(`div`);c.title=b;c.textContent=b;c.style.cursor=`pointer`;if(b!==d){if(typeof b===`object`){b=JSON.stringify(b,d,4)};if(b.length>50){c.title=b;c.innerHTML=feather.icons[`mouse-pointer`].toSvg()+ feather.icons[`copy`].toSvg();c.style.cssText=`font-size: 20px; display: flex; justify-content: center; align-items: center; background-color: #f4f4f4; border: 1px solid #d4d4d4; border-radius: 5px; height: 30px; width: 60px;`}};c.onclick=(()=>{navigator.clipboard.writeText(b).then((()=>{console.log(`Successfully copied to clipboard`)})).catch((a=>{console.error(`Could not copy text to clipboard: `,a)}))});c.oncontextmenu=(()=>{const c=window.open(``,`_blank`);c.document.write(`<pre>`+ b+ `</pre>`);c.document.title=a.node.data[`id.run`]+ ` : `+ a.colDef.headerName;c.document.close()});return c});let LAYOUT_MANAGER=null;let PLOTLY_LAYOUTS={};let GRID_API=null;const DEFAULT_XUNITS=`samples`;const SETTINGS_WIDTH_PX=100;const PLOT_BOTTOM_MARGIN_PX=5;const SNAP_INTERVAL_DEFAULT=50;const NOTIFICATION_COLORS={\'info\':`lightgreen`,\'warning\':`lightyellow`,\'error\':`lightcoral`};const DEFAULT_STYLE={border:`1px solid black`,backgroundColor:`#f0f0f0`,borderRadius:`10px`,padding:`3px`};const PLOTLY_LAYOUT_MARGIN={l:40,r:30,b:40,t:50,pad:0};class IOManager{async fetchJson(a){try{const b=await fetch(a);if(!b.ok)throw new Error(`HTTP error! status: ${b.status}`);return await b.json()}catch(a){console.error(`Fetch JSON error:`,a);return null}}async fetchJsonLines(a){let b=1,c=JSON.parse;try{const d=await fetch(a);if(!d.ok)throw new Error(`HTTP error! status: ${d.status}`);const e=await d.text();const f=e.trim().split(`\\\\n`);const g=f.slice(0,-b).map(a=>c(a));try{const a=c(f[f.length- b]);g.push(a)}catch(b){console.error(`Invalid JSON in the last line of ${a}: `,b)}return g}catch(a){console.error(`Fetch JSON Lines error:`,a);return null}}async saveJsonLocal(a,b){const c=JSON.stringify(b);localStorage.setItem(a,btoa(c))}async readJsonLocal(a){const b=localStorage.getItem(a);if(b){const a=atob(b);return JSON.parse(a)}else{return null}}async deleteJsonLocal(a){localStorage.removeItem(a)}}const IO_MANAGER=new IOManager();class RunData{constructor(a){let b=null;this.path=a;this.config=b;this.meta=b;this.metrics=b;this.logs=b;this.artifacts=b}async loadData(){this.config=await IO_MANAGER.fetchJson(`${this.path}/config.json`);this.meta=await IO_MANAGER.fetchJson(`${this.path}/meta.json`);this.metrics=await IO_MANAGER.fetchJsonLines(`${this.path}/metrics.jsonl`);this.logs=await IO_MANAGER.fetchJsonLines(`${this.path}/log.jsonl`);this.artifacts=await IO_MANAGER.fetchJsonLines(`${this.path}/artifacts.jsonl`)}pairMetrics(a,b){let e=isNaN;const c=[];const d=[];if(this.metrics){this.metrics.forEach(f=>{const g=f[a];const h=f[b];if(!e(g)&&!e(h)){c.push(g);d.push(h)}})};return [c,d]}static smoothData(a,b=null,c=g){let h=0,j=1,k=2,g=`SMA`,i=Math;if(a.some(isNaN)){createNotification(`Data contains NaN values`,`warning`)};if(!b){return a};const d=[];switch(c){case g:for(let c=h;c<a.length;c++){const e=i.max(h,c- b+ j);const f=a.slice(e,c+ j);const g=f.reduce((a,b)=>a+ b,h);const k=g/f.length;d.push(k)};break;case `EMA`:let e=a[h];const f=k/(b+ j);for(let b=h;b<a.length;b++){e=f*a[b]+ (j- f)*(b>h?e:a[b]);d.push(e)};break;case `Gaussian`:for(let c=h;c<a.length;c++){let e=h;let f=h;for(let d=-b;d<=b;d++){if(c+ d>=h&&c+ d<a.length){const g=i.exp(-(d*d)/(k*b*b));e+=a[c+ d]*g;f+=g}};const g=e/f;d.push(g)};break;default:console.error(`Invalid smoothing method.`);return []}return d}}class DataManager{constructor(){let a=null;this.manifest=a;this.allRuns={};this.metricNames=new Set();this.projectName=a;this.summaryManifest=a}async loadManifest(){let b=`error`;this.manifest=await IO_MANAGER.fetchJsonLines(`runs.jsonl`);if(!this.manifest){createNotification(`Failed to load manifest`,b)};const a=new Set();for(const b of this.manifest){a.add(b.project);b.metric_names.forEach(a=>{this.metricNames.add(a)})};if(a.size===1){this.projectName=a.values().next().value}else{createNotification(`Project names are not consistent across runs: ${a}`,b)}}async loadRuns(){if(!this.manifest){this.loadManifest()};for(const a of this.manifest){const b=new RunData(`runs/${a.run_id}`);await b.loadData();this.allRuns[a.run_id]=b};this.updateSummaryManifest()}updateSummaryManifest(){let b=0,c=1,e=Date,a=Object,d=undefined;try{if(a.keys(this.allRuns).length===b){throw `No runs found`}}catch(a){createNotification(`Could not find any runs to update summary manifest: ${a}`,`error`)}this.summaryManifest=a.values(this.allRuns).map(a=>{const f=a.logs.length>b?a.logs[a.logs.length- c].timestamp:null;let g={};for(let e=a.metrics.length- c;e>=b;e--){this.metricNames.forEach(b=>{if(a.metrics[e][b]!==d&&g[b]===d){g[b]=a.metrics[e][b]}})};return {id:{syllabic:a.meta.syllabic_id,run:a.meta.run_id},timing:{start:a.meta.run_init_timestamp,final:f,duration:new e(f)- new e(a.meta.run_init_timestamp)},final_metrics:g,config:a.config}})}}const DATA_MANAGER=new DataManager();class LayoutManager{constructor(a,b=300,c=0.4){this.projectName=a;this.layout={};this.do_snap=!0;this.snapInterval=SNAP_INTERVAL_DEFAULT;this.plot_configs={};this.grid_state=null;this.init_y=this.round_to_snap_interval(130),this.default_plot_cont_height=this.round_to_snap_interval(b);const d=window.innerWidth;this.default_plot_cont_width=this.round_to_snap_interval(d*c);this.table_width=this.round_to_snap_interval(d- (this.default_plot_cont_width+ this.snapInterval))}round_to_snap_interval(a){return Math.ceil(a/this.snapInterval)*this.snapInterval}get_default_layout(a,b=!0){let f=0;const c=Array.from(a);var d={};const e=this.round_to_snap_interval(this.default_plot_cont_height*1.1);for(let a=f;a<c.length;a++){const b=c[a];d[`plotContainer-${b}`]={x:f,y:this.init_y+ a*e,height:this.default_plot_cont_height,width:this.default_plot_cont_width}};d[`runsManifest`]={x:this.default_plot_cont_width+ SNAP_INTERVAL_DEFAULT,y:this.init_y,height:800,width:this.table_width};if(b){this.layout=d};return d}async getDefaultPlotConfig(){let a=`linear`;return {size:{width:this.default_plot_cont_width- SETTINGS_WIDTH_PX,height:this.default_plot_cont_height},axisScales:{x:a,y:a},smoothing_method:`SMA`,smoothing_span:null,xUnits:DEFAULT_XUNITS}}async getPlotConfig(a){if(!(a in this.plot_configs)){this.plot_configs[a]=await this.getDefaultPlotConfig()};return this.plot_configs[a]}makeElementDraggable(a){let d=`draggable`;const b=a.id;let c=this.getInitialPosition(a);if(!a.classList.contains(d)){a.classList.add(d)};this.initializeDragInteraction(a,c);this.initializeResizeInteraction(a,c);this.updateElementLayout(a,c.x,c.y,!0)}getInitialPosition(a){let d=0,c=parseFloat;const b=a.id;if(this.layout[b]){return {x:this.layout[b].x,y:this.layout[b].y}}else{return {x:c(a.getAttribute(`data-x`))||d,y:c(a.getAttribute(`data-y`))||d}}}initializeDragInteraction(a,b){let d=1,c=0;interact(a).draggable({ignoreFrom:`.draglayer, .ag-header, .ag-center-cols-container, .no-drag, .legend, .bg, .scrollbox`,modifiers:[interact.modifiers.snap({targets:[interact.snappers.grid({x:this.snapInterval,y:this.snapInterval})],range:Infinity,relativePoints:[{x:c,y:c}]}),interact.modifiers.restrict({restriction:`parent`,elementRect:{top:c,left:c,bottom:d,right:d},endOnly:!0})],inertia:!0}).on(`dragmove`,a=>{b.x+=a.dx;b.y+=a.dy;this.updateElementLayout(a.target,b.x,b.y,!0)})}initializeResizeInteraction(a,b){interact(a).resizable({edges:{left:!0,right:!0,bottom:!0,top:!0},modifiers:[interact.modifiers.snapSize({targets:[interact.snappers.grid({width:this.snapInterval,height:this.snapInterval})],range:Infinity}),interact.modifiers.restrictSize({min:{width:250,height:150}})],inertia:!0}).on(`resizemove`,a=>{const {width:c,height:d}=a.rect;b.x+=a.deltaRect.left;b.y+=a.deltaRect.top;const e=a.target;this.updateElementLayout(e,b.x,b.y,!1,c,d);const f=e.classList.contains(`plotContainer`);if(f){const b=e.querySelector(`.plotSettings`);const c=e.querySelector(`.plotDiv`);var g=a.rect.width- SETTINGS_WIDTH_PX;b.style.width=g;c.style.width=`${g}px`;c.style.height=`${a.rect.height}px`;const d=c.id;Plotly.relayout(d,{width:g,height:a.rect.height- PLOT_BOTTOM_MARGIN_PX});const f=d.split(`-`)[1];this.plot_configs[f].size={width:g,height:a.rect.height};PLOTLY_LAYOUTS[f].width=g;PLOTLY_LAYOUTS[f].height=a.rect.height- PLOT_BOTTOM_MARGIN_PX}})}updateElementLayout(a,b,c,d=!0,e=g,f=g){let g=null;if(d){a.style.left=`${b}px`;a.style.top=`${c}px`};if(e&&f){a.style.width=`${e}px`;a.style.height=`${f}px`}else{e=a.offsetWidth;f=a.offsetHeight};this.layout[a.id]={x:b,y:c,width:e,height:f}}updateAllLayouts(){for(const a in this.layout){const b=this.layout[a];const c=document.getElementById(a);const d=this.getInitialPosition(c);this.updateElementLayout(c,b.x,b.y,!0,b.width,b.height)}}get_local_storage_key(){return `${this.projectName}_layout`}async saveLayout(){let c=JSON.stringify;this.updateGridState();const a=this.get_local_storage_key();IO_MANAGER.saveJsonLocal(a,this);const b=await IO_MANAGER.readJsonLocal(a);if(b&&c(b)==c(this)){console.log(`Layout saved:`,b);createNotification(`Layout saved`,`info`)}else{console.error(`Layout not saved:`,this,b);createNotification(`Layout not saved`,`error`)}}async loadLayout(a=!0){const b=this.get_local_storage_key();const c=await IO_MANAGER.readJsonLocal(b);if(c){this.projectName=c.projectName;this.layout=c.layout;this.do_snap=c.do_snap;this.snapInterval=c.snapInterval;this.plot_configs=c.plot_configs;this.grid_state=c.grid_state}else{this.layout=this.get_default_layout(DATA_MANAGER.metricNames)};console.log(`Layout loaded:`,this);if(a){this.updateAllLayouts()}}async updateSnap(a=!0,b=SNAP_INTERVAL_DEFAULT){this.do_snap=a;if(!a){b=1};this.snapInterval=b;console.log(`Snap settings updated:`,this.do_snap,this.snapInterval);for(const a in this.layout){const b=document.getElementById(a);let c=this.getInitialPosition(b);this.initializeDragInteraction(b,c);this.initializeResizeInteraction(b,c)}}updateGridState(){this.grid_state=GRID_API.getState()}}class PlotManager{constructor(){this.plots={}}async createPlot(a){let i=JSON;const b=`plotContainer-${a}`;const c=`plot-${a}`;const d=`plotSettings-${a}`;const e=await LAYOUT_MANAGER.getPlotConfig(a);const f=LAYOUT_MANAGER.layout[b];const g=`\n\t\t\t<div\n\t\t\t\tid="${b}"\n\t\t\t\tclass="plotContainer" \n\t\t\t\tstyle="margin-bottom: 10px; display: flex; flex-direction: row; position: absolute; width: ${f.width}px; height: ${f.height}px; left: ${f.x}px; top: ${f.y}px; ${DEFAULT_STYLE}"\n\t\t\t>\n\t\t\t\t<div \n\t\t\t\t\tid="${c}"\n\t\t\t\t\tclass="plotDiv" \n\t\t\t\t\tstyle="width: ${f.width- SETTINGS_WIDTH_PX}px; height: ${f.height- PLOT_BOTTOM_MARGIN_PX}px;"\n\t\t\t\t></div>\n\t\t\t\t<div \n\t\t\t\t\tid="${d}"\n\t\t\t\t\tclass="plotSettings" \n\t\t\t\t\tstyle="width: ${SETTINGS_WIDTH_PX}; flex-shrink: 0; flex-grow: 0;"\n\t\t\t\t></div>\n\t\t\t</div>\n\t\t`;document.getElementById(`rootContainerDiv`).insertAdjacentHTML(`beforeend`,g);this.plots[a]={plotID:c,containerID:b,settingsID:d};const h={title:`${a} over ${e.xUnits}`,autosize:!0,xaxis:{title:e.xUnits,type:e.axisScales.x,showgrid:!0},yaxis:{title:a,type:e.axisScales.y,showgrid:!0},margin:PLOTLY_LAYOUT_MARGIN,width:f.width- SETTINGS_WIDTH_PX,height:f.height- PLOT_BOTTOM_MARGIN_PX};PLOTLY_LAYOUTS[a]=h;Plotly.newPlot(c,[],i.parse(i.stringify(h)));this.createAxisToggles(a);this.createSmoothingInput(a);LAYOUT_MANAGER.makeElementDraggable(document.getElementById(b))}async createAllPlots(a=50,b=150){const c=DATA_MANAGER.metricNames;let d=0;c.forEach(a=>{d+=1;console.log(`creating plot ${d} for ${a}`);this.createPlot(a)})}async updatePlot(a){let e=JSON;const b=this.plots[a];const c=await LAYOUT_MANAGER.getPlotConfig(a);if(!b){console.error(`Plot for metric ${a} not found.`);return};var d=[];for(const b in DATA_MANAGER.allRuns){const e=DATA_MANAGER.allRuns[b];const f=e.meta.syllabic_id;const [g,h]=e.pairMetrics(DEFAULT_XUNITS,a);let i=RunData.smoothData(h,c.smoothing_span,c.smoothing_method);const j={x:g,y:i,mode:`lines`,line:c.smoothing_span?{shape:`spline`}:{},name:f};d.push(j)};PLOTLY_LAYOUTS[a].xaxis.type=c.axisScales.x;PLOTLY_LAYOUTS[a].yaxis.type=c.axisScales.y;PLOTLY_LAYOUTS[a].uirevision=a;Plotly.react(b.plotID,d,e.parse(e.stringify(PLOTLY_LAYOUTS[a])))}async populateAllPlots(){for(const a of DATA_MANAGER.metricNames){this.updatePlot(a)}}updateAxisScale(a,b,c){const d=this.plots[a];if(!d){console.error(`Plot for metric ${a} not found.`);return};const e=LAYOUT_MANAGER.plot_configs[a];e.axisScales[b]=c;Plotly.relayout(d.plotID,{[`${b}axis`]:{type:c},uirevision:a})}createAxisToggles(a){let e=`log`;const b=this.plots[a].settingsID;const c=this.plots[a].plotID;const d=document.getElementById(b);[`x`,`y`].forEach(b=>{const f=`${c}-${b}Toggle`;const g=`\n\t\t\t\t<div class="axis-toggle-container" style="display: block;">\n\t\t\t\t\t<label for="${f}" style="display: block;">${b.toUpperCase()} Scale</label>\n\t\t\t\t\t<div style="display: flex; align-items: center;">\n\t\t\t\t\t\t<i data-feather="arrow-up-right">lin</i>\n\t\t\t\t\t\t<label class="switch">\n\t\t\t\t\t\t\t<input type="checkbox" id="${f}">\n\t\t\t\t\t\t\t<span class="slider round"></span>\n\t\t\t\t\t\t</label>\n\t\t\t\t\t\t<i data-feather="corner-right-up">log</i>\n\t\t\t\t\t</div>\n\t\t\t\t</div>\n\t\t\t`;const h=document.createElement(`div`);h.innerHTML=g.trim();d.appendChild(h);const i=document.getElementById(f);i.checked=LAYOUT_MANAGER.plot_configs[a].axisScales[b]===e;i.onchange=()=>{const c=i.checked?e:`linear`;this.updateAxisScale(a,b,c)}})}async createSmoothingInput(a){const b=this.plots[a].plotID;const c=this.plots[a].settingsID;const d=[`SMA`,`EMA`,`Gaussian`];const e=`\n\t\t\t<div class="smoothing-input-container no-drag" style="display: block; margin-top: 10px; border: 1px solid grey; border-radius: 3px;">\n\t\t\t\t<label for="smoothingInput-${b}" style="font-weight: bold;">Smooth:</label><br>\n\t\t\t\t<label for="smoothingMethodSelect-${b}">Method</label><br>\n\t\t\t\t<select class="no-drag" id="smoothingMethodSelect-${b}" style="width: 6em;">\n\t\t\t\t\t${d.map(a=>`<option value="${a}">${a}</option>`).join(``)}\n\t\t\t\t</select><br>\n\t\t\t\t<label for="smoothingInput-${b}">Span</label><br>\n\t\t\t\t<input class="no-drag" type="number" min="0" max="1000" value="0" id="smoothingInput-${b}" style="width: 4.2em;">\n\t\t\t</div>\n\t\t`;const f=document.createElement(`div`);f.innerHTML=e.trim();const g=document.getElementById(c);g.appendChild(f);const h=document.getElementById(`smoothingInput-${b}`);const i=document.getElementById(`smoothingMethodSelect-${b}`);h.value=LAYOUT_MANAGER.plot_configs[a].smoothing_span;i.value=LAYOUT_MANAGER.plot_configs[a].smoothing_method;h.onchange=()=>{LAYOUT_MANAGER.plot_configs[a].smoothing_span=parseInt(h.value);this.updatePlot(a)};i.onchange=()=>{LAYOUT_MANAGER.plot_configs[a].smoothing_method=i.value;this.updatePlot(a)}}}let PLOT_MANAGER=new PlotManager()</script>'
)