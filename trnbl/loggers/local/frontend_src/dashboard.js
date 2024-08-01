/*
 ######   #######  ##    ##  ######  ########  ######
##    ## ##     ## ###   ## ##    ##    ##    ##    ##
##       ##     ## ####  ## ##          ##    ##
##       ##     ## ## ## ##  ######     ##     ######
##       ##     ## ##  ####       ##    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##    ##
 ######   #######  ##    ##  ######     ##     ######
*/

// APIs and global variables
let LAYOUT_MANAGER = null;
let PLOTLY_LAYOUTS = {};
let GRID_API = null;
// const xUnitsKeys = ['samples', 'batches', "timestamp"]; // TODO: read dynamically, add epochs & runs
const DEFAULT_XUNITS = 'samples';

// settings

const LAYOUT_CONFIG = {
	"plot_cont_height": 300,
	"plotcont_frac": 0.4,
	"elements_initial_offset_y": 200,
	"minimum_dims": { width: 250, height: 150 },
	"table_init_height": 800,
	"plot_bottom_margin_px": 5,
	"snap_interval_default": 50,
	"settings_width_px": 100,
}

const NOTIFICATION_CONFIG = {
	colors: {
		'info': 'lightgreen',
		'warning': 'lightyellow',
		'error': 'lightcoral',
	},
	border_colors: {
		'info': 'green',
		'warning': 'orange',
		'error': 'red',
	},
	timeout: 5000,
}

const AUTO_REFRESH_CONFIG = {
	interval: -1,
	select_options: [
		{ value: -1, label: 'Off' },
		{ value: 1, label: '1' },
		{ value: 5, label: '5' },
		{ value: 10, label: '10' },
		{ value: 30, label: '30' },
		{ value: 60, label: '60' },
		{ value: 100, label: '100' },
		{ value: 1000, label: '1000' }
	],
}

const DEFAULT_AUTO_REFRESH = 10;

const DEFAULT_STYLE = {
	border: '1px solid black',
	backgroundColor: '#f0f0f0',
	borderRadius: '10px',
	padding: '3px',
}
const PLOTLY_LAYOUT_MARGIN = { l: 40, r: 30, b: 40, t: 50, pad: 0 };


/*
####  #######     ##     ## ##    ##  ######
 ##  ##     ##    ###   ### ###   ## ##    ##
 ##  ##     ##    #### #### ####  ## ##
 ##  ##     ##    ## ### ## ## ## ## ##   ####
 ##  ##     ##    ##     ## ##  #### ##    ##
 ##  ##     ##    ##     ## ##   ### ##    ##
####  #######     ##     ## ##    ##  ######
*/

class IOManager {
	constructor() {
		this.fileTimestamps = {};
	}

	async fetchWithNoCache(path) {
		const headers = new Headers({
			'Cache-Control': 'no-cache, no-store, must-revalidate',
			'Pragma': 'no-cache',
			'Expires': '0'
		});

		return fetch(path, {
			method: 'GET',
			headers: headers,
			credentials: 'same-origin'
		});
	}

	async fetchJson(path, force_no_cache = false) {
		try {
			let response = null;
			if (force_no_cache) {
				response = await this.fetchWithNoCache(path);
			}
			else {
				response = await fetch(path);
			}
			if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
			return await response.json();
		} catch (error) {
			createNotification(`Fetch JSON error: ${error}`, 'error', error);
			return null;
		}
	}

	async fetchJsonLines(path, force_no_cache = false, notify_last_line_error = false) {
		try {
			let response = null;
			if (force_no_cache) {
				response = await this.fetchWithNoCache(path);
			}
			else {
				response = await fetch(path);
			}
			if (!response.ok) {
				let response_status = response ? response.status : 'unknown';
				throw new Error(`HTTP error! status: ${response_status}`);
			}
			const text = await response.text();
			const lines = text.trim().split('\n');
			const validLines = lines.slice(0, -1).map(line => JSON.parse(line));

			// Try parsing the last line
			try {
				const lastLine = JSON.parse(lines[lines.length - 1]);
				validLines.push(lastLine);
			} catch (error) {
				if (notify_last_line_error) {
					createNotification(`Invalid JSON in the last line of ${path}: ${error}`, 'error', error);
				}
			}

			return validLines;
		} catch (error) {
			createNotification(`Fetch JSON Lines error: ${error}`, 'error', error);
			return null;
		}
	}

	async saveJsonLocal(name, data) {
		const data_json = JSON.stringify(data);
		localStorage.setItem(name, btoa(data_json));
	}

	async readJsonLocal(name) {
		const data_encoded = localStorage.getItem(name);
		if (data_encoded) {
			const data_json = atob(data_encoded);
			return JSON.parse(data_json);
		} else {
			return null;
		}
	}

	async deleteJsonLocal(name) {
		localStorage.removeItem(name);
	}

	async getFileModificationTime(path) {
		try {
			const response = await fetch(path + '?t=' + new Date().getTime(), { method: 'HEAD' });
			if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
			return new Date(response.headers.get('Last-Modified'));
		} catch (error) {
			createNotification(`Error fetching file modification time: ${error}`, 'error', error);
			return null;
		}
	}

	async fetchJsonIfModified(path) {
		const lastModified = await this.getFileModificationTime(path);
		if (!lastModified) return null;

		if (!this.fileTimestamps[path] || lastModified > this.fileTimestamps[path]) {
			const data = await this.fetchJson(path);
			if (data !== null) {
				this.fileTimestamps[path] = lastModified;
			}
			return data;
		}
		return null; // File not modified
	}

	async fetchJsonLinesIfModified(path) {
		const lastModified = await this.getFileModificationTime(path);
		if (!lastModified) return null;

		if (!this.fileTimestamps[path] || lastModified > this.fileTimestamps[path]) {
			const data = await this.fetchJsonLines(path);
			if (data !== null) {
				this.fileTimestamps[path] = lastModified;
			}
			return data;
		}
		return null; // File not modified
	}
}

const IO_MANAGER = new IOManager();


/*
########  ##     ## ##    ##    ########     ###    ########    ###
##     ## ##     ## ###   ##    ##     ##   ## ##      ##      ## ##
##     ## ##     ## ####  ##    ##     ##  ##   ##     ##     ##   ##
########  ##     ## ## ## ##    ##     ## ##     ##    ##    ##     ##
##   ##   ##     ## ##  ####    ##     ## #########    ##    #########
##    ##  ##     ## ##   ###    ##     ## ##     ##    ##    ##     ##
##     ##  #######  ##    ##    ########  ##     ##    ##    ##     ##
*/

class RunData {
	constructor(path) {
		this.path = path;
		this.config = null;
		this.meta = null;
		this.metrics = null;
		this.logs = null;
		this.artifacts = null;
	}

	async loadData() {
		this.config = await IO_MANAGER.fetchJson(`${this.path}/config.json`);
		this.meta = await IO_MANAGER.fetchJson(`${this.path}/meta.json`);
		this.metrics = await IO_MANAGER.fetchJsonLines(`${this.path}/metrics.jsonl`);
		this.logs = await IO_MANAGER.fetchJsonLines(`${this.path}/log.jsonl`);
		this.artifacts = await IO_MANAGER.fetchJsonLines(`${this.path}/artifacts.jsonl`);
	}

	pairMetrics(xKey, yKey) {
		const xVals = [];
		const yVals = [];
		if (this.metrics) {
			this.metrics.forEach(metric => {
				const xv = metric[xKey];
				const yv = metric[yKey];
				if (!isNaN(xv) && !isNaN(yv)) {
					xVals.push(xv);
					yVals.push(yv);
				}
			});
		}
		return [xVals, yVals];
	}

	static smoothData(data, span = null, method = 'SMA') {
		if (data.some(isNaN)) {
			createNotification('Data contains NaN values', 'warning');
		}

		if (!span) {
			return data;
		}

		const smoothed = [];
		switch (method) {
			case 'SMA':
				for (let i = 0; i < data.length; i++) {
					const start = Math.max(0, i - span + 1);
					const window = data.slice(start, i + 1);
					const sum = window.reduce((acc, val) => acc + val, 0);
					const avg = sum / window.length;
					smoothed.push(avg);
				}
				break;
			case 'EMA':
				let ema = data[0]; // Starting with the first data point
				const alpha = 2 / (span + 1);
				for (let i = 0; i < data.length; i++) {
					ema = alpha * data[i] + (1 - alpha) * (i > 0 ? ema : data[i]);
					smoothed.push(ema);
				}
				break;
			case 'Gaussian':
				// Gaussian smoothing requires calculating weights for each point in the window
				// We'll use a simplified Gaussian kernel for demonstration purposes
				for (let i = 0; i < data.length; i++) {
					let weightedSum = 0;
					let weightSum = 0;
					for (let j = -span; j <= span; j++) {
						if (i + j >= 0 && i + j < data.length) {
							// Calculate the Gaussian weight
							const weight = Math.exp(-(j * j) / (2 * span * span));
							weightedSum += data[i + j] * weight;
							weightSum += weight;
						}
					}
					const gaussianAverage = weightedSum / weightSum;
					smoothed.push(gaussianAverage);
				}
				break;
			default:
				createNotification(`Invalid smoothing method: ${method}`, 'error');
				return [];
		}

		return smoothed;
	}
}

/*
########     ###    ########    ###       ##     ## ##    ##  ######
##     ##   ## ##      ##      ## ##      ###   ### ###   ## ##    ##
##     ##  ##   ##     ##     ##   ##     #### #### ####  ## ##
##     ## ##     ##    ##    ##     ##    ## ### ## ## ## ## ##   ####
##     ## #########    ##    #########    ##     ## ##  #### ##    ##
##     ## ##     ##    ##    ##     ##    ##     ## ##   ### ##    ##
########  ##     ##    ##    ##     ##    ##     ## ##    ##  ######
*/

class DataManager {
	constructor() {
		this.manifest = null;
		this.allRuns = {};
		this.metricNames = new Set();
		this.projectName = null;
		this.summaryManifest = null;
		this.lastRefreshTime = null;
		this.updatedRuns = new Set();
	}

	async loadManifest() {
		// load data
		this.manifest = await IO_MANAGER.fetchJsonLines('runs.jsonl', true); // force_no_cache=true
		if (!this.manifest) {
			createNotification('Failed to load manifest', 'error');
		}

		// get project name, metric names
		const projectNames = new Set();
		for (const run of this.manifest) {
			projectNames.add(run.project);

			run.metric_names.forEach(metricName => {
				this.metricNames.add(metricName);
			});
		}
		// project names should match
		if (projectNames.size === 1) {
			this.projectName = projectNames.values().next().value;
		} else {
			createNotification(`Project names are not consistent across runs: ${projectNames}`, 'error');
		}
	}

	async loadRuns() {
		// load manifest if not already loaded
		if (!this.manifest) {
			this.loadManifest();
		}

		// load each run
		for (const run of this.manifest) {
			const runData = new RunData(`runs/${run.run_id}`);
			await runData.loadData();
			this.allRuns[run.run_id] = runData;
		}

		// update summary manifest (final metrics, timestamps, etc)
		this.updateSummaryManifest();
	}

	updateSummaryManifest() {
		// make sure there are runs
		try {
			if (Object.keys(this.allRuns).length === 0) {
				throw 'No runs found';
			}
		} catch (error) {
			createNotification(`Could not find any runs to update summary manifest: ${error}`, 'error');
		}

		// create the summary manifest dictionary
		this.summaryManifest = Object.values(this.allRuns).map(run => {
			// final timestamp, from logs
			const finalTimestamp = run.logs.length > 0 ? run.logs[run.logs.length - 1].timestamp : null;

			// final values for each metric
			let finalMetrics = {};
			// Iterate over the metrics array in reverse to find the last value for each metricName
			for (let i = run.metrics.length - 1; i >= 0; i--) {
				this.metricNames.forEach(metricName => {
					// Check if the metric name is present in the current metric and hasn't been added to lastValues yet
					if (run.metrics[i][metricName] !== undefined && finalMetrics[metricName] === undefined) {
						finalMetrics[metricName] = run.metrics[i][metricName];
					}
				});
			}

			// final state of the run
			const finalState_all = run.metrics[run.metrics.length - 1];
			let finalState_filtered = {};
			for (let key of ['samples', 'batches', 'epochs', 'latest_checkpoint']) {
				if (finalState_all) {
					finalState_filtered[key] = finalState_all[key];
				} else {
					finalState_filtered[key] = null;
				}				
			}

			// return the summary manifest object, for each run
			return {
				id: {
					syllabic: run.meta.syllabic_id,
					run: run.meta.run_id,
					group: run.meta.group,
				},
				timing: {
					start: run.meta.run_init_timestamp,
					final: finalTimestamp,
					duration: new Date(finalTimestamp) - new Date(run.meta.run_init_timestamp),
				},
				final_metrics: finalMetrics,
				final_state: finalState_filtered,
				config: run.config,
				meta: run.meta,
			};
		});
	}

	async refreshData(verbose = true) {

		if (verbose) {
			createNotification('Checking for data updates...', 'info');
		}

		let dataUpdated = false;
		this.updatedRuns.clear(); // Clear the set at the start of each refresh

		// Refresh manifest
		const newManifest = await IO_MANAGER.fetchJsonLines('runs.jsonl', true); // force_no_cache=true
		if (JSON.stringify(newManifest) !== JSON.stringify(this.manifest)) {
			this.manifest = newManifest;
			console.log(this.manifest);
			dataUpdated = true;
			createNotification("Manifest file updated", "info");
		}

		// Refresh run data
		for (const run of this.manifest) {
			const runPath = `runs/${run.run_id}`;
			const newConfig = await IO_MANAGER.fetchJsonIfModified(`${runPath}/config.json`);
			const newMeta = await IO_MANAGER.fetchJsonIfModified(`${runPath}/meta.json`);
			const newMetrics = await IO_MANAGER.fetchJsonLinesIfModified(`${runPath}/metrics.jsonl`);
			const newLogs = await IO_MANAGER.fetchJsonLinesIfModified(`${runPath}/log.jsonl`);
			const newArtifacts = await IO_MANAGER.fetchJsonLinesIfModified(`${runPath}/artifacts.jsonl`);

			if (newConfig || newMeta || newMetrics || newLogs || newArtifacts) {
				const runData = this.allRuns[run.run_id] || new RunData(runPath);
				if (newConfig) runData.config = newConfig;
				if (newMeta) runData.meta = newMeta;
				if (newMetrics) runData.metrics = newMetrics;
				if (newLogs) runData.logs = newLogs;
				if (newArtifacts) runData.artifacts = newArtifacts;
				this.allRuns[run.run_id] = runData;
				dataUpdated = true;
				this.updatedRuns.add(run.run_id);
				console.log(`Updated data for run ${run.run_id}`);
			}
		}

		if (dataUpdated) {
			// Update metric names
			this.metricNames.clear();
			for (const run of this.manifest) {
				run.metric_names.forEach(metricName => {
					this.metricNames.add(metricName);
				});
			}

			// Update summary manifest
			this.updateSummaryManifest();

			this.lastRefreshTime = new Date();

			const updatedRunsInfo = DATA_MANAGER.getUpdatedRunsInfo();

			// Update plots
			await PLOT_MANAGER.populateAllPlots();

			// Update table
			GRID_API.setGridOption('rowData', DATA_MANAGER.summaryManifest);
			GRID_API.refreshCells({ force: true });

			// Detailed notification
			if (updatedRunsInfo.count > 0) {
				createNotification(
					(
						`Data refreshed successfully. ${updatedRunsInfo.count} run(s) updated`
						// use ternary operator to add the list of updated runs if there are less than 3
						+ (updatedRunsInfo.count < 3 ? `: ${updatedRunsInfo.runs.join(', ')}` : '')
					),
					'info',
					updatedRunsInfo,
					verbose,
				);
			} else {
				createNotification(
					'Manifest updated, but no individual runs were changed',
					'info',
					updatedRunsInfo, verbose,
				);
			}

		} else {
			createNotification(
				'No new data updates found', 'info',
				null, verbose,
			);
		}

		return dataUpdated;
	}

	initAutoRefresh() {
		const autoRefreshSelect = document.getElementById('autoRefreshSelect');

		// Populate the select element with options
		AUTO_REFRESH_CONFIG.select_options.forEach(option => {
			const optionElement = document.createElement('option');
			optionElement.value = option.value;
			optionElement.textContent = option.label;
			if (option.value === AUTO_REFRESH_CONFIG.interval) {
				optionElement.selected = true;
			}
			autoRefreshSelect.appendChild(optionElement);
		});

		autoRefreshSelect.addEventListener('change', () => {
			this.setAutoRefresh(parseFloat(autoRefreshSelect.value));
		});

		// Initialize auto-refresh with default value
		this.setAutoRefresh(AUTO_REFRESH_CONFIG.interval);
	}

	setAutoRefresh(seconds) {
		// Clear existing interval if any
		if (this.autoRefreshInterval) {
			clearInterval(this.autoRefreshInterval);
		}

		// If seconds is greater than 0, set new interval
		if (seconds > 0) {
			this.autoRefreshInterval = setInterval(
				() => {
					this.refreshData(false).then(() => {
						// Blink the refresh button after each refresh
						blinkElement('autoRefreshButton');
					});
				},
				seconds * 1000,
			);
			createNotification(`Auto refresh set to ${seconds} seconds`, 'info');
		} else {
			createNotification('Auto refresh turned off', 'info');
		}
	}

	getUpdatedRunsInfo() {
		return {
			count: this.updatedRuns.size,
			runs: Array.from(this.updatedRuns)
		};
	}
}

const DATA_MANAGER = new DataManager();




/*
##          ###    ##    ##  #######  ##     ## ########
##         ## ##    ##  ##  ##     ## ##     ##    ##
##        ##   ##    ####   ##     ## ##     ##    ##
##       ##     ##    ##    ##     ## ##     ##    ##
##       #########    ##    ##     ## ##     ##    ##
##       ##     ##    ##    ##     ## ##     ##    ##
######## ##     ##    ##     #######   #######     ##
*/

class LayoutManager {
	constructor(projectName) {
		this.projectName = projectName;
		this.layout = {};
		this.do_snap = true;
		this.snapInterval = LAYOUT_CONFIG.snap_interval_default;
		this.plot_configs = {};
		this.grid_state = null;
		this.visibilityState = {};
		// default layout stuff
		this.init_y = this.round_to_snap_interval(LAYOUT_CONFIG.elements_initial_offset_y);
		this.default_plot_cont_height = this.round_to_snap_interval(LAYOUT_CONFIG.plot_cont_height);
		// calculate widths
		const window_width = window.innerWidth;
		this.default_plot_cont_width = this.round_to_snap_interval(window_width * LAYOUT_CONFIG.plotcont_frac);
		this.table_width = this.round_to_snap_interval(window_width - (this.default_plot_cont_width + this.snapInterval));
	}

	round_to_snap_interval(value) {
		return Math.ceil(value / this.snapInterval) * this.snapInterval;
	}

	get_default_layout(
		plot_names,
		update_to_default = true,
	) {
		// convert plot_names to list
		const plot_names_arr = Array.from(plot_names);

		// init layout
		var layout = {};
		const plot_y_step = this.round_to_snap_interval(this.default_plot_cont_height * 1.1)

		// plot containers
		for (let i = 0; i < plot_names_arr.length; i++) {
			const metricName = plot_names_arr[i];
			layout[`plotContainer-${metricName}`] = {
				x: 0,
				y: this.init_y + i * plot_y_step,
				height: this.default_plot_cont_height,
				width: this.default_plot_cont_width,
			};
		};

		// table
		layout['runsManifest'] = {
			x: this.default_plot_cont_width + LAYOUT_CONFIG.snap_interval_default,
			y: this.init_y,
			height: LAYOUT_CONFIG.table_init_height,
			width: this.table_width,
		};

		// write to global
		if (update_to_default) {
			this.layout = layout;
		}

		return layout;
	}

	async getDefaultPlotConfig() {
		return {
			size: { width: this.default_plot_cont_width - LAYOUT_CONFIG.settings_width_px, height: this.default_plot_cont_height },
			axisScales: { x: 'linear', y: 'linear' },
			smoothing_method: 'SMA',
			smoothing_span: null,
			xUnits: DEFAULT_XUNITS,
		};
	}

	async getPlotConfig(metricName) {
		if (!(metricName in this.plot_configs)) {
			this.plot_configs[metricName] = await this.getDefaultPlotConfig();
		}
		return this.plot_configs[metricName];
	}

	makeElementDraggable(element) {
		// get id and position
		const id = element.id;
		let position = this.getInitialPosition(element);

		// add .draggable class if its not there
		if (!element.classList.contains('draggable')) {
			element.classList.add('draggable');
		}

		// make draggable and resizable
		this.initializeDragInteraction(element, position);
		this.initializeResizeInteraction(element, position);

		// update layout
		this.updateElementLayout(element, position.x, position.y, true);
	}

	getInitialPosition(element) {
		const id = element.id;
		if (this.layout[id]) {
			return { x: this.layout[id].x, y: this.layout[id].y };
		} else {
			return {
				x: parseFloat(element.getAttribute('data-x')) || 0,
				y: parseFloat(element.getAttribute('data-y')) || 0,
			};
		}
	}

	initializeDragInteraction(element, position) {
		interact(element).draggable({
			ignoreFrom: '.draglayer, .ag-header, .ag-center-cols-container, .no-drag, .legend, .bg, .scrollbox',
			modifiers: [
				interact.modifiers.snap({
					targets: [interact.snappers.grid({ x: this.snapInterval, y: this.snapInterval })],
					range: Infinity,
					relativePoints: [{ x: 0, y: 0 }]
				}),
				interact.modifiers.restrict({
					restriction: 'parent',
					elementRect: { top: 0, left: 0, bottom: 1, right: 1 },
					endOnly: true
				})
			],
			inertia: true
		}).on('dragmove', (event) => {
			position.x += event.dx;
			position.y += event.dy;

			this.updateElementLayout(event.target, position.x, position.y, true);
		});
	}

	initializeResizeInteraction(element, position) {
		interact(element).resizable({
			edges: { left: true, right: true, bottom: true, top: true },
			modifiers: [
				interact.modifiers.snapSize({
					targets: [interact.snappers.grid({ width: this.snapInterval, height: this.snapInterval })],
					range: Infinity,
				}),
				interact.modifiers.restrictSize({
					min: LAYOUT_CONFIG.minimum_dims,
				}),
			],
			inertia: true
		}).on('resizemove', (event) => {
			const { width, height } = event.rect;
			position.x += event.deltaRect.left;
			position.y += event.deltaRect.top;

			const target = event.target;
			this.updateElementLayout(target, position.x, position.y, false, width, height);

			const isPlotContainer = target.classList.contains('plotContainer');
			if (isPlotContainer) {
				// Adjust sizes of plotDiv and plotSettings inside the container
				const plotSettings = target.querySelector('.plotSettings');
				const plotDiv = target.querySelector('.plotDiv');

				// Set plotSettings width and adjust plotDiv width
				var plotDivWidth = event.rect.width - LAYOUT_CONFIG.settings_width_px;
				plotSettings.style.width = plotDivWidth;

				// Update plotDiv and Plotly plot size
				plotDiv.style.width = `${plotDivWidth}px`;
				plotDiv.style.height = `${event.rect.height}px`;

				// Now, instruct Plotly to resize the plot
				const plotID = plotDiv.id;
				Plotly.relayout(plotID, {
					width: plotDivWidth, // New width for the plot
					height: event.rect.height - LAYOUT_CONFIG.plot_bottom_margin_px, // New height for the plot
				});

				// save in plot configs
				const metricName = plotID.split('-')[1];
				this.plot_configs[metricName].size = { width: plotDivWidth, height: event.rect.height };
				PLOTLY_LAYOUTS[metricName].width = plotDivWidth;
				PLOTLY_LAYOUTS[metricName].height = event.rect.height - LAYOUT_CONFIG.plot_bottom_margin_px;
			}
		});
	}

	updateElementLayout(element, x, y, updatePosition = true, width = null, height = null) {
		// update position if provided
		if (updatePosition) {
			// element.style.transform = `translate(${x}px, ${y}px)`;
			// element.setAttribute('data-x', x);
			// element.setAttribute('data-y', y);
			element.style.left = `${x}px`;
			element.style.top = `${y}px`;
		}

		// update width and height if provided
		if (width && height) {
			element.style.width = `${width}px`;
			element.style.height = `${height}px`;
		}
		else {
			width = element.offsetWidth;
			height = element.offsetHeight;
		}

		// store in layout
		this.layout[element.id] = {
			x: x,
			y: y,
			width: width,
			height: height,
		};
		// console.log('Updated layout for:', element.id, this.layout[element.id]);
	}

	updateAllLayouts() {
		for (const id in this.layout) {
			const new_layout = this.layout[id];
			const element = document.getElementById(id);
			// console.log('Updating layout for: ', id, new_layout);
			const position = this.getInitialPosition(element);
			this.updateElementLayout(element, new_layout.x, new_layout.y, true, new_layout.width, new_layout.height);
		}
	}

	get_local_storage_key() {
		return `${this.projectName}_layout`;
	}

	async saveLayout() {
		this.updateGridState();
		const layoutKey = this.get_local_storage_key();
		IO_MANAGER.saveJsonLocal(layoutKey, this);
		const layout_read = await IO_MANAGER.readJsonLocal(layoutKey);
		if (layout_read && (JSON.stringify(layout_read) == JSON.stringify(this))) {
			console.log('Layout saved:', layout_read);
			createNotification('Layout saved', 'info');
		} else {
			console.error('Layout not saved:', this, layout_read);
			createNotification('Layout not saved', 'error');
		}
	}

	async loadLayout(do_update = true) {
		const layoutKey = this.get_local_storage_key();
		const savedLayout = await IO_MANAGER.readJsonLocal(layoutKey);
		if (savedLayout) {
			this.projectName = savedLayout.projectName;
			this.layout = savedLayout.layout;
			this.do_snap = savedLayout.do_snap;
			this.snapInterval = savedLayout.snapInterval;
			this.plot_configs = savedLayout.plot_configs;
			this.grid_state = savedLayout.grid_state;
			this.visibilityState = savedLayout.visibilityState;
		} else {
			this.layout = this.get_default_layout(DATA_MANAGER.metricNames);
		}
		console.log('Layout loaded:', this);
		if (do_update) {
			this.updateAllLayouts();
		}
	}

	async updateSnap(do_snap = true, snapInterval = LAYOUT_CONFIG.snap_interval_default) {
		this.do_snap = do_snap;
		if (!do_snap) {
			snapInterval = 1;
		}
		this.snapInterval = snapInterval;

		console.log('Snap settings updated:', this.do_snap, this.snapInterval);

		for (const id in this.layout) {
			const element = document.getElementById(id);
			let position = this.getInitialPosition(element);

			this.initializeDragInteraction(element, position);
			this.initializeResizeInteraction(element, position);
		}
	}

	updateGridState() {
		this.grid_state = GRID_API.getState();
	}
}



/*
########  ##        #######  ########     ######  ########  ######
##     ## ##       ##     ##    ##       ##    ## ##       ##    ##
##     ## ##       ##     ##    ##       ##       ##       ##
########  ##       ##     ##    ##       ##       ######   ##   ####
##        ##       ##     ##    ##       ##       ##       ##    ##
##        ##       ##     ##    ##       ##    ## ##       ##    ##
##        ########  #######     ##        ######  ##        ######
*/

class PlotManager {
	constructor() {
		this.plots = {}; // Keyed by metricName, values are objects with Plotly plot div ID and settings
	}

	async createPlot(metricName) {
		// get ids
		const plotContainer_id = `plotContainer-${metricName}`;
		const plotDiv_id = `plot-${metricName}`;
		const plotSettings_id = `plotSettings-${metricName}`;

		// config and layout
		const plotConfig = await LAYOUT_MANAGER.getPlotConfig(metricName);
		const layout = LAYOUT_MANAGER.layout[plotContainer_id];

		const plotContainerHTML = `
			<div
				id="${plotContainer_id}"
				class="plotContainer" 
				style="margin-bottom: 10px; display: flex; flex-direction: row; position: absolute; width: ${layout.width}px; height: ${layout.height}px; left: ${layout.x}px; top: ${layout.y}px; ${DEFAULT_STYLE}"
			>
				<div 
					id="${plotDiv_id}"
					class="plotDiv" 
					style="width: ${layout.width - LAYOUT_CONFIG.settings_width_px}px; height: ${layout.height - LAYOUT_CONFIG.plot_bottom_margin_px}px;"
				></div>
				<div 
					id="${plotSettings_id}"
					class="plotSettings" 
					style="width: ${LAYOUT_CONFIG.settings_width_px}; flex-shrink: 0; flex-grow: 0;"
				></div>
			</div>
		`;

		// Add plot container to the root div
		document.getElementById('rootContainerDiv').insertAdjacentHTML('beforeend', plotContainerHTML);

		// Store plot info for later reference
		this.plots[metricName] = {
			plotID: plotDiv_id,
			containerID: plotContainer_id,
			settingsID: plotSettings_id,
		};

		// Specify plot layout and create empty plot
		const plotly_layout = {
			title: `${metricName} over ${plotConfig.xUnits}`,
			autosize: true,
			xaxis: {
				title: plotConfig.xUnits,
				type: plotConfig.axisScales.x,
				showgrid: true,
			},
			yaxis: {
				title: metricName,
				type: plotConfig.axisScales.y,
				showgrid: true,
			},
			margin: PLOTLY_LAYOUT_MARGIN,
			width: layout.width - LAYOUT_CONFIG.settings_width_px,
			height: layout.height - LAYOUT_CONFIG.plot_bottom_margin_px,
		};

		// Store layout
		PLOTLY_LAYOUTS[metricName] = plotly_layout;

		// To newPlot, pass copy, don't reference
		Plotly.newPlot(plotDiv_id, [], JSON.parse(JSON.stringify(plotly_layout)));

		// Add settings menu items
		this.createAxisToggles(metricName);
		this.createSmoothingInput(metricName);

		// Make draggable
		LAYOUT_MANAGER.makeElementDraggable(document.getElementById(plotContainer_id));
	}

	async createAllPlots(
		origin_x = 50,
		origin_y = 150,
	) {
		const metrics = DATA_MANAGER.metricNames;
		let n_metrics_counter = 0;

		metrics.forEach(metricName => {
			n_metrics_counter += 1;
			console.log(`creating plot ${n_metrics_counter} for ${metricName}`);
			this.createPlot(metricName);
		});
	}

	async updatePlot(metricName) {
		// get and set settings & config
		const plotInfo = this.plots[metricName];
		const plotConfig = await LAYOUT_MANAGER.getPlotConfig(metricName);
		if (!plotInfo) {
			console.error(`Plot for metric ${metricName} not found.`);
			return;
		}

		// get data
		// data manager will handle reloading the data, if necessary
		var traces = [];
		for (const runId in DATA_MANAGER.allRuns) {
			const run = DATA_MANAGER.allRuns[runId];
			const run_syllabic_id = run.meta.syllabic_id;

			const [x_vals, y_vals] = run.pairMetrics(DEFAULT_XUNITS, metricName);

			// Apply smoothing based on the selected method and span
			let smoothedYVals = RunData.smoothData(y_vals, plotConfig.smoothing_span, plotConfig.smoothing_method);

			const trace = {
				x: x_vals,
				y: smoothedYVals,
				mode: 'lines',
				line: plotConfig.smoothing_span ? { shape: 'spline' } : {},
				name: run_syllabic_id,
				visible: LAYOUT_MANAGER.visibilityState[run_syllabic_id] !== false ? true : 'legendonly',
			};
			traces.push(trace);
		}

		// Update the layout properties
		PLOTLY_LAYOUTS[metricName].xaxis.type = plotConfig.axisScales.x;
		PLOTLY_LAYOUTS[metricName].yaxis.type = plotConfig.axisScales.y;
		PLOTLY_LAYOUTS[metricName].uirevision = metricName;

		// Update Plotly plot
		Plotly.react(
			plotInfo.plotID,
			traces,
			JSON.parse(JSON.stringify(PLOTLY_LAYOUTS[metricName])),
		);
	}

	updateTraceVisibility(runId, isVisible) {
		LAYOUT_MANAGER.visibilityState[runId] = isVisible;
		for (const metricName of DATA_MANAGER.metricNames) {
			const plotInfo = this.plots[metricName];
			if (plotInfo) {
				Plotly.restyle(
					plotInfo.plotID,
					{ visible: isVisible ? true : 'legendonly' },
					[this.getTraceIndex(plotInfo.plotID, runId)],
				);
			}
		}
	}

	getTraceIndex(plotId, runId) {
		const plotDiv = document.getElementById(plotId);
		const data = plotDiv.data;
		const index = data.findIndex(trace => trace.name === runId);
		if (index < 0) {
			console.error(`Trace for run ${runId} not found in plot ${plotId}`);
		}
		return index;
	}

	updateAllVisibility() {
		for (const metricName of DATA_MANAGER.metricNames) {
			this.updatePlot(metricName);
		}
	}

	async populateAllPlots() {
		// for each metric
		for (const metricName of DATA_MANAGER.metricNames) {
			this.updatePlot(metricName);
		}
	}

	updateAxisScale(metricName, axis, scale) {
		// get plot info
		const plotInfo = this.plots[metricName];
		if (!plotInfo) {
			console.error(`Plot for metric ${metricName} not found.`);
			return;
		}

		// Update scale in settings
		const plotConfig = LAYOUT_MANAGER.plot_configs[metricName];
		plotConfig.axisScales[axis] = scale;

		// Reflect change in Plotly plot
		Plotly.relayout(
			plotInfo.plotID,
			{
				[`${axis}axis`]: { type: scale },
				uirevision: metricName,
			},
		);
	}


	createAxisToggles(metricName) {
		const plotSettingsId = this.plots[metricName].settingsID;
		const plotDivId = this.plots[metricName].plotID;
		const plotSettings = document.getElementById(plotSettingsId);

		['x', 'y'].forEach(axis => {
			const toggleId = `${plotDivId}-${axis}Toggle`;
			const toggleHtml = `
				<div class="axis-toggle-container" style="display: block;">
					<label for="${toggleId}" style="display: block;">${axis.toUpperCase()} Scale</label>
					<div style="display: flex; align-items: center;">
						<i data-feather="arrow-up-right">lin</i>
						<label class="switch">
							<input type="checkbox" id="${toggleId}">
							<span class="slider round"></span>
						</label>
						<i data-feather="corner-right-up">log</i>
					</div>
				</div>
			`;

			const toggleDiv = document.createElement('div');
			toggleDiv.innerHTML = toggleHtml.trim();
			plotSettings.appendChild(toggleDiv);

			const input = document.getElementById(toggleId);
			input.checked = LAYOUT_MANAGER.plot_configs[metricName].axisScales[axis] === 'log';
			input.onchange = () => {
				const scale = input.checked ? 'log' : 'linear';
				this.updateAxisScale(metricName, axis, scale);
			};

		});
	}

	async createSmoothingInput(metricName) {
		// get the div ids
		const plotDivId = this.plots[metricName].plotID;
		const plotSettingsId = this.plots[metricName].settingsID;

		// Define the smoothing methods
		const smoothingMethods = ['SMA', 'EMA', 'Gaussian'];

		// Create the HTML string for the smoothing input container
		const smoothSettingHtml = `
			<div class="smoothing-input-container no-drag" style="display: block; margin-top: 10px; border: 1px solid grey; border-radius: 3px;">
				<label for="smoothingInput-${plotDivId}" style="font-weight: bold;">Smooth:</label><br>
				<label for="smoothingMethodSelect-${plotDivId}">Method</label><br>
				<select class="no-drag" id="smoothingMethodSelect-${plotDivId}" style="width: 6em;">
					${smoothingMethods.map(method => `<option value="${method}">${method}</option>`).join('')}
				</select><br>
				<label for="smoothingInput-${plotDivId}">Span</label><br>
				<input class="no-drag" type="number" min="0" max="1000" value="0" id="smoothingInput-${plotDivId}" style="width: 4.2em;">
			</div>
		`;

		// Create a container for the smoothing input
		const smoothSettingContainer = document.createElement('div');
		smoothSettingContainer.innerHTML = smoothSettingHtml.trim();

		// Append the input container to the plot settings
		const plotSettings = document.getElementById(plotSettingsId);
		plotSettings.appendChild(smoothSettingContainer);

		// Get references to the input elements
		const spanInput = document.getElementById(`smoothingInput-${plotDivId}`);
		const smoothingMethodSelect = document.getElementById(`smoothingMethodSelect-${plotDivId}`);

		// Set values to those from plot_configs
		spanInput.value = LAYOUT_MANAGER.plot_configs[metricName].smoothing_span;
		smoothingMethodSelect.value = LAYOUT_MANAGER.plot_configs[metricName].smoothing_method;

		// On change, modify the plot config and call updatePlot
		spanInput.onchange = () => {
			LAYOUT_MANAGER.plot_configs[metricName].smoothing_span = parseInt(spanInput.value);
			this.updatePlot(metricName);
		};
		smoothingMethodSelect.onchange = () => {
			LAYOUT_MANAGER.plot_configs[metricName].smoothing_method = smoothingMethodSelect.value;
			this.updatePlot(metricName); // Update the plot when smoothing method changes
		};
	}
}

let PLOT_MANAGER = new PlotManager();



/*
##     ## ########    ###    ########  ######## ########
##     ## ##         ## ##   ##     ## ##       ##     ##
##     ## ##        ##   ##  ##     ## ##       ##     ##
######### ######   ##     ## ##     ## ######   ########
##     ## ##       ######### ##     ## ##       ##   ##
##     ## ##       ##     ## ##     ## ##       ##    ##
##     ## ######## ##     ## ########  ######## ##     ##
*/

async function headerButtons() {
	// get the project name, set the header
	const projectH1 = document.getElementById('projectH1');
	projectH1.innerHTML = `${DATA_MANAGER.projectName} <a href="https://github.com/mivanit/trnbl">trnbl</a> Dashboard`;
	const gridSnapCheckbox = document.getElementById('gridSnapCheckbox');

	// set up grid snap checkbox
	gridSnapCheckbox.checked = LAYOUT_MANAGER.do_snap;
	gridSnapCheckbox.addEventListener('change', function () {
		LAYOUT_MANAGER.updateSnap(gridSnapCheckbox.checked);
	});

	// save layout to local storage
	document.getElementById('saveLayoutButton').addEventListener(
		'click',
		async () => {
			await LAYOUT_MANAGER.saveLayout();
		}
	);

	// download layout as json
	document.getElementById('downloadLayoutButton').addEventListener(
		'click',
		async () => {
			const layoutKey = LAYOUT_MANAGER.get_local_storage_key();
			const layout_json = JSON.stringify(LAYOUT_MANAGER, null, '\t');
			const blob = new Blob([layout_json], { type: 'application/json' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = layoutKey + '.json';
			document.body.appendChild(a);
			a.click();
			a.remove();
			URL.revokeObjectURL(url);
		}
	);

	// reset layout to default
	document.getElementById('resetLayoutButton').addEventListener(
		'click',
		async () => {
			// delete
			const layoutKey = LAYOUT_MANAGER.get_local_storage_key();
			IO_MANAGER.deleteJsonLocal(layoutKey);
			// reload page
			location.reload();
			createNotification('Layout resetting...', 'info');
		}
	);

	// Set up manual refresh button
	document.getElementById('refreshButton').addEventListener(
		'click',
		async () => {
			await DATA_MANAGER.refreshData();
		}
	);

	// Set up auto-refresh
	DATA_MANAGER.initAutoRefresh();


	// reset colum state of table
	document.getElementById('resetColumnStateButton').addEventListener(
		'click',
		async () => {
			GRID_API.resetColumnState();
			createNotification('Column state reset', 'info');
		}
	);

	// Toggle visibility of runs
	document.getElementById('toggleVisibleRowsButton').addEventListener(
		'click',
		() => toggleRowsVisibility(true)
	);

	document.getElementById('toggleFilteredRowsButton').addEventListener(
		'click',
		() => toggleRowsVisibility(false)
	);
}

/*
##    ##  #######  ######## #### ########
###   ## ##     ##    ##     ##  ##
####  ## ##     ##    ##     ##  ##
## ## ## ##     ##    ##     ##  ######
##  #### ##     ##    ##     ##  ##
##   ### ##     ##    ##     ##  ##
##    ##  #######     ##    #### ##
*/


function blinkElement(elementId, color, duration) {
	const element = document.getElementById(elementId);
	element.classList.add('blink-border');
	setTimeout(() => {
		element.classList.remove('blink-border');
	}, 1000); // Remove class after animation completes
}

function createNotification(message, type = 'info', extra = null, show = true) {
	const log_str = `[${type}]: ${message}\n${extra ? extra : ''}`;
	// print to console
	switch (type) {
		case 'info':
			console.log(log_str);
			if (extra) { console.log(extra); };
			break;
		case 'warning':
			console.warn(log_str);
			if (extra) { console.warn(extra); };
			break;
		case 'error':
			console.error(log_str);
			if (extra) { console.error(extra); };
			break;
		default:
			console.log(log_str);
			if (extra) { console.log(extra); };
	}

	if (show) {

		// create notification div
		const notificationDiv = document.createElement('div');
		notificationDiv.textContent = message;
		notificationDiv.style.cssText = `
			position: fixed;
			top: 10px;
			right: 10px;
			padding: 10px;
			border-radius: 5px;
			background-color: ${NOTIFICATION_CONFIG.colors[type]};
			border: 1px solid ${NOTIFICATION_CONFIG.border_colors[type]};
			box-shadow: 0 2px 5px rgba(0,0,0,0.2);
			transition: transform 0.3s ease-out, opacity 0.3s ease-out;
			z-index: 1000;
			opacity: 0;  // Start with 0 opacity for fade-in effect
		`;

		// Function to update positions of all notifications
		function updateNotificationPositions() {
			const notifications = document.querySelectorAll('.notification');
			let currentTop = 10;
			notifications.forEach((notification) => {
				notification.style.transform = `translateY(${currentTop}px)`;
				currentTop += notification.offsetHeight + 10; // 10px gap between notifications
			});
		}

		// Add a class for easier selection
		notificationDiv.classList.add('notification');

		// Insert the new notification at the top
		const firstNotification = document.querySelector('.notification');
		if (firstNotification) {
			document.body.insertBefore(notificationDiv, firstNotification);
		} else {
			document.body.appendChild(notificationDiv);
		}

		// Trigger reflow to ensure the initial state is applied before changing opacity
		notificationDiv.offsetHeight;

		// Fade in the notification
		notificationDiv.style.opacity = '1';

		// Update positions after a short delay to allow for DOM update
		setTimeout(updateNotificationPositions, 10);

		// Remove the notification after 3 seconds
		setTimeout(
			() => {
				notificationDiv.style.opacity = '0';
				notificationDiv.style.transform += ' translateX(100%)';
				setTimeout(() => {
					notificationDiv.remove();
					updateNotificationPositions();
				}, 300); // Match this with the CSS transition time
			},
			NOTIFICATION_CONFIG.timeout,
		);
	}
}


/*
########    ###    ########  ##       ########
   ##      ## ##   ##     ## ##       ##
   ##     ##   ##  ##     ## ##       ##
   ##    ##     ## ########  ##       ######
   ##    ######### ##     ## ##       ##
   ##    ##     ## ##     ## ##       ##
   ##    ##     ## ########  ######## ########
*/


function isISODate(value) {
	// This regex matches ISO 8601 date strings with optional fractional seconds and timezone information
	const dateRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(.\d+)?(Z|[+-]\d{2}:\d{2})?$/;
	return dateRegex.test(value);
}

// fancy cell rendering -- hover/copy/open the data, make it emojis if its too long
function fancyCellRenderer(params) {
	// check if params.value is undefined
	var value;
	if (params.value === undefined) {
		return div;
	}
	else {
		value = params.value;
	}
	// Create the div element
	var div = document.createElement('div');
	// set content
	div.title = value;
	div.textContent = value;
	div.style.cursor = 'pointer';
	// if its too long, make it emojis
	if (value !== null) {
		// if object, convert to string
		if (typeof value === 'object') {
			value = JSON.stringify(value, null, 4);
		}
		if (value.length > 50) {
			div.title = value;
			div.innerHTML = feather.icons["mouse-pointer"].toSvg() + feather.icons["copy"].toSvg();
			div.style.cssText = 'font-size: 20px; display: flex; justify-content: center; align-items: center; background-color: #f4f4f4; border: 1px solid #d4d4d4; border-radius: 5px; height: 30px; width: 60px;';
		}
	}

	// Add click event listener to copy text to the clipboard
	div.onclick = function () {
		navigator.clipboard.writeText(value).then(function () {
			console.log('Successfully copied to clipboard');
		}).catch(function (err) {
			console.error('Could not copy text to clipboard: ', err);
		});
	};

	// on right click, open a new plain text tab whose contents are the cell's value
	div.oncontextmenu = function () {
		const newWindow = window.open('', '_blank');
		// set the contents of the new window to the cell's value
		newWindow.document.write('<pre>' + value + '</pre>');
		// set the title of the page to the rows "name.default_alias" and the column's header
		newWindow.document.title = params.node.data.id.run + ' : ' + params.colDef.headerName; // TODO: page has "undefined" in title
		newWindow.document.close();
	};

	// Return the div as the cell's DOM
	return div;
}


function createColumnDefs(summaryManifest) {
	var columnDefs = [
		{
			headerName: 'View/Hide',
			field: 'visible',
			width: 100,
			cellRenderer: params => {
				const cellDiv = document.createElement('div');
				cellDiv.className = 'ag-cell-wrapper';
				cellDiv.style.display = 'flex';
				cellDiv.style.alignItems = 'center';
				cellDiv.style.justifyContent = 'center';

				const checkbox = document.createElement('input');
				checkbox.type = 'checkbox';
				checkbox.checked = params.value !== false;
				checkbox.style.marginRight = '5px';

				const iconDiv = document.createElement('div');
				iconDiv.innerHTML = feather.icons[checkbox.checked ? 'eye' : 'eye-off'].toSvg();
				iconDiv.style.pointerEvents = 'none'; // Make icon non-interactive

				cellDiv.appendChild(checkbox);
				cellDiv.appendChild(iconDiv);

				const updateVisibility = (isVisible) => {
					params.setValue(isVisible);
					checkbox.checked = isVisible;
					iconDiv.innerHTML = feather.icons[isVisible ? 'eye' : 'eye-off'].toSvg();
					PLOT_MANAGER.updateTraceVisibility(params.data.id.syllabic, isVisible);
				};

				checkbox.addEventListener('change', () => updateVisibility(checkbox.checked));
				cellDiv.addEventListener('click', (event) => {
					if (event.target !== checkbox) {
						updateVisibility(!checkbox.checked);
					}
				});

				return cellDiv;
			},
		},
	];

	// date filter
	const date_def = {
		filter: 'agDateColumnFilter',
		filterParams: {
			comparator: function (filterValue, cellValue) {
				// Assuming cellValue is an ISO date string
				const cellDate = new Date(cellValue);
				const filterDate = new Date(filterValue);
				if (cellDate < filterDate) {
					return -1;
				} else if (cellDate > filterDate) {
					return 1;
				}
				return 0;
			},
			// Disable the use of the browser-provided date picker for this filter
			browserDatePicker: false,
			// Add the inRange filter option
			inRangeInclusive: true,
		}
	}


	// Define column groups
	const columnGroupDefs = [
		{
			headerName: 'Name',
			children: [
				{ field: 'id.syllabic', headerName: 'Syllabic ID', columnGroupShow: null },
				{ field: 'id.run', headerName: 'Full Run ID', columnGroupShow: 'open' },
				{ field: 'id.group', headerName: 'Group', columnGroupShow: 'open' },
			],
			marryChildren: true,
		},
		{
			headerName: 'Timing',
			children: [
				{ field: 'timing.start', headerName: 'Start', columnGroupShow: null, ...date_def },
				{ field: 'timing.final', headerName: 'End', columnGroupShow: 'open', ...date_def },
				{ field: 'timing.duration', headerName: 'Duration (ms)', columnGroupShow: 'open', },
			],
			marryChildren: true,
		},
		{
			headerName: 'Final Metrics',
			children: [],
		},
		{
			headerName: 'Final State',
			children: [
				{ field: 'final_state.samples', headerName: 'Samples', columnGroupShow: null },
				{ field: 'final_state.batches', headerName: 'Batches', columnGroupShow: 'open' },
				{ field: 'final_state.epochs', headerName: 'Epochs', columnGroupShow: 'open' },
				{ field: 'final_state.latest_checkpoint', headerName: 'Checkpoints', columnGroupShow: 'open' },
			],
		},
		{
			headerName: 'Config',
			children: [
				{
					field: 'config',
					headerName: 'Config',
					columnGroupShow: null,
					// width: 50, // TODO: this width is broken
					cellRenderer: fancyCellRenderer,
					valueFormatter: params => {
						if (params.value === null || params.value === undefined) {
							return '';
						}
						if (typeof params.value === 'object') {
							return JSON.stringify(params.value);
						}
						return params.value.toString();
					},
				},
				{
					field: 'meta',
					headerName: 'Metadata',
					columnGroupShow: null,
					cellRenderer: fancyCellRenderer,
					valueFormatter: params => {
						if (params.value === null || params.value === undefined) {
							return '';
						}
						if (typeof params.value === 'object') {
							return JSON.stringify(params.value);
						}
						return params.value.toString();
					},
				}
			],
			marryChildren: true,
		},
	];

	// Dynamically add final metric columns
	const finalMetricKeys = new Set();
	summaryManifest.forEach(item => {
		Object.keys(item.final_metrics).forEach(key => finalMetricKeys.add(key));
	});
	var final_metrics_counter = 0;
	finalMetricKeys.forEach(key => {
		columnGroupDefs[2].children.push({
			field: `final_metrics.${key}`,
			headerName: key,
			columnGroupShow: final_metrics_counter === 1 ? null : 'open',
		});
		final_metrics_counter += 1;
	});

	// Add column group definitions to the main column definitions
	columnDefs = columnDefs.concat(columnGroupDefs);

	return columnDefs;
}

function adjustTableHeight(table) {
	// Adjust the height of the table container
	const gridHeight = table.querySelector('.ag-center-cols-viewport').offsetHeight;
	const headerHeight = table.querySelector('.ag-header').offsetHeight;
	const paginationHeight = table.querySelector('.ag-paging-panel').offsetHeight;
	const tableMinHeight = gridHeight + headerHeight + paginationHeight + 50;
	table.style.minHeight = `${tableMinHeight}px`;
}

function toggleRowsVisibility(affectVisible) {
	const visibleRows = new Set(GRID_API.getRenderedNodes().map(node => node.data.id.syllabic));
	let rowsToToggle = [];
	let newVisibility;

	GRID_API.forEachNode(node => {
		const isVisible = visibleRows.has(node.data.id.syllabic);
		if (affectVisible === isVisible) {
			rowsToToggle.push(node);
		}
	});

	if (rowsToToggle.length > 0) {
		newVisibility = !rowsToToggle[0].data.visible;
	}

	rowsToToggle.forEach(node => {
		node.setDataValue('visible', newVisibility);
		LAYOUT_MANAGER.visibilityState[node.data.id.syllabic] = newVisibility;
	});

	// Update plots
	DATA_MANAGER.metricNames.forEach(metricName => {
		PLOT_MANAGER.updatePlot(metricName);
	});

	GRID_API.refreshCells({
		force: true,
		columns: ['visible'],
		rowNodes: rowsToToggle
	});
}

function createRunsManifestTable(summaryManifest) {
	// create plot container
	const runsManifestTable = document.createElement('div');
	runsManifestTable.id = 'runsManifest';
	runsManifestTable.classList.add('runsManifestBox', 'ag-theme-alpine');
	document.getElementById('rootContainerDiv').appendChild(runsManifestTable);

	// load layout
	const layout = LAYOUT_MANAGER.layout[runsManifestTable.id];
	if (layout) {
		runsManifestTable.style.cssText = `position: absolute; width: ${layout.width}px; height: ${layout.height}px; left: ${layout.x}px; top: ${layout.y}px; margin-bottom: 20px; ${DEFAULT_STYLE}`;
	}
	// make draggable
	LAYOUT_MANAGER.makeElementDraggable(runsManifestTable);

	// create the grid options
	const gridOptions = {
		columnDefs: createColumnDefs(summaryManifest),
		rowData: summaryManifest,
		pagination: true,
		enableCellTextSelection: true,
		enableBrowserTooltips: true,
		rowSelection: 'multiple',
		// customize pagination
		pagination: true,
		paginationPageSize: 10,
		paginationPageSizeSelector: [1, 2, 5, 10, 25, 50, 100, 500, 1000],
		defaultColDef: {
			resizable: true,
			filter: true,
			// always show the floating filter
			floatingFilter: true,
			// disable filter hamburger menu (for space)
			menuTabs: [],
		},
		domLayout: 'autoHeight',
		onFirstDataRendered: function (params) {
			adjustTableHeight(runsManifestTable);
		},
		onPaginationChanged: function (params) {
			adjustTableHeight(runsManifestTable);
		},
		initialState: LAYOUT_MANAGER.grid_state,
	};

	// create the ag-Grid table, api to global
	// api is used in LayoutManager.updateGridState
	GRID_API = agGrid.createGrid(runsManifestTable, gridOptions);

	// load in the visibility state of the runs
	GRID_API.forEachNode(node => {
		const runId = node.data.id.syllabic;
		const isVisible = LAYOUT_MANAGER.visibilityState[runId] !== false;
		node.setDataValue('visible', isVisible);
	});
	GRID_API.refreshCells({ force: true, columns: ['visible'] });
	// set the checkbox state
	GRID_API.onFilterChanged();
}




/*
#### ##    ## #### ########
 ##  ###   ##  ##     ##
 ##  ####  ##  ##     ##
 ##  ## ## ##  ##     ##
 ##  ##  ####  ##     ##
 ##  ##   ###  ##     ##
#### ##    ## ####    ##
*/


async function init() {
	// load basic data
	await DATA_MANAGER.loadManifest();

	// layout stuff
	LAYOUT_MANAGER = new LayoutManager(DATA_MANAGER.projectName);
	await LAYOUT_MANAGER.loadLayout(do_update = false);

	// set up header and buttons
	await headerButtons();

	// create empty plots
	await PLOT_MANAGER.createAllPlots();

	// load data
	await DATA_MANAGER.loadRuns();

	// populate table and get grid API
	await createRunsManifestTable(DATA_MANAGER.summaryManifest);

	// populate plots
	await PLOT_MANAGER.populateAllPlots();

	// feather icons
	try {
		// replace the icons
		feather.replace();
		// if no errors, look for any <i> tags with class `data-feather` and remove the text
		const featherIcons = document.querySelectorAll('i[data-feather]');
		featherIcons.forEach(icon => {
			icon.innerHTML = '';
		});
	}
	catch (e) {
		createNotification('Feather icons not found, keeping text fallback', 'error');
	}
	console.log('init complete');

	feather.replace();

	await DATA_MANAGER.refreshData()
}
