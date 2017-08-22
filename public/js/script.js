"use strict";function createAllCharts(){Chart.defaults.global.defaultFontSize=16,outcomeCharts=_createOutcomeCharts(),caseDurationCharts?caseDurationCharts.update():caseDurationCharts=_createBarChart("caseDurationCharts"),remainingTimeChart?remainingTimeChart.update():remainingTimeChart=_createBarChart("remainingTimeChart"),lengthDistributionCharts?lengthDistributionCharts.update():lengthDistributionCharts=_createBarChart("lengthDistributionChart")}function _createOutcomeCharts(){var e=[];document.getElementById("TabOutcomes").innerHTML="";for(var t=0;t<outcomeConfigs.length;t++){var a=document.querySelector("#TabOutcomes"),n=document.createElement("div"),r=document.createElement("div"),i=document.createElement("div"),c=document.createElement("div"),o=document.createElement("div"),s=document.createElement("div"),l=document.createElement("div"),m=document.createElement("div"),d=document.createElement("canvas"),u=document.createElement("canvas"),h=document.createElement("canvas");n.className="row",r.className="outcomeTitle",r.innerHTML=outcomeConfigs[t].name,i.className="outcomeTitleType",i.innerHTML="Running",c.className="outcomeTitleType",c.innerHTML="Completed",o.className="outcomeTitleType",o.innerHTML="Historical",s.className="col-xxs-12 col-xs-4",l.className="col-xxs-12 col-xs-4",m.className="col-xxs-12 col-xs-4",d.setAttribute("id","chart-predicted-"+t),u.setAttribute("id","chart-completed-"+t),h.setAttribute("id","chart-historical-"+t),s.appendChild(i),s.appendChild(d),l.appendChild(c),l.appendChild(u),m.appendChild(o),m.appendChild(h),n.appendChild(r),n.appendChild(s),n.appendChild(l),n.appendChild(m),a.appendChild(n),e.push({predicted:new Chart(d.getContext("2d"),outcomeConfigs[t].predicted),completed:new Chart(u.getContext("2d"),outcomeConfigs[t].completed),historical:new Chart(h.getContext("2d"),outcomeConfigs[t].historical)})}return e}function _createBarChart(e){const t=document.getElementById(e).getContext("2d");switch(e){case"remainingTimeChart":return new Chart(t,remainingTimeConfigs);case"caseDurationCharts":return new Chart(t,caseDurationConfigs);case"lengthDistributionChart":return new Chart(t,lengthDistributionConfigs)}}var outcomeCharts=null,remainingTimeChart=null,caseDurationCharts=null,lengthDistributionCharts=null;
"use strict";function sendLogChange(t){var e=document.getElementById("logSelect");socket.emit(t||"changeLog",e.options[e.selectedIndex].value)}function onInit(t){t&&(initConfigs(t.configs),initTable(!0),createAllCharts(),updateParams(t),loadTable(t),reloadCharts(t),$(".first-load-spinner").hide())}function updateUI(t){t&&(updateParams(t),reloadCharts(t),updateRow(t))}function updateParams(t){$("#running").text(t.runningCases.toLocaleString()),$("#completedCases").text(t.completedCases.toLocaleString()),$("#completedEvents").text(t.completedEvents.toLocaleString()),$("#averageCaseLength").text(t.averageCaseLength),$("#averageDuration").text(timestampToDuration(t.averageRunningTime))}function loadTable(t){$("#table").bootstrapTable("load",t.table)}function reloadCharts(t){_reloadOutcomeCharts(t),_reloadCaseDurationCharts(t),_reloadRemainingTimeChart(t),_reloadLengthDistributionChart(t)}function _reloadOutcomeCharts(t){if(outcomeCharts){const e=t.outcomes,a=Object.keys(e);for(var n=0;n<a.length;n++)outcomeCharts[n].predicted.data.datasets[0].data=e[a[n]].predicted,outcomeCharts[n].completed.data.datasets[0].data=e[a[n]].completed,outcomeCharts[n].predicted.update(),outcomeCharts[n].completed.update()}}function _reloadRemainingTimeChart(t){remainingTimeChart&&(remainingTimeChart.data.datasets[0].data=_aggregateBarChartResults(t.remainingTimeCounts,generalDurationConfigs.daysInterval,generalDurationConfigs.barsNumber),remainingTimeChart.update())}function _reloadCaseDurationCharts(t){caseDurationCharts&&(caseDurationCharts.data.datasets[0].data=_aggregateBarChartResults(t.completedCaseDurationCounts,generalDurationConfigs.daysInterval,generalDurationConfigs.barsNumber),caseDurationCharts.data.datasets[1].data=_aggregateBarChartResults(t.runningCaseDurationCounts,generalDurationConfigs.daysInterval,generalDurationConfigs.barsNumber),caseDurationCharts.update())}function _reloadLengthDistributionChart(t){lengthDistributionCharts&&(lengthDistributionCharts.data.datasets[0].data=_aggregateBarChartResults(t.lengthDistributionCounts,lengthDistributionConfigs.width,lengthDistributionConfigs.barsNumber),lengthDistributionCharts.update())}function updateRow(t){var e=t.row[0];if(!e)return console.log("Row param is empty.");$("#table").bootstrapTable("remove",{field:"case_id",values:[e.case_id]}),$("#table").bootstrapTable("prepend",e)}function handleError(t){t&&console.log(t)}function _aggregateBarChartResults(t,e,a){for(var n=Array(a).fill(0),r=0,o=e,i=0;i<a-1;i++)n[i]=t.slice(r,o).reduce(function(t,e){return t+e},0),r=o,o+=e;return n[a-1]=t.slice(r).reduce(function(t,e){return t+e},0),n}var socket=io();socket.on("provideLog",function(){sendLogChange("provideLog")}).on("init",function(t){onInit(t)}).on("event",function(t){updateUI(t)}).on("last event",function(t){updateUI(t)}).on("results",function(t){updateUI(t)}).on("error",function(t){handleError(t)});
"use strict";function _getPieChartConfig(t,a){return{type:"pie",data:{datasets:[{data:t,backgroundColor:pieChartColors}],labels:a},options:{responsive:!0}}}function initConfigs(t){if(t||console.log("Configs from server are empty"),t.columns&&(tableConfig.columns=t.columns),t.ui){generalDurationConfigs.daysInterval=t.ui.daysInterval,generalDurationConfigs.barsNumber=t.ui.barsCountForTimeIntervals,lengthDistributionConfigs.width=t.ui.barsWidthForLength,lengthDistributionConfigs.barsNumber=t.ui.barsCountInLengthDistribution;const a=_generateBarChartsLabels(generalDurationConfigs.daysInterval,generalDurationConfigs.barsNumber,"Day");remainingTimeConfigs.data.labels=a,caseDurationConfigs.data.labels=a,lengthDistributionConfigs.data.labels=_generateBarChartsLabels(lengthDistributionConfigs.width,lengthDistributionConfigs.barsNumber,null)}if(t.outcomes){outcomeConfigs=[];const e=t.outcomes;for(var i=0;i<e.length;i++){const r=e[i].labels;outcomeConfigs[i]={name:e[i].name,predicted:_getPieChartConfig([0,0],r),completed:_getPieChartConfig([0,0],r),historical:_getPieChartConfig(e[i].historical,r)}}}}function _generateBarChartsLabels(t,a,e){e&&(e+=1===t?"":"s");for(var i=0,r=i+t,o=[],n=0;n<a-1;n++)o.push(i.toString()+"-"+r.toString()+(e?" "+e:"")),i=r,r+=t;return o.push(">"+i.toString()+(e?" "+e:"")),o}var outcomeConfigs=[];const pieChartColors=["rgba(191, 63, 63, 0.8)","rgba(63, 191, 63, 0.8)","blue","yellow"],barChartOptions={responsive:!0,scales:{display:!0,yAxes:[{ticks:{beginAtZero:!0}}]}};var generalDurationConfigs={},caseDurationConfigs={type:"bar",data:{labels:[],datasets:[{label:"Actual case duration (over completed)",backgroundColor:Array(20).fill("rgba(63, 191, 63, 0.8)"),data:[]},{label:"Predicted case duration (over running)",backgroundColor:Array(20).fill("rgba(54, 162, 235, 0.5)"),data:[]}]},options:barChartOptions},remainingTimeConfigs={type:"bar",data:{labels:[],datasets:[{label:"Remaining time distribution",backgroundColor:Array(20).fill("rgba(54, 162, 235, 0.5)"),data:[]}]},options:barChartOptions},lengthDistributionConfigs={type:"bar",data:{labels:[],datasets:[{label:"Case length distribution (over completed)",backgroundColor:Array(20).fill("rgba(63, 191, 63, 0.8)"),data:[]}]},options:barChartOptions,width:1,barsNumber:9},tableConfig={showToggle:"true",showColumns:"true",showExport:"true",showPaginationSwitch:"true",idField:"case_id",search:"true",striped:"true",pagination:"true",pageSize:5,paginationLoop:!0,pageList:[5,10,25,50,100,200],buttonsAlign:"right",searchAlign:"left",rowStyle:"rowStyle",columns:[]};
"use strict";function timestampToDuration(I){if(!I&&0!==I)return"";const L=Math.floor(I/MILLISEC_IN_DAY),_=Math.floor(I%MILLISEC_IN_DAY/MILLISEC_IN_HOUR),M=Math.floor(I%MILLISEC_IN_HOUR/MILLISEC_IN_MINUTE),E=Math.floor(I%MILLISEC_IN_MINUTE/1e3);var N="";return L>0&&(N=L+"d "),N+=_+"h "+M+"m ",0==L&&(N+=E+"s"),N}const MILLISEC_IN_MINUTE=6e4,MILLISEC_IN_HOUR=60*MILLISEC_IN_MINUTE,MILLISEC_IN_DAY=24*MILLISEC_IN_HOUR;
"use strict";$(document).ready(function(){initTable(!1),$(".ttab").on("click",function(){$(".ttabs-head .ttab").removeClass("active-btn"),$(this).addClass("active-btn");var t=$(".ttabs-head .ttab").index(this);$(".ttabs-body .ttabs-content").removeClass("active-content"),$(".ttabs-body .ttabs-content").eq(t).addClass("active-content")})}),$(window).on("resize",function(){window.matchMedia("(min-width: 350px) and (max-width: 480px)").matches||window.matchMedia("(max-width: 350px)").matches?tableState.setCurrentState("mobile"):tableState.setCurrentState("desktop")});
"use strict";function TableState(t){this.tableStates=["mobile","desktop"],this.currentState=this.isState(t)?t:null,this.makeState()}function rowStyle(t){return!0===t.finished?{classes:"success"}:{classes:"active"}}function cellStyle(t){return!0===t?{classes:"danger"}:{}}function toStringFormatter(t){return null===t||void 0===t?"-":t.toString()}function arrayFormatter(t){return t.join("; \n")}function timestampFormatter(t){if(t)return moment(new Date(t)).format("MMM DD YYYY HH:mm:ss")}function initTable(t){t?$("#table").bootstrapTable("refreshOptions",tableConfig):$("#table").bootstrapTable(tableConfig),$("#table").bootstrapTable("hideColumn","current_trace"),tableState=new TableState(window.matchMedia("(min-width: 350px) and (max-width: 480px)").matches||window.matchMedia("(max-width: 350px)").matches?"mobile":"desktop")}TableState.prototype.isState=function(t){return-1!==this.tableStates.indexOf(t)},TableState.prototype.setCurrentState=function(t){this.isState(t)&&this.currentState!=t&&(this.currentState=t,this.makeState())},TableState.prototype.makeState=function(){if(this.currentState&&("mobile"==this.currentState&&($("#table").bootstrapTable("getOptions").cardView||$("#table").bootstrapTable("toggleView"),$(".bootstrap-table .fixed-table-toolbar").hide()),"desktop"==this.currentState)){$("#table").bootstrapTable("getOptions").cardView&&$("#table").bootstrapTable("toggleView"),$(".bootstrap-table .fixed-table-toolbar").show();var t=$(".bootstrap-table .fixed-table-toolbar .search").outerHeight();$(".ttabs-content .fixed-table-toolbar").animate({height:t},50)}};var tableState=null;