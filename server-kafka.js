/*
 Copyright (c) 2016-2017 The Nirdizati Project.
 This file is part of "Nirdizati".

 "Nirdizati" is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as
 published by the Free Software Foundation; either version 3 of the
 License, or (at your option) any later version.

 "Nirdizati" is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty
 of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 See the GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this program.
 If not, see <http://www.gnu.org/licenses/lgpl.html>.
 */

'use strict';

const bodyParser = require('body-parser'),
	favicon = require('serve-favicon'),
	config = require('config'),
	express = require('express'),
	http = require('http'),
	logger = require('morgan'),
	path = require('path');

const log = require('./libs/utils/log.js')(module),
	routes = require('./routes/index');

const app = express();

app.use(favicon(path.join(__dirname, 'public', 'media', 'favicon.ico')));

if (app.get('env') === 'development') {
	app.use(logger('dev'));
} else {
	app.use(logger('combined'));
}

app.use(bodyParser.json());
app.use(bodyParser.text());
app.use(bodyParser.urlencoded({
	    extended: false
}));

app.use(['/'], routes);
app.use(express.static(path.join(__dirname, '/public')));

//catch 404 and forward to error handler
app.use(function(req, res, next) {
	const err = new Error('Not Found');
	err.status = 404;
	next(err);
});

// no stacktraces leaked to user unless in development environment
app.use(function(err, req, res, next) {
	res.status(err.status || 500).json({
		message: err.message,
		error: (app.get('env') === 'development') ? err : {}
	});
});

const server = http.createServer(app);
const port = process.argv[2] || config.get('app.port');

server.listen(port, () => {
	log.info(`Express server listening on port ${port}`);

	// clean database and queue from previous runs
	if (app.get('env') === 'development') {
		require('./libs/utils/dbSetup')();
		require('./libs/utils/queueSetup')();
	}

	const io = require('./libs/socket')(server);
	require('./libs/queue/jobWorker-kafka')(io);
	require('./libs/queue/uiWorker')(io);
});
