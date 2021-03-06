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

const config = require('config'),
	kafka = require('kafka-node');

const db = require('../../db'),
	consumerTopic = config.get("kafka.eventsWithPredictions"),
	consumer = new kafka.ConsumerGroup({}, [ consumerTopic ]),
	log = require('../utils/logger.js')(module);

module.exports = function(io) {
	log.info(`UI worker connecting to topic ${consumerTopic}...`);

	consumer.on('error', (error) => {
		log.error(`Error: ${error}`);
	});

	consumer.on('offsetOutOfRange', (error) => {
		log.error(`Offset out of range: ${error.message}`);
	});

	consumer.on('message', (message) => {
		const results = JSON.parse(message.value),
			payload = results.payload,
			logName = payload.event['log'];

		log.info(`Event with predictions: ${JSON.stringify(results)}`);

		db.consumeEvent(payload.event, (err) => {
			if (err) {
				return io.to(logName).emit('error', err);
			}

			log.info(`Consumed event ${JSON.stringify(payload.event)}`);

			db.handleResults(results, (err) => {
				if (err) {
					return io.to(logName).emit('error', err);
				}

				log.info("Handled results");

				db.getSystemState(payload, false, updateClient('event', io, logName));
			});
		});
	});
};

function updateClient(channel, io, logName) {
	return function(err, info) {
		if (err) {
				return io.to(logName).emit('error', err);
		}

		log.info("Updated client");

		if (!info) {
			return log.warn(`Info is empty about system state for ${payload}`);
		}

		io.to(logName).emit(channel, info);
	}
}
