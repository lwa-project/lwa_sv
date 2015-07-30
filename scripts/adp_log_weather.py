#!/usr/bin/env python

from AdpLogging import AdpFileLogger
import AdpConfig
import weather
from urllib2 import HTTPError

class AdpWeatherLogger(AdpFileLogger):
	def __init__(self, config, filename, nattempts=3):
		fileheader = ['#'+'\t'.join(['TEMP','DEWPT','WNDDIR','WNDSPD','TEC']),
		              '#'+'\t'.join(['DEG_C','DEG_C','DEG',  'KT',    'TECU'])]
		super(AdpWeatherLogger, self).__init__(config, filename, fileheader)
		self.station   = config['telescope']['weather_station']
		self.lat       = config['telescope']['lat']
		self.lon       = config['telescope']['lon']
		self.nattempts = nattempts
	def get_ground_weather(self):
		gnd = None
		for attempt in xrange(self.nattempts):
			try:
				gnd = weather.GroundWeather(self.station)
			except HTTPError:
				continue
			else:
				break
		return gnd
	def get_ionosphere_weather(self):
		tec = None
		for attempt in xrange(self.nattempts):
			try:
				tec = weather.TotalElectronContent()
			except HTTPError:
				continue
			else:
				break
		return tec
	def update(self):
		gnd = self.get_ground_weather()
		ion = self.get_ionosphere_weather()
		logstr = '\t'.join(['%.2f' % gnd.temp_c,
		                    '%.2f' % gnd.dewpoint_c,
		                    #'%.1f' % gnd.pressure_mb,
		                    '%.1f' % gnd.wind_dir_degs,
		                    '%.1f' % gnd.wind_speed_kt,
		                    '%.1f' % ion.tecmap(self.lon,self.lat)])
		self.log(logstr)

if __name__ == "__main__":
	import sys
	if len(sys.argv) <= 1:
		print "Usage:", sys.argv[0], "config_file"
		sys.exit(-1)
	config_filename = sys.argv[1]
	config = AdpConfig.parse_config_file(config_filename)
	
	filename = config['log']['files']['weather']
	logger = AdpWeatherLogger(config, filename)
	logger.update()
	sys.exit(0)
