
# ADP Control Documentation

## Terminology

- `headnode`: The ADP cluster head machine, which is named `adp` (aka
`adp0`). This machine runs the `adp-control` service and interfaces
between the cluster and the outside (of ADP) world.

- `roaches`: The ADP cluster FPGA boards, of which there are 16, named
  'roach1-16'. These boards run the ADP `F-engine` firmware to
  frequency-channelise data from the ADCs and send it through the data
  switch to the servers.

- `corner-turn`: The implicit data transposition performed as
  packetised data travel from the roaches, through the data switch and
  into the servers. Each roach sends all frequency channels for a
  subset of inputs, and each server receives all inputs for a subset
  of frequency channels.

- `servers`: The ADP cluster processing machines, of which there are
six, named `adp1-6`. These machines run the `adp-pipeline` service
(aka the `X-engine`), which captures and processes data streams from
the roaches.

- `streams`: Individual streams of (frequency-domain) data being
processed, of which there are two per server for a total of 12
streams. Each stream is processed independently until they are
merged together in a T-engine.

- `T-engine`: A data processing pipeline run on one or more servers
  that merges and converts frequency-domain data from multiple sources
  into time-domain output streams.

## Headnode and server configuration

All ADP software executables on the headnode and servers are run as
system services using the 'upstart' approach
(http://upstart.ubuntu.com/cookbook/). Services can be started,
stopped and restarted via the corresponding commands:

    $ start   <service-name>
	$ stop    <service-name>
	$ restart <service-name>

The ADP services will start automatically when the system boots, and
will restart automatically if they fail or are killed unexpectedly.

## Intra-cluster communication

Communication between the control script (on the headnode) and the
individual pipelines on the servers is done via ZeroMQ sockets with
messages consisting of a json header optionally followed by a binary
blob (as the second part of a multi-part message).  E.g., FST and BAM
updates are sent via REQ sockets to REP sockets in the corresponding
task in each pipeline.

## FST and BAM commands

The antenna FIR coefficients and beam delays/gains specified by FST
and BAM commands are translated by the `adp-control` service into
frequency-domain complex weights before being sent to the X-engine
pipelines for application to the live data streams. The full band (all
4096 channels) is sent to every pipeline to allow each to extract the
channels it needs, rather than trying to keep track of which channels
are being processed where from the headnode.

Complex weights are sent in favour of the raw FIR coefficients or beam
delays/gains so that complicated equations (e.g., pseudo-inversion of
the beam weights matrix, RFI nulling etc.) can be implemented in
Python and added freely in future.

## SHT commands

Clean shutdown of ADP services is handled via system signals (e.g.,
SIGTERM), which are caught by the main thread to perform a clean exit.

Power management of the servers is handled via IPMI commands from the
headnode. This allows both soft and hard shutdowns/restarts, as
required to implement both regular and `SCRAM` shutdowns.


ADP services
  Headnode
    adp-control (/etc/init/adp-control.conf)
  Servers
    adp-server-monitor (/etc/init/adp-server-monitor.conf)
Cron jobs
  Headnode
    adp_log_weather.py
    adp_log_down_hosts.py
  Servers
    adp_log_device_temps.py
    adp_log_disk_usage.py
