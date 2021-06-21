#include "test/CorrelatorTest/Options.h"

#include <cstdlib>
#include <iostream>

#include <unistd.h>


Options::Options(int argc, char *argv[])
:
  nrBits(8),
  nrChannels(480),
  nrReceivers(576),
  nrReceiversPerBlock(64),
  nrSamplesPerChannel(3072),
  nrTimesPerBlock(128 / nrBits),
  innerRepeatCount(1), outerRepeatCount(1),
  deviceNumber(0),
  verifyOutput(true)
{
  opterr = 0;

  for (int opt; (opt = getopt(argc, argv, "b:c:d:hn:N:r:R:t:V:")) >= 0;)
    switch (opt) {
      case 'b' : nrBits = atoi(optarg);
		 break;

      case 'c' : nrChannels = atoi(optarg);
		 break;

      case 'd' : deviceNumber = atoi(optarg);
		 break;

      case 'h' : std::cout << usage(argv[0]) << std::endl;
		 exit(0);

      case 'n' : nrReceivers = atoi(optarg);
		 break;

      case 'N' : nrReceiversPerBlock = atoi(optarg);
		 break;

      case 'r' : innerRepeatCount = atoi(optarg);
		 break;

      case 'R' : outerRepeatCount = atoi(optarg);
		 break;

      case 't' : nrSamplesPerChannel = atoi(optarg);
		 break;

      case 'V' : verifyOutput = atoi(optarg);
		 break;

      default  : throw Error(usage(argv[0]));
    }

  if (nrBits != 4 && nrBits != 8 && nrBits != 16)
    throw Error("nrBits must be 4, 8, or 16");

  if (nrChannels == 0)
    throw Error("nrChannels must be > 0");

  if (nrReceivers == 0)
    throw Error("nrReceivers must be > 0");

  if (nrReceiversPerBlock != 32 && nrReceiversPerBlock != 48 && nrReceiversPerBlock != 64)
    throw Error("nrReceiversPerBlock must be 32, 48, or 64");

  if (nrSamplesPerChannel == 0)
    throw Error("nrSamplesPerChannel must be > 0");

  nrTimesPerBlock = 128 / nrBits;

  if (nrSamplesPerChannel % nrTimesPerBlock != 0)
    throw Error("nrSamplesPerChannel must be a multiple of " + std::to_string(nrTimesPerBlock));
}


std::string Options::usage(const std::string &execName)
{
  return "usage: " + execName + " [-b nrBits] [-c nrChannels] [-n nrReceivers] [-N nrReceiversPerBlock] [-r innerRepeatCount] [-R outerRepeatCount] [-t nrSamplesPerChannel] [-V verifyOutput]";
}


const char *Options::Error::what() const noexcept
{
  return msg.c_str();
}
