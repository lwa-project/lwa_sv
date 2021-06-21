#if !defined OPTIONS_H
#define OPTIONS_H

#include <exception>
#include <string>


class Options
{
  public:
    class Error : public std::exception {
      public:
	Error(const std::string &msg)
	:
	  msg(msg)
	{
	}

	virtual const char *what() const noexcept;

      private:
	std::string msg;
    };

    Options(int argc, char *argv[]);

    unsigned nrBaselines() const { return nrReceivers * (nrReceivers + 1) / 2; }

    unsigned nrBits;
    unsigned nrChannels;
    unsigned nrReceivers;
    unsigned nrReceiversPerBlock;
    unsigned nrSamplesPerChannel;
    unsigned nrTimesPerBlock;
    unsigned innerRepeatCount, outerRepeatCount;
    unsigned deviceNumber;
    bool     verifyOutput;

    static const unsigned nrPolarizations = 2;

  private:
    static std::string usage(const std::string &execName);
};

#endif
