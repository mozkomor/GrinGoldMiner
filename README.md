# GPU Miner for Cuckoo Cycle PoW
This miner currently runs on windows and linux and exist only as a proof of concept for bounty claim. It does not actually interface with grin software yet (see below). It only works on nvidia GPUs (for now, see below) with at least 8GB of RAM. 

50% bounty here - btc (Photon): 3MRRCuFmS3GciugRRawkuAbLLcpNVXSJqm

50% bounty here - btc (Urza): 3AcRFwqKx6P8rBngUz4hRTG2QEwodLqAVE

Contact us at mozkomorkaban@gmail.com

Based on great work of John Tromp, the inventor of Cuckoo cycle PoW  https://github.com/tromp/cuckoo

And designed for https://github.com/mimblewimble/grin

## How to compile on Windows
You need Visual Studio 2017 with dotnet core 2.0 and CUDA 9.1 SDK installed on your system. Simply open the solution and build Release target for x64 architecture. Grab both executables and their libraries and put them in a single folder somewhere else. Open windows power shell in that folder and run ./Theta -r 100 -n 0

## How to compile on Linux
You need latest nvidia drivers, CUDA SDK 9.1 and g++ compiler it is happy with. You can execute my cuda Makefile I copied from cuda samples. This will produce Cudacka.exe file (same name as in windows as this is executed from dotnet master process).

Go into Theta miner master process manager directory. Install dotnet core 2.0 and run

    dotnet publish --self-contained --runtime ubuntu-x64 --configuration "Release"
    
but first replace “ubuntu-x64” with your system RID from this catalog https://docs.microsoft.com/en-us/dotnet/core/rid-catalog

This will build the app with all libraries included. Copy Theta and Cudacka and their libraries to a same folder. Create “edges” directory there.
Now create a 32MB ramdisk of the “edges” subdirectory if you want. I use regular files for communication in linux so you don’t want it to be writing it to mechanical disk. If you don’t have SSD you should do this even for testing. This is temporary solution until better communication system is made for linux.

If you don’t want to install dotnet core package, just use Mono – it is much slower, but it should work too.

Run it

    ./Theta -r 100 -n 0
    
This launches 100 iterations and at the end prints total time that you simply divide by 100 to get single graph time.

## Questions

**Q:** Can I use this to mine grin?

**A:** Not at this moment. We would like to create a TCP/IP bridge between the miner and grin node so both are separate. Also possibly add support for local network pools or something similar that would keep complexity on grin node side at minimum. 

**Q:** Can it use multiple GPUs?

**A:** Yes in theory, but this is not enabled yet. Cycle detection is on CPU side so it may not be possible to run 6 GPUs on el cheapo Celeron as it is with Ethereum or Monero. 

**Q:** Does it run on AMD Vega?

**A:** OpenCL version is planned next as time allows.

**Q:** Will it get any faster?

**A:** For GPUs with GDDR5X (GTX 1080+Ti ) and GDDR6 (future) and large number of compute units at the same time, special modifications can be, in theory, made to utilize those cards better. Other GPUs may see additional boost of 20-30% as well. This would significantly increase complexity of the miner so it will come later.

**Q:** Can you explain the license?

**A:** This work is not derived from original John Tromp solvers. However, we still request that portion of the fee goes to main grin developers no matter what. Our own releases (source code defaults and binary) will use 1% fee for grin developers and 1% fee for miner developers (us). Any further derived work must honour 2:1:1 fee distribution. So for example 1% to the derived work developer, 0.5% to grin developers and 0.5% to original miner developers (us). If the mined coin is Monero, 0.1% always goes to its Lead Maintainer's lifetime gym membership.

**Q:** What GPU will be the best for power efficiency or performace per dollar?

**A:** This miner is not bottlenecked by memory like more usual eth and other miners, I only have GTX 1070, so hard to draw any conclusions - the code is currently only optimized for GTX 1070. Do your own benchmarks or send us new GPUs :)


