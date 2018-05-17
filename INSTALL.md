## How to compile on Windows
You need Visual Studio 2017 with dotnet core 2.0 and CUDA 9.1 SDK installed on your system. Simply open the solution and build Release target for x64 architecture (or better publish). Grab both executables and their libraries and put them in a single folder somewhere else. Open windows power shell in that folder and run

    ./Theta -d <cuda_device_id> -a <grin_node_IP>
or

    dotnet Theta.dll -d <cuda_device_id> -a <grin_node_IP>

CUDA 9.1 needs specific maximum version of VS2017, it may not work with the latest VC++ compiler version!

## How to compile on Linux
You need latest nvidia drivers, CUDA SDK 9.1 and g++ compiler it is happy with. You can execute my cuda Makefile I copied from cuda samples. This will produce Cudacka.exe file (same name as in windows as this is executed from dotnet master process).

Go into Theta miner master process manager directory. Install dotnet core 2.0 and run

    dotnet publish --self-contained --runtime ubuntu-x64 --configuration "Release"
    
but first replace “ubuntu-x64” with your system RID from this catalog https://docs.microsoft.com/en-us/dotnet/core/rid-catalog

This will build the app with all libraries included. Copy Theta and Cudacka and their libraries to a same folder. Create “edges” directory there.
Now create a 32MB ramdisk of the “edges” subdirectory if you want. I use regular files for communication in linux so you don’t want it to be writing it to mechanical disk. If you don’t have SSD you should do this even for testing. This is temporary solution until better communication system is made for linux.

If you don’t want to install dotnet core package, just use Mono – it is much slower, but it should work too.

Run it

    ./Theta -d <cuda_device_id> -a <grin_node_IP>
or

    dotnet Theta.dll -d <cuda_device_id> -a <grin_node_IP>
    
This launches 100 iterations and at the end prints total time that you simply divide by 100 to get single graph time.
