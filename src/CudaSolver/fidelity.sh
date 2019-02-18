#!/bin/sh
dotnet publish -c Release -r linux-x64
cd bin/Release/netcoreapp2.2/linux-x64/publish
./CudaSolver fidelity:0:0:1000