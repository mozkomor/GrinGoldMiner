#!/bin/sh

mkdir bin
cd ..
cd GrinProMInerAPI
dotnet publish -c Release -r linux-x64 --self-contained --framework netcoreapp2.2 --output ../_build/bin
cd ..
cd CudaSolver
dotnet publish -c Release -r linux-x64 --self-contained --framework netcoreapp2.2 --output ../_build/bin
cd ..
cd OclSolver
dotnet publish -c Release -r linux-x64 --self-contained --framework netcoreapp2.2 --output ../_build/bin
