md bin
cd ..
cd GrinGoldMiner3
dotnet publish -c Release -r win-x64 --self-contained --framework netcoreapp2.2 --output ..\_build\bin
cd ..
cd CudaSolver
dotnet publish -c Release -r win-x64 --self-contained --framework netcoreapp2.2 --output ..\_build\bin
cd ..
cd OclSolver
dotnet publish -c Release -r win-x64 --self-contained --framework netcoreapp2.2 --output ..\_build\bin

pause
