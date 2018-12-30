
# OpenCL.NET

This is an OpenCL wrapper for .NET Core, which implements the .NET Standard Profile and therefore should run cross-platform on Linux, macOS, and Windows, as well as on many different .NET implementations, such as .NET Core, Mono, .NET Framework,
and Xamarin. The project aims at implementing the full OpenCL 2.1 standard on a low-level PInvoke level and create a wrapper, which exposes most of the functionality in a more CLR-style fashion. More abstract wrappers are planned for the future.
This wrapper should help with implementing image manipulation software as well as with machine learning (especially deep learning). Once the components have reached a certain maturity, they will be released to *__NuGet__*.

## Installation

In order to use OpenCL you have to install OpenCL. Under macOS you are in luck, because macOS has been shipping with OpenCL ever since Snow Leopard and you have to do nothing more, to get everything set up. Many standard Windows installations
also alread ship with OpenCL installed. If you have a custom setup, then installing the correct drivers should suffice. Under Linux things are a little more complicated. First of all you have to install the ICD loader. For example under Ubuntu
and Debian based systems you can do it like so:

```bash
sudo apt update
sudo apt install ocl-icd-opencl-dev
```

Then you have to install the correct drivers for your platform, e.g. the graphics adapter, and the OpenCL platform of choice. For example for Nvidia you have to install the CUDA framework.

*__More information on installation and setup will follow soon.__*

## Running the Sample

You can try the test application, that comes with this repository. It just performs a simple multiplication of a matrix with a vector. To build and run the sample application, you can do the following:

```bash
git clone https://github.com/lecode-official/opencl-dotnet.git
cd opencl-dotnet
cd OpenCl.DotNetCore
dotnet restore
cd ..
cd OpenCl.DotNetCore.Interop
dotnet restore
cd ..
cd OpenCl.DotNetCore.Tests
dotnet restore
dotnet build
dotnet run
```

## Troubleshooting

When your are experiencing a `DllNotFoundException` on macOS, then please make sure that you have the OpenCL framework in your library load path:

```bash
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/System/Library/Frameworks/OpenCL.framework/OpenCL
dotnet run
```

## Roadmap

In the following sections the roadmap to a stable 0.1.0-beta release is mapped out.

### General Todos

- [x] Separate into three projects: Native Core, CLR Wrapper, Test Program

### *__OpenCl.DotNetCore.Interop__* Project Todos

- [x] Rename the arguments of the native methods
- [x] Rename Info to Information
- [x] Put the different APIs of the native wrapper into different classes
- [x] Port the whole API surface area
    - [x] Command Queues API
    - [x] Contexts API
    - [x] Devices API
    - [x] Enqueued Commands API
    - [x] Events API
    - [x] Extensions API
    - [x] Kernels API
    - [x] Memory API
    - [x] Platforms API
    - [x] Profiling API
    - [x] Programs API
    - [x] Samplers API
    - [x] SVM Allocations API
- [x] Finish API documentation
    - [x] Command Queues API
    - [x] Contexts API
    - [x] Devices API
    - [x] Enqueued Commands API
    - [x] Events API
    - [x] Extensions API
    - [x] Kernels API
    - [x] Memory API
    - [x] Platforms API
    - [x] Profiling API
    - [x] Programs API
    - [x] Samplers API
    - [x] SVM Allocations API
- [x] Mark everything with the Obsolete attribute that have been deprecated in OpenCL
- [x] Mark everything with an attribute that contains the minimum version of OpenCL required

### *__OpenCl.DotNetCore__* Project Todos

- [x] `HandleBase` class, which is the base class for all OpenCL classes that need a handle
- [x] Add compiler log to exception, when program cannot be build
- [x] Add method to compile multiple sources at once
- [x] Add method to compile from file/files
- [x] Add method to compile from Stream/Streams
- [x] Create a class that converts `byte` arrays into CLR types
- [x] Implement the `equals` and the `==` operator, which compares the `Handle`
- [x] Split the different APIs in sub-namespaces
- [x] Make `MemoryObject` abstract and derive `Image`, `Pipe`, and `Buffer` from it
- [x] Add `async` methods for all native methods that have an `event`
- [x] Create an base class for event and then derive a user event from it, which is returned when calling CreateUserEvent (this should be done to ensure, that SetUserEventStatus can only be called on valid user events)
- [ ] Fix compilation and linking problems for Intel platform (maybe only on Windows)
- [ ] Make `WaitEvent` awaitable (maybe even rename it to `AwaitableEvent`)

### *__OpenCl.DotNetCore.Tests__* Project Todos

![Nothing to do here](http://img4.wikia.nocookie.net/__cb20120208030738/meme/es/images/thumb/8/8a/Nothing-to-do-here.jpg/170px-Nothing-to-do-here.jpg)
