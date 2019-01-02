﻿using SharedSerialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Mozkomor.GrinGoldMiner
{
    public class Worker
    {
        private const bool DEBUG = true;

        private static Process worker;
        private static TcpClient client;
        private static NetworkStream stream;
        private static Task listener;
        private static AutoResetEvent flushEvent;

        private WorkerType type;
        private int workerDeviceID;
        private int workerCommPort;

        private DateTime workerStartTime = DateTime.Now;
        private DateTime workerLastMessage = DateTime.Now;
        private long workerTotalSolutions = 0;
        private long workerTotalLogs = 0;
        private volatile bool IsTerminated;

        public Worker(WorkerType gpuType, int gpuID)
        {
            type = gpuType;
            workerDeviceID = gpuID;
            workerCommPort = 13500 + (int)gpuType + gpuID;
        }

        public List<GpuDevice> GetDevices()
        {
            try
            {
                TcpListener l = new TcpListener(IPAddress.Parse("127.0.0.1"), 13500);
                l.Start();
                var process = Process.Start(new ProcessStartInfo()
                {
                    FileName = (type == WorkerType.NVIDIA) ? "solvers\\CudaSolver.exe" : "solvers\\OclSolver.exe",
                    Arguments = string.Format("-1 13500"),
                    CreateNoWindow = true,
                    UseShellExecute = false
                });
                var client = l.AcceptTcpClient();
                NetworkStream stream = client.GetStream();
                l.Stop();
                object devices = (new BinaryFormatter()).Deserialize(stream);
                try
                {
                    stream.Close();
                    client.Close();
                    process.Kill();
                }
                catch { }
                if (devices is GpuDevicesMessage)
                    return (devices as GpuDevicesMessage).devices;
                else
                    return new List<GpuDevice>(); // and log ?!
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.ERROR, "Failed to enumerate devices: " + ex.Message);
                return new List<GpuDevice>();
            }
        }

        //send job to worker
        public bool SendJob(SharedSerialization.Job job)
        {
            try
            {
                (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Serialize(stream, job);
                return true;
            }
            catch (Exception ex)
            {
                Logger.Log(ex);
                return false;
            }

        }

        public bool Start()
        {
            try
            {
                TcpListener l = new TcpListener(IPAddress.Parse("127.0.0.1"), workerCommPort);
                l.Start();
                worker = Process.Start(new ProcessStartInfo()
                {
                    FileName = (type == WorkerType.NVIDIA) ? "solvers\\CudaSolver.exe" : "solvers\\OclSolver.exe",
                    Arguments = string.Format("{0} {1}", workerDeviceID, workerCommPort),
                    CreateNoWindow = !DEBUG,
                    UseShellExecute = false
                });
                client = l.AcceptTcpClient();
                l.Stop();
                stream = client.GetStream();
                listener = Task.Factory.StartNew(() => { Listen(); }, TaskCreationOptions.LongRunning);
                return true;
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.ERROR, "Failed to start worker process: " + ex.Message);
                return false;
            }
        }

        //receive solution from worker
        private void Listen()
        {
            while (!IsTerminated)
            {
                try
                {
                    object payload = (new BinaryFormatter()).Deserialize(stream);

                    switch (payload)
                    {
                        case SharedSerialization.Solution sol:
                            WorkerManager.SubmitSolution(sol);
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log(LogLevel.ERROR, "Listen error" + ex.Message);
                }
            }
        }

        public void Close()
        {
            try
            {
                IsTerminated = true;
                if (stream != null)
                    stream.Close();
                if (client != null)
                    client.Close();
            }
            catch { }
        }
    }


}
