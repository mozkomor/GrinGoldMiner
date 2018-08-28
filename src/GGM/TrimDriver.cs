// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Kukacka optimized miner is covered by the FAIR MINING license


using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

namespace GGM
{
    public enum TrimmerState
    {
        Starting,
        Ready,
        Trimming,
        SendingEdges,
        Terminated,
        Solving,
        Error
    }

    class TrimDriver
    {
        public static bool DEBUG = true;
        private GPU gpu;
        private Process ocl;
        private TcpClient bridge;
        private NetworkStream stream;
        private StreamReader reader;
        private object listener;
        public TrimmerState Status = TrimmerState.Starting;
        public volatile bool Calcelled = false;
        static Random rnd = new Random((int)DateTime.Now.Ticks);

        private string statusMessage;
        public string StatusMessage
        {
            set
            {
                if (DEBUG)
                    Console.WriteLine("GPU " + gpu.GPUID.ToString() + ": " + value);

                statusMessage = value;
            }
            get
            {
                return statusMessage;
            }
        }

        bool NetworkOk = false;
        bool DeviceOk = false;
        bool DeviceReady = false;

        ~TrimDriver()
        {
            Calcelled = true;
            if (bridge != null && bridge.Connected)
            {
                bridge.Close();
            }
            if (ocl != null && !ocl.HasExited)
                ocl.Kill();
        }

        public TrimDriver(GPU gpu)
        {
            this.gpu = gpu;
            StatusMessage = "Idle";

            ocl = Process.Start(new ProcessStartInfo()
            {
                FileName = "OCLacka.exe",                // platform devID mode(0) port(13430)
                Arguments = string.Format("{0} {1} {2} {3}", gpu.Type == GPUtype.CUDA ? "NVIDIA" : "AMD", gpu.DeviceID, 0, gpu.GPUID+13430),
                CreateNoWindow = true,
                RedirectStandardError = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                StandardErrorEncoding = Encoding.ASCII,
                StandardOutputEncoding = Encoding.ASCII,
                UseShellExecute = false
            });

            ocl.ErrorDataReceived += (sender,e) => { LogError(e.Data, gpu); };
            ocl.OutputDataReceived += (sender, e) => { LogStd(e.Data, gpu); };
            ocl.BeginOutputReadLine();
            ocl.BeginErrorReadLine();

            Task.Delay(3000).Wait();

            if (ocl.HasExited)
            {
                Status = TrimmerState.Error;
                StatusMessage = "Instacrash";
                return;
            }

            try
            {
                bridge = new TcpClient("127.0.0.1", gpu.GPUID + 13430);
                stream = bridge.GetStream();
                reader = new StreamReader(stream);
                listener = Task.Factory.StartNew(() => { Listen(); });
            }
            catch
            {
                Status = TrimmerState.Error;
                StatusMessage = "TCP Con Failed";
                return;
            }

            Task.Delay(1000).Wait();

            if (bridge != null && bridge.Connected)
            {
                if (Status == TrimmerState.Ready && !ocl.HasExited)
                {
                    Task.Factory.StartNew(() => { TrimmingLoop(); }, TaskCreationOptions.LongRunning);
                }
                else
                {
                    Status = TrimmerState.Error;
                    StatusMessage = ocl.HasExited ?  "Trimmer Exit" : "Trimmer !Ready";
                }
            }
            else
            {
                Status = TrimmerState.Error;
                StatusMessage = "TCP Con Discn";
            }
        }

        private void TrimmingLoop()
        {
            while (!Calcelled)
            {
                GrinConeeect conn = GetCurrentConn();

                if (conn.lastComm.AddMinutes(30) < DateTime.Now)
                    conn.WaitForJob = true;

                if (conn.WaitForJob | !conn.IsConnected)
                {
                    Task.Delay(100).Wait();
                    Console.Write(".");
                    continue;
                }

                JobTemplate job = conn.CurrentJob;

                UInt64 hnonce = (UInt64)(long)rnd.Next() | ((UInt64)(long)rnd.Next() << 32);
                var bytes = BitConverter.GetBytes(hnonce).Reverse().ToArray();
                header = header.Concat(bytes).ToArray();
                var hash = new Crypto.Blake2B(256);
                byte[] blaked = hash.ComputeHash(header);
                //blaked = hash.ComputeHash(blaked); -- testnet2 bug

                k0 = BitConverter.ToUInt64(blaked, 0);
                k1 = BitConverter.ToUInt64(blaked, 8);
                k2 = BitConverter.ToUInt64(blaked, 16);
                k3 = BitConverter.ToUInt64(blaked, 24);

            }
        }

        private GrinConeeect GetCurrentConn()
        {
            return Program.gc;
        }

        private void Listen()
        {
            while (bridge.Connected)
            {
                try
                {
                    string message = reader.ReadLine();

                    if (message.StartsWith("#"))
                    {
                        switch (message[1])
                        {
                            case 'N':
                                Status = TrimmerState.Starting;
                                StatusMessage = "TRM NET OK";
                                break;
                            case 'A':
                                Status = TrimmerState.Trimming;
                                StatusMessage = "TRIMMING";
                                break;
                            case 'D':
                                {
                                    string[] data = message.Split(';');
                                    if (data.Length == 3)
                                    {
                                        gpu.DeviceName = data[1];
                                        gpu.DeviceMemory = long.Parse(data[2]);
                                        Logger.Log(LogType.Info, "Detected " + gpu.DeviceName + " ID:" + gpu.DeviceID);
                                    }
                                }
                                break;
                            case 'R':
                                Status = TrimmerState.Ready;
                                StatusMessage = "TRM READY";
                                break;
                            case 'E':
                                Status = TrimmerState.SendingEdges;
                                StatusMessage = "TRM EDGES OUT";
                                break;
                        }
                    }
                    else
                    {
                        Logger.Log(LogType.Error, "Unknown message from trimmer " + gpu.GPUID + ": " + message);
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log(LogType.Error, "Trimmer message read fail " + gpu.GPUID, ex);
                    Task.Delay(500);
                }
            }
        }

        private void LogStd(string data, GPU gpu)
        {
            Console.WriteLine(data);
        }

        private void LogError(string data, GPU gpu)
        {
            Console.WriteLine(data);
        }

        internal void Terminate()
        {
            Calcelled = true;

            if (bridge != null && bridge.Connected)
            {
                bridge.Close();
            }
            if (ocl != null && !ocl.HasExited)
                ocl.Kill();
        }
    }
}
