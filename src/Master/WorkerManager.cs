using SharedSerialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace Mozkomor.GrinGoldMiner
{
    public class WorkerManager
    {
        static List<Worker> workers = new List<Worker>();
        public static DateTime lastJob = DateTime.Now;

        public static readonly bool IsLinux = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
        public static List<GpuDevice> GetDevices(WorkerType type)
        {
            try
            {
                //if (type == WorkerType.NVIDIA) return new List<GpuDevice>();

                TcpListener l = new TcpListener(IPAddress.Parse("127.0.0.1"), 13500);
                l.Start();
                var process = Process.Start(new ProcessStartInfo()
                {
                    FileName = (IsLinux ?
                          ((type == WorkerType.NVIDIA) ? "CudaSolver" : "OclSolver")
                        : ((type == WorkerType.NVIDIA) ? "CudaSolver.exe" : "OclSolver.exe")),
                    Arguments = string.Format("-1 13500"),
                    CreateNoWindow = true,
                    UseShellExecute = false
                });
                var client = l.AcceptTcpClient();
                NetworkStream stream = client.GetStream();
                l.Stop();
                object devices = (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Deserialize(stream);
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
                //Console.WriteLine(ex.ToString());
                Logger.Log(LogLevel.ERROR, "Failed to enumerate devices: " + ex.Message);
                return new List<GpuDevice>();
            }
        }

        public static List<WorkerInfo> GetWorkersInfo()
        {
            List<WorkerInfo> ws = new List<WorkerInfo>();
            foreach (var w in workers)
            {
                ws.Add(new WorkerInfo()
                {
                    GPUStatus = w.GetStatus().ToString(),
                    GraphsPerSecond = w.GetStatus() == GPUStatus.ONLINE ? w.currentGPS : 0,
                    ID = w.ID,
                    lastSolution = w.lastSolTime,
                    TotalSols = w.totalSols,
                    Errors = w.errors,
                    LastLog = w.lastLog,
                    LastDebugLog = w.lastDebugLog,
                    LastErrLog = w.lastErrLog,
                    GPUName = w.gpu.GPUName,
                    GPUOption = w.gpu,
                    Fidelity = w.fidelity
                });
            }
            return ws;
        }

        public static void Init(Config config)
        {
            if (config == null) return;

            int i = 0;
            foreach(var gpu in config.GPUOptions)
            {
                Console.WriteLine($"Init started for gpu {i} - {gpu.GPUName}");
                Worker w = new Worker(gpu, i++);
                workers.Add(w);
                bool initOk = w.Start(config);
                string result = initOk ? "SUCCESS" : "FAILURE";
                Task.Delay(4000).Wait();
                Console.WriteLine($"Init completed for gpu {i-1} - {gpu.GPUName} with {result}");
            }
        }

        public static void PrintWorkerInfo()
        {
            try
            {
                Console.WriteLine("--------------------------------------------------------------------------------");
                Console.Write("Shares (sub/acc/stale/rej): ");
                var conn = ConnectionManager.GetCurrConn();
                if (conn != null)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.Write($"{conn.totalShares}"); Console.ResetColor(); Console.Write("/");
                    if (conn.sharesAccepted > 0) Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write($"{conn.sharesAccepted}"); Console.ResetColor(); Console.Write("/");
                    if (conn.sharesTooLate > 0) Console.ForegroundColor = ConsoleColor.Cyan;
                    Console.Write($"{conn.sharesTooLate}"); Console.ResetColor(); Console.Write("/");
                    if (conn.sharesRejected > 0) Console.ForegroundColor = ConsoleColor.Red;
                    Console.Write($"{conn.sharesRejected}"); Console.ResetColor();
                    Console.ResetColor();
                    Console.WriteLine($"     Last share:   {(DateTime.Now - conn.lastShare).TotalSeconds:F0} seconds");
                }

                Console.WriteLine("--------------------------------------------------------------------------------");
                
                PrintFidelity();
                foreach(var w in workers)
                {
                    w.PrintStatusLinesToConsole();
                }
                Console.WriteLine("--------------------------------------------------------------------------------");
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"                                              Total : {workers.Sum(w => w.currentGPS):F2} gps");
                Console.ResetColor();
                Console.WriteLine("--------------------------------------------------------------------------------");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to print GPU info to console: " + ex.Message);
            }
        }

        private static void PrintFidelity()
        {
            double sum = 0;
            int count = 0;
            foreach(var w in workers)
            {
                if (w.fidelity != 0 && w.totalSols > 1000)
                {
                    sum += w.fidelity;
                    count++;
                }
            }
            if (count == 0) { return; }

            Console.Write($"{DateTime.Now.ToString("HH:mm:ss"),-10} Global solvers fidelity: ");
            double diff = Math.Abs(2.0 - (sum / count));
            if (diff < 0.05)
                Console.ForegroundColor = ConsoleColor.Green;
            else if (diff < 0.1)
                Console.ForegroundColor = ConsoleColor.Yellow;
            else
                Console.ForegroundColor = ConsoleColor.Red;

            Console.Write((sum / count).ToString("0.000"));
            Console.ResetColor();
            Console.WriteLine();
            Console.WriteLine();
        }

        //worker found solution
        public static void SubmitSolution(SharedSerialization.Solution sol)
        {
            //todo wrap Solution into richer class with internal info
            // diff check !!
            ConnectionManager.SubmitSol(sol);
        }

        private static string lockPush = "";
        //new job received from stratum connection
        public static void newJobReceived(SharedSerialization.Job job)
        {
            lock (lockPush)
            {
                lastJob = DateTime.Now;
                //update workers..
                foreach (var worker in workers)
                {
                    worker.SendJob(job);
                }
            }
        }

        //tell workers to chill
        internal static void PauseAllWorkers()
        {
            foreach (var worker in workers)
            {
                worker.SendJob(new SharedSerialization.Job()
                    {
                    pre_pow = "",
                    jobID = 666,
                    timestamp = DateTime.Now
                });
            }
        }

        public static void PrintStatusLinesToConsole(WorkerInfo w)
        {
            try
            {
                Console.Write($"GPU {w.ID,-3}: {w.GPUName,-20} ");
                switch (w.GPUStatus)
                {
                    case "STARTING":
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        Console.Write($"{"STARTING",-10}");
                        break;
                    case "ONLINE":
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.Write($"{"ONLINE",-10}");
                        break;
                    case "OFFLINE":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.Write($"{"OFFLINE",-10}");
                        break;
                    case "DISABLED":
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.Write($"{"DISABLED",-10}");
                        break;
                    case "ERROR":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.Write($"{"ERROR",-10}");
                        break;
                    case "NO MEMORY":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.Write($"{"NO MEMORY",-12}");
                        break;
                }
                Console.ResetColor();
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"   Mining at {w.GraphsPerSecond,5:F2} gps");
                Console.ResetColor();
                Console.WriteLine($"  Solutions: {w.TotalSols}");
                Console.ForegroundColor = IsLinux ? ConsoleColor.Gray : ConsoleColor.DarkGray;
                Console.WriteLine($"       Last Message: {w.LastLog.ToShortString()}"); Console.ResetColor();
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"ERROR GPU {w.ID}, MSG: {ex.Message}");
                Console.ResetColor();
            }
        }

        public static void PrintStatusLinesToLog(WorkerInfo w)
        {
            try
            {
                if (w.GPUStatus == "ONLINE")
                {
                    Logger.Log(LogLevel.INFO, $"Statistics: GPU {w.ID}: mining at {w.GraphsPerSecond:F2} gps, solutions: {w.TotalSols}");
                }
                if (w.GPUStatus == "ERROR" && w.LastErrLog != null && w.LastErrLog.message != null)
                {
                    Logger.Log(LogLevel.INFO, $"Error: GPU {w.ID}: message: {w.LastErrLog.message}");
                }
            }
            catch (Exception ex)
            {

            }
        }
    }
}
