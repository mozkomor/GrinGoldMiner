using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SharedSerialization;

namespace CudaSolver
{
    public class Comms
    {
        public static ConcurrentQueue<Solution> graphSolutionsOut = new ConcurrentQueue<Solution>();
        public static ConcurrentQueue<LogMessage> logsOut = new ConcurrentQueue<LogMessage>();
        public static GpuDevicesMessage gpuMsg = null;

        public static AutoResetEvent flushToMaster;
        private static TcpClient client;
        private static NetworkStream stream;
        private static Task listener;
        private static Task sender;
        private static int _port;
        public static Job nextJob = new Job();
        public static volatile bool IsTerminated = false;
        public static DateTime lastIncoming = DateTime.Now;

        public static volatile int cycleFinderTargetOverride = 0;
        public static volatile int numberOfGPUs = 0;

        static int errorCounter = 0;

        internal static void ConnectToMaster(int port)
        {
            _port = port;
            flushToMaster = new AutoResetEvent(false);
            client = new TcpClient("127.0.0.1", port);
            //client = new TcpClient("10.0.0.122", port);
            if (client.Connected)
            {
                stream = client.GetStream();
                listener = Task.Factory.StartNew(() => { Listen(); }, TaskCreationOptions.LongRunning);
                sender = Task.Factory.StartNew(() => { WaitSend(); }, TaskCreationOptions.LongRunning);
            }
        }

        private static void WaitSend()
        {
            while (!IsTerminated)
            {
                try
                {
                    flushToMaster.WaitOne();

                    Solution s;
                    while (graphSolutionsOut.TryDequeue(out s))
                    {
                        (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Serialize(stream, s);
                    }
                    if (gpuMsg != null)
                    {
                        //Console.WriteLine("sending now");
                        (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Serialize(stream, gpuMsg);
                        gpuMsg = null;
                        //Console.WriteLine("flushing now");
                        stream.Flush();
                    }
                    LogMessage lm;
                    while (logsOut.TryDequeue(out lm))
                    {
                         (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Serialize(stream, lm);
                    }
                }
                catch (Exception ex)
                {
                    // log to local file and console
                    //Logger.Log(LogLevel.Warning, "WaitSend error", ex);
                    //Console.WriteLine("exc " + ex.InnerException.Message);
                }
            }
        }

        private static void Listen()
        {
            while (!IsTerminated)
            {
                try
                {
                    object payload = (new BinaryFormatter()).Deserialize(stream);

                    switch (payload)
                    {
                        case Job job:
                            nextJob = job;
                            lastIncoming = DateTime.Now;
                            Logger.Log(LogLevel.Debug, $"New job received: {job.pre_pow}");
                            break;
                        case GpuSettings settings:
                            cycleFinderTargetOverride = settings.targetGraphTimeOverride;
                            numberOfGPUs = settings.numberOfGPUs;
                            break;
                    }

                    errorCounter = 0;
                }
                catch (Exception ex)
                {
                    //Logger.Log(LogLevel.Warning, "Listen error! ", ex);
                    Console.WriteLine("Connection lost...");

                    Task.Delay(5);
                    try
                    {
                        while (stream.DataAvailable)
                            stream.ReadByte();
                    }
                    catch { }

                    if (errorCounter++ > 5)
                    {
                        Close();
                    }
                }
            }
        }

        internal static bool IsConnected()
        {
            return client != null && client.Connected;
        }

        internal static void SetEvent()
        {
            if (flushToMaster != null)
                flushToMaster.Set();
        }

        internal static void Close()
        {
            try
            {
                if (stream != null)
                    stream.Flush();
                Task.Delay(500).Wait();
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
