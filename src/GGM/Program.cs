// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Kukacka optimized miner is covered by the FAIR MINING license


using System;
using Terminal.Gui;
using Mono.Terminal;
using System.Threading.Tasks;

namespace GGM
{
    class Program
    {
        public static GrinConeeect gc;
        public static GrinConeeect gc2;
        public static GrinConeeect gc3;

        static void Main(string[] args)
        {
            Console.CancelKeyPress += delegate {
                Console.WriteLine("Ctrl+C - Exitting");

                if (gc != null)
                    gc.GrinClose();
                if (gc2 != null)
                    gc2.GrinClose();
                if (gc3 != null)
                    gc3.GrinClose();

                if (Config.GGMC != null && Config.GGMC.GPUs != null)
                    foreach (var gpu in Config.GGMC.GPUs)
                        gpu.Terminate();

                Environment.Exit(0);
            };

            System.Console.InputEncoding = System.Text.Encoding.ASCII;

            if (Config.LoadConfig(args))
            {
                RunMiner();
            }
        }

        private static void RunMiner()
        {
            try
            {
               Task.Run(() =>
               {
                   gc = new GrinConeeect(Config.GGMC);

                   if (!gc.IsConnected)
                   {
                       Logger.PushError("Unable to connect to node.");
                       return;
                   }

                   Parallel.ForEach(Config.GGMC.GPUs, (gpu) =>
                   {
                       gpu.Driver = new TrimDriver(gpu);
                   });
               });

                UI.StartUI();
            }
            catch (Exception ex)
            {
                Logger.Log(LogType.FatalError, "Fatal error.", ex);
            }

        }
    }
}
