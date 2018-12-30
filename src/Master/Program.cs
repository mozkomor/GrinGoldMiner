// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace Mozkomor.GrinGoldMiner
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.CancelKeyPress += delegate {
                Console.WriteLine("Ctrl+C - Exitting");
                Close();
            };

            var dir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var configPath = Path.Combine(dir, "config.xml");
            Config config = new Config();
            if (File.Exists(configPath))
            {
                config = Serialization.DeSerialize<Config>(configPath);
            }
            else
            {
                Console.WriteLine("ERROR: missing config.xml");
                Serialization.Serialize<Config>(Config.GetEmptyConfig(), configPath);
                Console.ReadLine();
            }

            ConnectionManager.Init(config);


            string prevprepow = "";
            while (true)
            {

                var job = ConnectionManager.GetJob();
                if (job != null)
                {
                    if (job.pre_pow != prevprepow)
                    {
                        ConnectionManager.SubmitSol(
                                new Solution()
                                {
                                    jobId = job.job_id,
                                    difficulty = job.difficulty,
                                    height = job.height,
                                    k0 = 111, k1 = 111, k2 = 111, k3 = 111,
                                    nonce = 123456, nonces = null,
                                    prepow = job.pre_pow,
                                });

                        prevprepow = job.pre_pow;
                    }
                    Task.Delay(2500).Wait();
                }
                else
                {
                    Console.Write(".");
                    Task.Delay(500).Wait();
                }

                //Theta.ConnectionManager.SubmitJob();
            }
            Console.WriteLine();

        }

        public static void Close()
        {
            ConnectionManager.CloseAll();
            Environment.Exit(0);
        }
    }

    public class Solution
    {
        public UInt64 k0, k1, k2, k3, nonce, height, difficulty, jobId;
        public List<Tuple<uint, uint>> nonces = new List<Tuple<uint, uint>>();

        public string prepow = "";

        public override string ToString()
        {
            return $"jobId: {jobId},k0 {k0}, k1 {k1}, k2 {k2}, k3 {k3}, nonce {nonce}, height {height}, difficulty {difficulty}";
        }
    }
}
