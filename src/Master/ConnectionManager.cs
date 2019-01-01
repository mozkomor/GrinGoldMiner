// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using SharedSerialization;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Mozkomor.GrinGoldMiner
{
    class ConnectionManager
    {
        private static string gflogin = "grindev";
        private static string gfpwd = "";
        
        private static int solutionCounter = 0;
        private static int solutionRound = 15;
        private static int solverswitchmin = 5;
        private static int solverswitch = 3;
        private static int prepConn = 2;
        private static int solmfcnt = 0;
        private static int solgfcnt = 0;
        private static ulong totalsolutionCounter = 0;
        private static ulong totalsolmfcnt = 0;
        private static ulong totalsolgfcnt = 0;
        private static Episode activeEpisode = Episode.main_first;
        private static bool IsGfConnecting = false;
        private static bool isMfConnecting = false;
        private static bool pauseConnecting = false;

        private static DateTime roundTime;

        private static StratumConnet curr_m; 
        private static StratumConnet con_m1;
        private static StratumConnet con_m2;

        private static StratumConnet curr_mf;
        private static StratumConnet con_mf1;
        private static StratumConnet con_mf2;

        private static StratumConnet curr_gf;
        private static StratumConnet con_gf1;
        private static StratumConnet con_gf2;

        private static StratumConnet curr;

        public static void Init(Config config)
        {
            //main
            con_m1 = new StratumConnet(config.ConnectionAddress, config.ConnectionPort, 1, config.Login, config.Password);
            con_m2 = null; // new StratumConnet("10.0.0.239", 13416, 4, "huflepuf", "azkaban");  //null; // new StratumConnet(config.SecondaryConnectionAddress, config.SecondaryConnectionPort, 2);
            //miner dev
            con_mf1 = new StratumConnet("10.0.0.237", 13416, 3, "huflepuf", "azkaban");
            con_mf2 = null; // new StratumConnet("10.0.0.237", 13416, 4, "huflepuf", "azkaban");
            //girn dev
            con_gf1 = new StratumConnet("10.0.0.239", 13416, 5, gflogin, gfpwd);
            con_gf2 = null; // new StratumConnet("10.0.0.237", 13416, 6);

            solutionCounter = 0;
            solverswitch = 3;// new Random(DateTime.UtcNow.Millisecond).Next(solverswitchmin,solutionRound);

            roundTime = DateTime.Now;
            ConnectMain();
            //ConnectMf();
            //ConnectGf();
        }

        private static void ConnectMain()
        {
            bool connected = false;

            while (!connected && !pauseConnecting)
            {

                if (con_m1 == null)
                {
                    Logger.Log(LogLevel.DEBUG, "Conection 1 is null");
                    //Console.ReadLine();
                }

                con_m1.Connect();
                if (con_m1.IsConnected)
                {
                    curr_m = con_m1;
                    connected = true;
                    Logger.Log(LogLevel.DEBUG, "conection1 succ");
                }
                else
                {
                    if (con_m2 != null)
                    {
                        con_m2.Connect();
                        if (con_m2.IsConnected)
                        {
                            curr_m = con_m2;
                            connected = true;
                            Logger.Log(LogLevel.DEBUG, "conection2 succ");
                        }
                        else
                        {
                            Logger.Log(LogLevel.DEBUG, "both con1 & con2 failed, trying again");
                            Task.Delay(1000).Wait();
                        }
                    }
                }
            }

            if (curr_m != null)
            {
                curr_m.ReconnectAction = ReconnectMain;
                curr_m.SendLogin();
                curr_m.RequestJob();
                lock (lock1)
                {
                    curr = curr_m;
                }
            }
            
        }

        public static void ReconnectMain()
        {
            Logger.Log(LogLevel.INFO, "trying to reconnect...");
           // curr_m.StratumClose(); //already in watchdog
            curr_m = null;
            ConnectMain();
        }

        private static void ConnectMf()
        {
            Logger.Log(LogLevel.DEBUG, "conecting to mf");
            bool connected = false;
            isMfConnecting = true;

            while (!connected && !pauseConnecting)
            {

                con_mf1.Connect();
                if (con_mf1.IsConnected)
                {
                    curr_mf = con_mf1;
                    connected = true;
                    Logger.Log(LogLevel.DEBUG, "conection1 mf succ");
                }
                else
                {
                    if (con_m2 != null)
                    {
                        con_mf2.Connect();
                        if (con_mf2.IsConnected)
                        {
                            curr_mf = con_mf2;
                            connected = true;
                            Logger.Log(LogLevel.DEBUG, "conection2 mf succ");
                        }
                        else
                        {
                            Logger.Log(LogLevel.DEBUG, "both con1 & con2 mf failed, trying again");
                            Task.Delay(1000).Wait();
                        }
                    }
                }
            }

            if (curr_mf != null)
            {
                curr_mf.ReconnectAction = ReconnectMf;
                curr_mf.SendLogin();
                curr_mf.RequestJob();
            }

            isMfConnecting = false;
        }

        public static void ReconnectMf()
        {
            Logger.Log(LogLevel.DEBUG, "trying to reconnect...");
            curr_mf = null;
            ConnectMf();
        }

        private static void ConnectGf()
        {
            bool connected = false;
            IsGfConnecting = true;

            while (!connected && !pauseConnecting)
            {

                con_gf1.Connect();
                if (con_gf1.IsConnected)
                {
                    curr_gf = con_gf1;
                    connected = true;
                    Logger.Log(LogLevel.DEBUG, "conection1 gf succ");
                }
                else
                {
                    if (con_gf2 != null)
                    {
                        con_gf2.Connect();
                        if (con_gf2.IsConnected)
                        {
                            curr_gf = con_gf2;
                            connected = true;
                            Logger.Log(LogLevel.DEBUG, "conection2 gf succ");
                        }
                        else
                        {
                            Logger.Log(LogLevel.DEBUG, "both con1 & con2 failed, trying again");
                            Task.Delay(1000).Wait();
                        }
                    }
                }
            }

            if (curr_gf != null)
            {
                curr_gf.ReconnectAction = ReconnectGf;
                curr_gf.SendLogin();
                curr_gf.RequestJob();
            }
           
            IsGfConnecting = false;
        }

        public static void ReconnectGf()
        {
            Logger.Log(LogLevel.DEBUG, "trying to reconnect...");
            curr_gf = null;
            ConnectGf();
        }

        internal static void CloseAll()
        {
            con_m1.StratumClose();
            con_m2.StratumClose();
            con_mf1.StratumClose();
            con_mf2.StratumClose();
            con_gf1.StratumClose();
            con_gf2.StratumClose();
        }

        private static object lock1 = "";
        public static Job GetJob()
        {
            //bude muset poznat z kteryho connectu to jde, aby ho pak mohl submitnout na spravny pool (dev, user..)
            lock (lock1)
            {
                if (curr != null)
                {
                    Logger.Log(LogLevel.DEBUG, $"job from curr id: {curr.id}");
                    return curr.CurrentJob;
                }
                else
                {
                    Logger.Log(LogLevel.INFO, "no job :(");
                    return null;
                }
            }
        }

        private static StratumConnet GetConnectionForJob(string pre_pow)
        {

            if (con_m1?.CurrentJob?.pre_pow == pre_pow || con_m1?.PrevJob?.pre_pow == pre_pow)
                return con_m1;
            else if (con_m2?.CurrentJob?.pre_pow == pre_pow || con_m2?.PrevJob?.pre_pow == pre_pow)
                return con_m2;
            else if (con_gf1?.CurrentJob?.pre_pow == pre_pow || con_gf1?.PrevJob?.pre_pow == pre_pow)
                return con_gf1;
            else if (con_gf2?.CurrentJob?.pre_pow == pre_pow || con_gf2?.PrevJob?.pre_pow == pre_pow)
                return con_gf2;
            else if (con_mf1?.CurrentJob?.pre_pow == pre_pow || con_mf1?.PrevJob?.pre_pow == pre_pow)
                return con_mf1;
            else if (con_mf2?.CurrentJob?.pre_pow == pre_pow || con_mf2?.PrevJob?.pre_pow == pre_pow)
                return con_mf2;
            else
                return null;
        }

        public static void SubmitSol(SharedSerialization.Solution solution)
        {

            var conn = GetConnectionForJob(solution.job.pre_pow);
            if (conn == null)
            {
                Logger.Log(LogLevel.WARNING, $"No connection found for solution with job id {solution.job.jobID}");
                return;
            }
            else
            {
                //conn.SendSolution(solution);
            }

            if (conn.id == 1 || conn.id == 2)
                solutionCounter++;
            if (conn.id == 3 || conn.id == 4)
                solmfcnt++;
            if (conn.id == 5 || conn.id == 6)
                solgfcnt++;

            Logger.Log(LogLevel.DEBUG, $"submiting sol jobid: {solution.job.jobID}, solutionCounter is {solutionCounter}, mfcnt is {solmfcnt}, gfcnt is {solgfcnt}");
            SwitchEpoch();
        }

        private static void SwitchEpoch()
        {
            //manage connections
            if (solutionCounter < solverswitch - prepConn)
            {
                Logger.Log(LogLevel.DEBUG, "SWITCHER: in main 1");
            }
            if (solutionCounter == solverswitch - prepConn)
            {
                Logger.Log(LogLevel.DEBUG, "SWITCHER: start connecting mf gf");
                //start connecting mf gf
                if (!isMfConnecting)
                {
                    isMfConnecting = true;
                    Task.Factory.StartNew(() => { ConnectMf(); });
                }

                if (!IsGfConnecting)
                {
                    IsGfConnecting = true;
                    Task.Factory.StartNew(() => { ConnectGf(); });
                }
                activeEpisode = Episode.f_connecting;
            }
            else if ((solutionCounter > (solverswitch - prepConn)) && (solutionCounter < solverswitch))
            {
                //connecting mf gf

                Logger.Log(LogLevel.DEBUG, "SWITCHER: connecting mf gf, submiting main");
                if (curr_mf?.IsConnected == true)
                    curr_mf.KeepAlive();

                if (curr_gf?.IsConnected == true)
                    curr_gf.KeepAlive();
            }
            else if (solutionCounter == solverswitch)
            {
                //switch to mf
                Logger.Log(LogLevel.DEBUG, "SWITCHER: switch to mf");

                DateTime now = DateTime.Now;
                while (curr_mf == null || (curr.id != 3 && curr.id != 4))
                {
                    //Logger.Log(LogLevel.DEBUG, "SWITCHER: while curr_mf == null");
                    Task.Delay(100).Wait();
                    if ((DateTime.Now - now).TotalSeconds > 6)
                        break;
                }
                if (curr_mf != null)
                {
                    curr = curr_mf;

                    if (curr.id != 1 && curr.id != 2) //one time
                        solutionCounter++;

                    Logger.Log(LogLevel.DEBUG, "SWITCHER: switched to mf");
                    activeEpisode = Episode.mf;
                }
                else
                {
                    pauseConnecting = true;
                    curr_m.StratumClose();
                    Task.Delay(500).Wait();
                    Logger.Log(LogLevel.WARNING, "Could not connect to miner dev fee. Waiting 120 seconds.");
                    Logger.Log(LogLevel.WARNING, "Could not connect to miner dev fee. Waiting 120 seconds.");
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    printHeart();
                    Console.WriteLine("Please alow connection to MINER DEV to enable mining for dev fee.");
                    Console.WriteLine("Miner dev fee (2% of your hashpower) is used to support development of this miner and Grin coin developers.");
                    Console.WriteLine("Thank you very much for supporting this project.");
                    Console.ResetColor();
                    Task.Delay(TimeSpan.FromSeconds(10)).Wait();
                    pauseConnecting = false;

                    if (curr_gf == null || curr_gf.IsConnected == false)
                    {
                        pauseConnecting = false;
                        Task.Run(()=>ReconnectGf()); //async non-blocking so we dont block the miner forever in case gf is not reachable
                        Task.Delay(2000).Wait();
                    }

                    solutionCounter += 5;
                    SwitchEpoch();
                }

            }
            else if (solutionCounter > solverswitch && solutionCounter < solverswitch + 5)
            {
                //submiting mf
                Logger.Log(LogLevel.DEBUG, "SWITCHER: submiting mf");
                if (curr.id == 3 || curr.id == 4)
                    solutionCounter++;

                if (curr_m != null)
                    curr_m.KeepAlive();

                if (curr_gf != null)
                    curr_gf.KeepAlive();
            }
            else if (solutionCounter == solverswitch + 5)
            {
                //switch to gf
                Logger.Log(LogLevel.DEBUG, "SWITCHER: switch to gf");
                DateTime now = DateTime.Now;
                while (curr_gf == null || (curr.id != 5 && curr.id != 6))
                {
                    //Logger.Log(LogLevel.DEBUG, "SWITCHER: while curr_gf == null");
                    Task.Delay(100).Wait();
                    if ((DateTime.Now - now).TotalSeconds > 6)
                        break;
                }
                if (curr_gf != null)
                {
                    curr = curr_gf;

                    //if (curr.id != 1 && curr.id != 2) //one time
                        solutionCounter++;

                    Logger.Log(LogLevel.DEBUG, "SWITCHER: switched to gf");
                    activeEpisode = Episode.gf;
                }
                else
                {
                    pauseConnecting = true;
                    curr_m.StratumClose();
                    Task.Delay(500).Wait();
                    Logger.Log(LogLevel.WARNING, "Could not connect to miner dev fee. Waiting 120 seconds.");
                    Logger.Log(LogLevel.WARNING, "Could not connect to miner dev fee. Waiting 120 seconds.");
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    printHeart();
                    Console.WriteLine("Please alow connection to GRIN DEV to enable mining for dev fee.");
                    Console.WriteLine("Miner dev fee (2% of your hashpower) is used to support development of this miner and Grin coin developers.");
                    Console.WriteLine("Thank you very much for supporting this project.");
                    Console.ResetColor();
                    Task.Delay(TimeSpan.FromSeconds(10)).Wait();
                    pauseConnecting = false;
                    solutionCounter += 5;
                    SwitchEpoch();
                }
            }
            else if (solutionCounter > solverswitch + 5 && solutionCounter < solverswitch + 10)
            {
                //submiting gf
                Logger.Log(LogLevel.DEBUG, "SWITCHER: submiting gf");
                if (curr.id == 5 || curr.id == 6)
                    solutionCounter++;

                if (curr_m != null)
                    curr_m.KeepAlive();
            }
            else if (solutionCounter == solverswitch + 10)
            {
                if (curr_m == null || curr_m.IsConnected == false)
                {
                    pauseConnecting = false;
                    ReconnectMain(); //blocking until connected - main user minig
                }
                //main 2
                Logger.Log(LogLevel.DEBUG, "SWITCHER: switch to main 2");
                curr = curr_m;
                DateTime now = DateTime.Now;
                while (curr_m == null || (curr.id != 1 && curr.id != 2))
                {
                    //kdyz bude spojeni blokovany, aby to tady nezustalo navzdy viset
                    // jeste nejak zakazat posilatsolutions for main
                    if ((DateTime.Now - now).TotalSeconds > 60)
                        break;
                }
                curr = curr_m;
                if (curr_gf != null && curr_gf.IsConnected)
                    curr_gf.StratumClose();

                activeEpisode = Episode.main_second;
            }
            else if (solutionCounter > solverswitch + 10 && solutionCounter < solutionRound)
            {
                Logger.Log(LogLevel.DEBUG, "SWITCHER: main 2");
            }
            else if (solutionCounter >= solutionRound)
            {
                Logger.Log(LogLevel.DEBUG, "SWITCHER: restart round");
                //restart round
                totalsolutionCounter += (ulong)solutionCounter;
                totalsolmfcnt += (ulong)solmfcnt;
                totalsolgfcnt += (ulong)solgfcnt;
                Logger.Log(LogLevel.DEBUG, $"Total sols submitted: {totalsolutionCounter}, totalmfcnt is {totalsolmfcnt}, totalgfcnt is {totalsolgfcnt}");
                Logger.Log(LogLevel.DEBUG, $"Round time: {(DateTime.Now - roundTime).TotalSeconds}  seconds");
                Logger.Log(LogLevel.DEBUG, $"Avg sol time: {(DateTime.Now - roundTime).TotalSeconds / solutionRound} seconds");
                roundTime = DateTime.Now;
                solutionCounter = 0;
                solgfcnt = 0;
                solmfcnt = 0;
                activeEpisode = Episode.main_first;
            }
        }

        enum Episode
        {
            main_first,
            f_connecting,
            mf,
            gf,
            main_second
        }

        private static void printHeart()
        {
            Console.WriteLine("       .....           .....");
            Console.WriteLine("   ,ad8PPPP88b,     ,d88PPPP8ba,");
            Console.WriteLine("  d8P\"      \"Y8b, ,d8P\"      \"Y8b");
            Console.WriteLine(" dP'           \"8a8\"           `Yd");
            Console.WriteLine(" 8(              \"              )8");
            Console.WriteLine(" I8                             8I");
            Console.WriteLine("  Yb,                         ,dP");
            Console.WriteLine("   \"8a,                     ,a8\"");
            Console.WriteLine("     \"8a,                 ,a8\"");
            Console.WriteLine("       \"Yba             adP\"");
            Console.WriteLine("         `Y8a         a8P'");
            Console.WriteLine("           `88,     ,88'");
            Console.WriteLine("             \"8b   d8\"");
            Console.WriteLine("              \"8b d8\"");
            Console.WriteLine("               `888'");
            Console.WriteLine("                 \"");
        }
    }
}
