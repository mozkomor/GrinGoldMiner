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
    public class ConnectionManager
    {
        private static string gflogin = "grindev";
        private static string gfpwd = "";

        public static volatile uint solutions = 0;
        public static volatile int solutionCounter = 0;
        private static volatile int solutionRound = 30;
        private static volatile int solverswitchmin = 5;
        private static volatile int solverswitch = 10;
        private static volatile int prepConn = 2;
        private static volatile int solmfcnt = 0;
        private static volatile int solgfcnt = 0;
        private static volatile uint totalsolutionCounter = 0;
        private static volatile uint totalsolmfcnt = 0;
        private static volatile uint totalsolgfcnt = 0;
        private static volatile bool IsGfConnecting = false;
        private static volatile bool isMfConnecting = false;
        private static volatile bool stopConnecting = false;
        

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

        public static bool IsInFee() => (GetCurrentEpoch() != Episode.user);

        public static void Init(Config config)
        {
            //main
            con_m1 = new StratumConnet(config.PrimaryConnection.ConnectionAddress, config.PrimaryConnection.ConnectionPort, 1, config.PrimaryConnection.Login, config.PrimaryConnection.Password, config.PrimaryConnection.Ssl);
            con_m2 = new StratumConnet(config.SecondaryConnection.ConnectionAddress, config.SecondaryConnection.ConnectionPort, 1, config.SecondaryConnection.Login, config.SecondaryConnection.Password, config.SecondaryConnection.Ssl);
            //miner dev
            con_mf1 = new StratumConnet("10.0.0.237", 13416, 3, "huflepuf", "azkaban");
            con_mf2 = null; // new StratumConnet("10.0.0.237", 13416, 4, "huflepuf", "azkaban");
            //girn dev
            con_gf1 = new StratumConnet("10.0.0.239", 13416, 5, gflogin, gfpwd);
            con_gf2 = null; // new StratumConnet("10.0.0.237", 13416, 6);

            solutionCounter = 0;
            //solverswitch = 30;// new Random(DateTime.UtcNow.Millisecond).Next(solverswitchmin,solutionRound);

            roundTime = DateTime.Now;
            stopConnecting = false;
            ConnectMain();
        }

        #region connecting
        //main (user) connection is disconnected, pause workers (stop burning electricity), disconnect fees, wait for reconnection
        private static void PauseAllMining()
        {
            if(curr_mf?.IsConnected == true)
                curr_mf.StratumClose();

            if(curr_gf?.IsConnected == true)
                curr_gf.StratumClose();

            Task.Delay(1000).Wait();

            WorkerManager.PauseAllWorkers();

            solutionCounter = 0;
        }

        private static void ConnectMain()
        {
            bool connected = false;

            DateTime started = DateTime.Now;

            while (!connected && !stopConnecting)
            {
                if (DateTime.Now - started > TimeSpan.FromSeconds(60))
                {
                    //main connection not available for more than 60s
                    PauseAllMining();
                }

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
            }
        }

        public static void ReconnectMain()
        {
            Logger.Log(LogLevel.DEBUG, "trying to reconnect main...");
           // curr_m.StratumClose(); //already in watchdog
            curr_m = null;
            stopConnecting = false;
            ConnectMain();
        }

        private static void ConnectMf()
        {
            Logger.Log(LogLevel.DEBUG, "conecting to mf");
            bool connected = false;
            isMfConnecting = true;

            while (!connected && !stopConnecting)
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
                    if (con_mf2 != null)
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
            stopConnecting = false;
            ConnectMf();
        }

        private static void ConnectGf()
        {
            bool connected = false;
            IsGfConnecting = true;

            while (!connected && !stopConnecting)
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
            stopConnecting = false;
            ConnectGf();
        }

        public static void CloseAll()
        {
            con_m1.StratumClose();
            con_m2.StratumClose();
            con_mf1.StratumClose();
            con_mf2.StratumClose();
            con_gf1.StratumClose();
            con_gf2.StratumClose();
        }
        #endregion

        #region state
        public static bool IsConnectionCurrent(int id)
        {
            if ((id == 1 || id == 2) && GetCurrentEpoch() == Episode.user)
                return true;

            if ((id == 3 || id == 4) && GetCurrentEpoch() == Episode.mf)
                return true;

            if ((id == 5 || id == 6) && GetCurrentEpoch() == Episode.gf)
                return true;

            return false;
        }

        public static StratumConnet GetCurrConn()
        {
            var ep = GetCurrentEpoch();
            switch (ep)
            {
                case Episode.user:
                    return curr_m;
                case
                    Episode.mf:
                    return curr_mf;
                case Episode.gf:
                    return curr_gf;
            }

            return curr_m;
        }

        private static Episode GetCurrentEpoch()
        {
            if (solutionCounter < solverswitch)
            {
                return Episode.user;
            }
            else if (solutionCounter < solverswitch +10)
            {
                return Episode.mf;
            }
            else if (solutionCounter < solverswitch + 20)
            {
                return Episode.gf;
            }
            else
            {
                return Episode.user;
            }
        }
        #endregion

        #region submit
        public static string lock_submit = "";
        public static void SubmitSol(SharedSerialization.Solution solution)
        {
            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SubmitSol...");

            var ep = GetCurrentEpoch();
            if (solution.job.origin == ep)
            {
                switch (ep)
                {
                    case Episode.user:
                        if (curr_m?.IsConnected == true)
                        {
                            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) Submitting solution Connection id {curr_m.id} SOL: job id {solution.job.jobID} origine {solution.job.origin.ToString()}. ");
                            curr_m.SendSolution(solution);
                        }
                        break;
                    case Episode.mf:
                        if (curr_mf?.IsConnected == true)
                        {
                            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) Submitting solution Connection id {curr_mf.id} SOL: job id {solution.job.jobID} origine {solution.job.origin.ToString()}. ");
                            curr_mf.SendSolution(solution);
                        }
                        break;
                    case Episode.gf:
                        if (curr_gf?.IsConnected == true)
                        {
                            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) Submitting solution Connection id {curr_gf.id} SOL: job id {solution.job.jobID} origine {solution.job.origin.ToString()}. ");
                            curr_gf.SendSolution(solution);
                        }
                        break;
                }
            }

            lock (lock_submit)
            {
                solutions++;
                solutionCounter++;

                if (solutionCounter == solverswitch - prepConn)
                {
                    Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SWITCHER: start connecting mf gf");
                    //start connecting mf gf
                    if (!isMfConnecting)
                    {
                        stopConnecting = false;
                        isMfConnecting = true;
                        Task.Run(() => ConnectMf());
                    }

                    if (!IsGfConnecting)
                    {
                        stopConnecting = false;
                        IsGfConnecting = true;
                        Task.Run(() => ConnectGf());
                    }
                }
                else if(solutionCounter == solverswitch)
                {
                    if (curr_mf?.IsConnected == true)
                    {
                        Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SWITCHER: pushing MF job to workers");
                        curr_mf.PushJobToWorkers();
                    }
                }
                else if(solutionCounter == solverswitch + 10)
                {
                    if (curr_gf?.IsConnected == true)
                    {
                        Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SWITCHER: pushing GF job to workers");
                        curr_gf.PushJobToWorkers();
                    }
                }
                else if(solutionCounter == solverswitch + 20)
                {
                    if (curr_m?.IsConnected == true)
                    {
                        Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SWITCHER: pushing USER job to workers");
                        curr_m.PushJobToWorkers();
                    }
                    else
                    {
                        stopConnecting = false;
                        ConnectMain();
                    }

                    /// !!!!!!!!!!!!!!! 
                    solutionCounter = 0; //remove this if we want two m-mf-gf-m, now we have m-mf-gf so it will never fall to next else-if

                    stopConnecting = true; //in case mf gf are not reachable, they are trying to connect here in loop

                    try { Task.Run(() => Task.Delay(5000).ContinueWith(_ => curr_mf.StratumClose())); } catch { }
                    try { Task.Run(() => Task.Delay(5000).ContinueWith(_ => curr_gf.StratumClose())); } catch { }
                }
                else if(solutionCounter > 30)
                {
                    solutionCounter = 0;
                }
            }
        }
        #endregion

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
