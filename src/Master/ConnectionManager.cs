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
        public static volatile uint solutions = 0;
        public static volatile int solutionCounter = 0;
        private static volatile int solutionRound = 1000;
        //private static volatile int solverswitchmin = 5;
        private const int solverswitch = 980;
        private static volatile int prepConn = 10;
        //private static volatile int solmfcnt = 0;
        //private static volatile int solgfcnt = 0;
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

        /// 
        /// GGM collects 1% as fee for the Grin Development Fund and 1% for further miner development.
        /// GGM is open source and solvers are released under fair mining licence,
        /// thanks for plyaing fair and keeping the fees here. It allows continuing developmlent of Grin and GGM.
        public static bool IsInFee() => (GetCurrentEpoch() != Episode.user);
        private static string gf_login = "grincouncil@protonmail.com";
        private static string mf_login = "ggmfee0115@protonmail.com";

        public static void Init(Config config)
        {
            //main
            con_m1 = new StratumConnet(config.PrimaryConnection.ConnectionAddress, config.PrimaryConnection.ConnectionPort, 1, config.PrimaryConnection.Login, config.PrimaryConnection.Password, config.PrimaryConnection.Ssl);
            con_m2 = new StratumConnet(config.SecondaryConnection.ConnectionAddress, config.SecondaryConnection.ConnectionPort, 2, config.SecondaryConnection.Login, config.SecondaryConnection.Password, config.SecondaryConnection.Ssl);
            //miner dev
            con_mf1 = new StratumConnet("us-east-stratum.grinmint.com", 4416, 3, mf_login, "", true);
            con_mf2 = new StratumConnet("eu-west-stratum.grinmint.com", 4416, 4, mf_login, "", true);
            //con_mf2 = new StratumConnet("gringoldminer2.mimwim.eu", 3334, 4, mf_login, "", true);

            //girn dev
            con_gf1 = new StratumConnet("us-east-stratum.grinmint.com", 4416, 5, gf_login, "", true);
            con_gf2 = new StratumConnet("eu-west-stratum.grinmint.com", 4416, 6, gf_login, "", true);
            //con_gf1 = new StratumConnet("10.0.0.239", 13416, 5, gf_login, "");
            //con_gf2 = new StratumConnet("gringoldminer2.mimwim.eu", 3334, 6, gf_login, "", true);

            solutionCounter = 0;
            //solverswitch = 30;// new Random(DateTime.UtcNow.Millisecond).Next(solverswitchmin,solutionRound);

            roundTime = DateTime.Now;
            stopConnecting = false;
            ConnectMain();

            if (config.ReconnectToPrimary > 0)
                Task.Factory.StartNew(() => CheckPrimary(config.ReconnectToPrimary), TaskCreationOptions.LongRunning);
        }

        public static void CheckPrimary(int minutes)
        {
            while (true)
            {
                Task.Delay(TimeSpan.FromMinutes(minutes)).Wait();
                TryPrimary();
            }
        }

        public static void TryPrimary()
        {
            if (con_m1?.IsConnected == false)
            {
                if (con_m2?.IsConnected == true)
                {
                    con_m1.Connect();
                    if (con_m1?.IsConnected == true)
                    {
                        con_m1.ReconnectAction = ReconnectMain;
                        con_m1.SendLogin();

                        curr_m = con_m1;
                        curr_m.RequestJob();

                        con_m2.StratumClose();
                    }
                }
            }
        }

        #region connecting
        //main (user) connection is disconnected, pause workers (stop burning electricity), disconnect fees, wait for reconnection
        private static void PauseAllMining()
        {
            if (curr_mf?.IsConnected == true)
                curr_mf.StratumClose();

            if (curr_gf?.IsConnected == true)
                curr_gf.StratumClose();

            Task.Delay(1000).Wait();

            WorkerManager.PauseAllWorkers();

            solutionCounter = 0;
        }

        private static void ConnectMain(bool chooseRandom = false)
        {

            bool connected = false;

            DateTime started = DateTime.Now;

            int i = 1;
            var rnd = new Random(DateTime.Now.Millisecond);

            while (!connected && !stopConnecting)
            {
                if (DateTime.Now - started > TimeSpan.FromSeconds(60))
                {
                    //main connection not available for more than 60s
                    PauseAllMining();
                }

                i = Math.Min(60, i);
                i++;
                Task.Delay(i * 1000).Wait();

                var flip = rnd.NextDouble();
                Logger.Log(LogLevel.DEBUG, $"reconnecting rnd {flip}");
                //

                if (chooseRandom)
                {
                    if (flip < 0.5)
                    {
                        con_m1.Connect();
                        if (con_m1.IsConnected)
                        {
                            curr_m = con_m1;
                            connected = true;
                            Logger.Log(LogLevel.DEBUG, "conection1 succ");
                        }
                    }
                    else
                    {
                        con_m2.Connect();
                        if (con_m2.IsConnected)
                        {
                            curr_m = con_m2;
                            connected = true;
                            Logger.Log(LogLevel.DEBUG, "conection2 succ");
                        }
                    }
                }
                else
                {
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
                                //Task.Delay(1000).Wait();
                            }
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
            Task.Delay(2500).Wait();
            Logger.Log(LogLevel.DEBUG, "trying to reconnect main...");
            // curr_m.StratumClose(); //already in watchdog
            curr_m = null;
            stopConnecting = false;
            ConnectMain(chooseRandom:true);
        }

        private static void ConnectMf()
        {
            Logger.Log(LogLevel.DEBUG, "conecting to mf");
            bool connected = false;
            isMfConnecting = true;
            int i = 1;
            while (!connected && !stopConnecting)
            {
                i = Math.Min(60, i);
                i++;
                Task.Delay(i * 1000).Wait();

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
            int i = 0;

            while (!connected && !stopConnecting)
            {
                i = Math.Min(60, i);
                i++;
                Task.Delay(i * 1000).Wait();

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
            else if (solutionCounter < solverswitch + 10)
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

        private static bool hasMfJob()
        {
            try
            {
                return curr_mf?.IsConnected == true && !string.IsNullOrEmpty(curr_mf?.CurrentJob?.pre_pow);
            }
            catch { return false; }
        }
        private static bool hasGfJob()
        {
            try
            {
                return curr_gf?.IsConnected == true && !string.IsNullOrEmpty(curr_gf?.CurrentJob?.pre_pow);
            }
            catch { return false; }
        }
        private static bool hasUserJob()
        {
            try
            {
                return curr_m?.IsConnected == true && !string.IsNullOrEmpty(curr_m?.CurrentJob?.pre_pow);
            }
            catch { return false; }
        }

        public static void SubmitSol(SharedSerialization.Solution solution)
        {
            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SubmitSol...");
            lock (lock_submit)
            {
                var ep = GetCurrentEpoch();
                switch (ep)
                {
                    case Episode.mf:
                        if (hasMfJob() && solution.job.origin == Episode.user)
                        {
                            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SubmitSol going out bc origin==user");
                            return;
                        }
                        break;
                    case Episode.gf:
                        if (hasGfJob() && solution.job.origin == Episode.mf)
                        {
                            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SubmitSol going out bc origin==mf");
                            return;
                        }
                        break;
                    case Episode.user:
                        if (hasUserJob() && solution.job.origin == Episode.gf)
                        {
                            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SubmitSol going out bc origin==gf");
                            return;
                        }
                        break;
                }

                if (solution.job.origin == ep)
                {
                    switch (ep)
                    {
                        case Episode.user:
                            if (curr_m?.IsConnected == true)
                            {
                                Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) Submitting solution Connection id {curr_m.id} SOL: job id {solution.job.jobID} origine {solution.job.origin.ToString()}. ");
                                curr_m.SendSolution(solution);
                                totalsolutionCounter++;
                            }
                            break;
                        case Episode.mf:
                            if (curr_mf?.IsConnected == true)
                            {
                                Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) Submitting solution Connection id {curr_mf.id} SOL: job id {solution.job.jobID} origine {solution.job.origin.ToString()}. ");
                                curr_mf.SendSolution(solution);
                                totalsolmfcnt++;
                            }
                            break;
                        case Episode.gf:
                            if (curr_gf?.IsConnected == true)
                            {
                                Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) Submitting solution Connection id {curr_gf.id} SOL: job id {solution.job.jobID} origine {solution.job.origin.ToString()}. ");
                                curr_gf.SendSolution(solution);
                                totalsolgfcnt++;
                            }
                            break;
                    }
                }
                else
                {
                    Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) Cant be here. (origin != ep)");
                }


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
                else if (solutionCounter == solverswitch)
                {
                    Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) solutionCounter == solverswitch");
                    if (curr_mf?.IsConnected == true)
                    {
                        Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SWITCHER: pushing MF job to workers");
                        curr_mf.PushJobToWorkers();
                    }
                }
                else if (solutionCounter == solverswitch + 10)
                {
                    Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) solutionCounter == solverswitch + 10");
                    if (curr_gf?.IsConnected == true)
                    {
                        Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SWITCHER: pushing GF job to workers");
                        curr_gf.PushJobToWorkers();
                    }
                }
                else if (solutionCounter == solverswitch + 20)
                {
                    Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) solutionCounter == solverswitch + 20");
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
                    resetRound(); //remove this if we want two m-mf-gf-m, now we have m-mf-gf so it will never fall to next else-if

                    stopConnecting = true; //in case mf gf are not reachable, they are trying to connect here in loop

                    tryCloseConn(curr_mf);
                    tryCloseConn(curr_gf);
                }
                else if (solutionCounter >= solutionRound)
                {
                    Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) Cant be here: solutionCounter >= solutionRound");
                    //this only executes in case we switch to m-mf-gf-m
                    //now it is only m-mf-gf co never comes here
                    resetRound();
                }
            }


        }

        private static void tryCloseConn(StratumConnet conn)
        {
            try
            {
                Task.Run(() => Task.Delay(5000).ContinueWith(_ =>
                {
                    try { conn.StratumClose(); } catch { }
                }
                ));
            }
            catch { }
        }

        private static void resetRound()
        {
            Logger.Log(LogLevel.DEBUG, $"({solutionCounter}) SWITCHER: resetting round, setting solutionCounter to 0");
            solutionCounter = 0;
            var time = DateTime.Now - roundTime;
            roundTime = DateTime.Now;
            //based on solution time, try to target prepConn to 10 seconds but minimum 10 sols
            prepConn = (int)Math.Round(Math.Max(10, (10 / (time.TotalSeconds / solutionRound))));

            try
            {
                //Task.Run(() =>
                //{
                Logger.Log(LogLevel.DEBUG, $"Round reset: solT:{totalsolutionCounter}, mfT:{totalsolmfcnt}, gfT:{totalsolgfcnt}");
                Logger.Log(LogLevel.DEBUG, $"Round time: {(time).TotalSeconds}  seconds");
                Logger.Log(LogLevel.DEBUG, $"PrepConn: {prepConn}");
                Logger.Log(LogLevel.INFO, $"Avg sol time: {(time).TotalSeconds / solutionRound} seconds");
                //});
            }
            catch { }
        }
        #endregion

        //public static void printHeart()
        //{
        //    Console.WriteLine("       .....           .....");
        //    Console.WriteLine("   ,ad8PPPP88b,     ,d88PPPP8ba,");
        //    Console.WriteLine("  d8P\"      \"Y8b, ,d8P\"      \"Y8b");
        //    Console.WriteLine(" dP'           \"8a8\"           `Yd");
        //    Console.WriteLine(" 8(              \"              )8");
        //    Console.WriteLine(" I8                             8I");
        //    Console.WriteLine("  Yb,                         ,dP");
        //    Console.WriteLine("   \"8a,                     ,a8\"");
        //    Console.WriteLine("     \"8a,                 ,a8\"");
        //    Console.WriteLine("       \"Yba             adP\"");
        //    Console.WriteLine("         `Y8a         a8P'");
        //    Console.WriteLine("           `88,     ,88'");
        //    Console.WriteLine("             \"8b   d8\"");
        //    Console.WriteLine("              \"8b d8\"");
        //    Console.WriteLine("               `888'");
        //    Console.WriteLine("                 \"");
        //}
    }
}
