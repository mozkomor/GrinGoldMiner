// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Kukacka optimized miner is covered by the FAIR MINING license


using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Terminal.Gui;

namespace GGM
{
    class UI
    {
        public static void MainLoop()
        {
            Application.Init();
            var top = Application.Top;

            // Creates the top-level window to show
            var title = new Window(new Rect(0, 0, top.Frame.Width, 5), "Info");
            top.Add(title);

            var win = new Window(new Rect(0, 5, top.Frame.Width / 3, top.Frame.Height - 5), "GPU 1");
            top.Add(win);
            var win2 = new Window(new Rect(top.Frame.Width / 3, 5, top.Frame.Width / 3, top.Frame.Height - 5), "GPU 2");
            top.Add(win2);
            var win3 = new Window(new Rect((top.Frame.Width / 3) * 2, 5, top.Frame.Width / 3, top.Frame.Height - 5), "GPU 3");
            top.Add(win3);

            // Creates a menubar, the item "New" has a help menu.
            //    var menu = new MenuBar(new MenuBarItem[] {
            //    new MenuBarItem ("_File", new MenuItem [] {
            //        new MenuItem ("_New", "Creates new file", NewFile),
            //        new MenuItem ("_Close", "", () => Close ()),
            //        new MenuItem ("_Quit", "", () => { if (Quit ()) top.Running = false; })
            //    }),
            //    new MenuBarItem ("_Edit", new MenuItem [] {
            //        new MenuItem ("_Copy", "", null),
            //        new MenuItem ("C_ut", "", null),
            //        new MenuItem ("_Paste", "", null)
            //    })
            //});
            //    top.Add(menu);

            // Add some controls
            //win.Add(
            //        new Label(3, 2, "Login: "),
            //        new TextField(14, 2, 40, ""),
            //        new Label(3, 4, "Password: "),
            //        new TextField(14, 4, 40, "") { Secret = true },
            //        new CheckBox(3, 6, "Remember me"),
            //        new RadioGroup(3, 8, new[] { "_Personal", "_Company" }),
            //        new Button(3, 14, "Ok"),
            //        new Button(10, 14, "Cancel"),
            //        new Label(3, 18, "Press ESC and 9 to activate the menubar"));

            Application.Run();
        }

        internal static void StartUI()
        {
            //MainLoop();
            while (true)
            {
                Task.Delay(100);
            }
        }
    }
}
