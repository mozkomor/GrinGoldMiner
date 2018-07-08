# GPU Miner for Cuckoo Cycle PoW - nvidia beta version

This miner is currently (testnet3) in early beta stage, OpenCL version is expected to arrive later with multiple improvements: amd support, speed boost, more reliable inter-process socket connection, easier linux install, multi-gpu, binary package (windows, linux).

Beta miner requires 8GB GPU for cuckoo30. Final miner may work on 6GB GPUs. Grin PoW is in flux.

## Expected Perfomance (OpenCL)

    GTX 1070    - up to 2.9 gps
    GTX 1070 Ti - up to 3.2 gps
    GTX 1080 Ti - up to 5.5 Graphs/s @ ~250W
    GTX 1080 Ti - up to 3.9 Graphs/s @ ~125W
    Vega64      - unknown (1080Ti > Vega64 > 1070)
    RX580       - unknown - do not have
    Intel iGPU  - in future update

## Support

Special thanks to John Tromp, Kristy-Leigh Minehan (OhGodACompany) and Manli Technology for sending us new shiny HW (1080 Ti) for development and tweaking + helping with slow AMD code, much appreciated!

License for the software was changed after publication to be compatible with the bounty. Closed sourced miners based on this work will now be required to pay full 50% of the fee to coin developers and keep the other 50%.

If you want to support or speedup development of this cross platform miner, contact us at mozkomorkaban@gmail.com.

----------------------------------

4x speedup bounty has been received. No further bounty claims will be made.

Contact us at mozkomorkaban@gmail.com

Based on great work of John Tromp, the inventor of Cuckoo cycle PoW  https://github.com/tromp/cuckoo

And designed for https://github.com/mimblewimble/grin
