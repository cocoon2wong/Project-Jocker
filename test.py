"""
@Author: Conghao Wong
@Date: 2023-06-07 15:32:41
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-17 10:11:14
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from main import main


class TestStructure():
    def test_CO(self):
        main(["main.py",
              "--model", "MKII",
              "--loada", "weights/Silverballers/EV_co_DFT_zara1",
              "--loadb", "weights/Silverballers/VB_co_DFT_zara1",])

    def test_BB(self):
        main(["main.py",
              "--model", "MKII",
              "--loada", "weights/Silverballers/EV_bb_DFT_sdd",
              "--loadb", "weights/Silverballers/VB_co_DFT_sdd",
              "-p", "0.7",
              "--test_mode", "one",
              "--force_clip", "little0",])

    def test_CO2BB(self):
        main(["main.py",
              "--model", "MKII",
              "--loada", "weights/Silverballers/EV_co_DFT_sdd",
              "--loadb", "weights/Silverballers/VB_co_DFT_sdd",
              "--test_mode", "one",
              "--force_clip", "little0",
              "--force_anntype", "boundingbox",])

    def test_3DBB(self):
        main(["main.py",
              "--model", "MKII",
              "--loada", "./weights/Silverballers/EV_3dbb_DFT_nuScenes",
              "--loadb", "l",
              "--test_mode", "one",
              "--force_clip", "scene-0003",])

    def test_3DSkeleton17(self):
        main(["main.py",
              "--model", "MKII",
              "--loada", "./weights/Silverballers/_experimental/EV_3dskeleton-17_Haar_h36m",
              "--loadb", "l",
              "--test_mode", "one",
              "--force_clip", "s_05_act_02",])


class TestSocialCircle():
    def test_train_EVSC(self):
        main(["main.py",
              "--model", "evsc",
              "--split", "sdd_debug",
              "-bs", "300",
              "--key_points", "4_8_11",
              "-lr", "3e-4",
              "--step", "4",
              "--test_step", "1",
              "--epochs", "2"])
