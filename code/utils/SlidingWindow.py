#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2020.09.22
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#


class SlidingWindow:
    def __init__(self, overlapping, samp_freq, window_feat_size):
        self.ov = overlapping
        self.fs = samp_freq
        self.wf = window_feat_size



    def extractRawVector(self):
        print(self.ov)