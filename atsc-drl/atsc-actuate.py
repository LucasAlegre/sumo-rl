from __future__ import absolute_import
from __future__ import print_function

import re
import numpy as np
import pandas as pd
import os
import sys
import traci  # noqa
import time

"""
这是一套比较完整的程序思路，不记得其出处了。值得研读并改造。
"""
class network_parameter_configuration():
    def __init__(self):
        # 交叉口编号为0的路口 ，第"0"号相位方案，第0个相位放行流向是东进口的掉头左转直行
        self.inter2phase_turn_dict = \
            {"0": {"0": {0: {"e": ["uturn", "left", "straight"]},
                         3: {"w": ["uturn", "left", "straight"]},
                         6: {"n": ["uturn", "left", "straight"]},
                         9: {"s": ["uturn", "left", "straight"]}}}}
        # 如果右转不受控就不要右转设置车检器
        self.inter2inductiondetwithturndir_dict_control = \
            {"0": {"e": {"uturn": [], "left": ["e_Loop5_2"], "straight": ["e_Loop2_2", "e_Loop3_2", "e_Loop4_2"], "right": []},
                   "w": {"uturn": [], "left": ["w_Loop5_2"], "straight": ["w_Loop2_2", "w_Loop3_2", "w_Loop4_2"], "right": []},
                   "s": {"uturn": [], "left": ["s_Loop5_2"], "straight": ["s_Loop2_2", "s_Loop3_2", "s_Loop4_2"], "right": []},
                   "n": {"uturn": [], "left": ["n_Loop5_2"], "straight": ["n_Loop2_2", "n_Loop3_2", "n_Loop4_2"], "right": []}}}


class inter_inductionpara_configuration():
    def __init__(self, interid, programeid, network_parameter_configuration_object):
        # 配置交叉口的参数
        self.interid = interid
        self.programeid = programeid
        # 单位绿灯延长时间(s)
        self.greenlength_extension = 6
        # 下一个、上一个感应控制决策时间点
        self.nextdecision_point_time = 13
        self.lastdecision_point_time = 0
        # 当前、下一个绿灯相位编号
        self.current_stage_label = 0
        self.next_stage_label = 3
        # 当前绿灯相位放行时长
        self.currentcarphase_greentime = 26
        # 绿灯相位开始时刻点
        self.nextcyclestart_nextgreenstagestart_time = 0

        # 初始化记录周期开始结束的时间
        self.cyclestarttime = 0
        self.cycleendtime = 0

        self.cycle_stage_passseq = []  # 记录周期运行的相序情况
        # 路口的信号控制参数 定义路口编号——对应的相位方案编号——对应的相位阶段编号下的下一个绿灯相位阶段编号、
        # 当前相位到下一个绿灯相位需要过渡几个阶段（黄灯/全红）、 当前相位到下一个绿灯相位的过渡时间（黄灯+全红）
        self.intersectionid_programeid_stage_nextstage_dict = \
            {'0': {'0': {0: (3, 2, 5.0),
                         1: (3, 1, 2.0),
                         2: (3, 0, 0),
                         3: (6, 2, 5.0),
                         4: (6, 1, 2.0),
                         5: (6, 0, 0),
                         6: (9, 2, 5.0),
                         7: (9, 1, 2.0),
                         8: (9, 0, 0),
                         9: (0, 2, 5.0),
                         10: (0, 1, 2.0),
                         11: (0, 0, 0)}}}
        # 定义路口编号——对应的相位方案编号——对应的相位阶段编号下的最小、最大绿灯时长
        self.intersectionid_programeid_stage_mingreentime_dict = \
            {'0': {'0': {0: 13.0, 1: 3.0, 2: 2.0, 3: 13.0, 4: 3.0, 5: 2.0, 6: 15.0, 7: 3.0, 8: 2.0, 9: 15.0, 10: 3.0, 11: 2.0}}}
        self.intersectionid_programeid_stage_maxgreentime_dict = \
            {'0': {'0': {0: 26.0, 1: 3.0, 2: 2.0, 3: 29.0, 4: 3.0, 5: 2.0, 6: 33.0, 7: 3.0, 8: 2.0, 9: 32.0, 10: 3.0, 11: 2.0}}}
        self.greenstagelabellist = [0, 3, 6, 9]
        # 车辆过车时刻记录
        self.inter_vehentertime_dataframe = pd.DataFrame()


class state_represent():
    # 当前相位状态下感应检测器下车辆的通过情况
    # 路网参数配置类  network_parameter_configuration_class
    # 某一个路口参数配置类  inter_parameter_configuration_class
    def obtain_vehentertime(step, network_parameter_configuration_class, inter_parameter_configuration_class):
        interid = inter_parameter_configuration_class.interid
        Phaselabel = int(traci.trafficlight.getPhase(interid))  # 得到路口当前相位阶段编号
        programeid = inter_parameter_configuration_class.programeid
        try:
            phase_turn_dict = network_parameter_configuration_class.inter2phase_turn_dict[interid][programeid][Phaselabel]
            interid_inductiondet = network_parameter_configuration_class.inter2inductiondetwithturndir_dict_control[interid]
            for enterdir in phase_turn_dict:
                for turndir in phase_turn_dict[enterdir]:
                    detid_list = interid_inductiondet[enterdir][turndir]
                    for detid in detid_list:
                        vehicledata = traci.inductionloop.getVehicleData(detid)
                        for onevehicle in vehicledata:
                            vehicleid = onevehicle[0]
                            entrytime = onevehicle[2]
                            x = {"interid": interid, "Phaselabel": Phaselabel, "enterdir": enterdir,
                                 "turndir": turndir, "step": step, "vehicleid": vehicleid, "entrytime": entrytime}
                            inter_parameter_configuration_class.inter_vehentertime_dataframe = (
                                inter_parameter_configuration_class.inter_vehentertime_dataframe.append(x, ignore_index=True))
        except:
            print("无车辆通过")


class choose_action():
    # 判断是延长绿灯还是切换至下一个相位
    def choose_action_induction(step, network_parameter_configuration_class, inter_parameter_configuration_class):
        greenlength_extension = inter_parameter_configuration_class.greenlength_extension
        interid = inter_parameter_configuration_class.interid
        programeid = inter_parameter_configuration_class.programeid
        Phaselabel = traci.trafficlight.getPhase(interid)  # 得到路口当前相位阶段编号
        try:
            # print("eee")
            vehsize = (
                len(inter_parameter_configuration_class.inter_vehentertime_dataframe[
                        (inter_parameter_configuration_class.inter_vehentertime_dataframe["step"] >= step - greenlength_extension)
                        & (inter_parameter_configuration_class.inter_vehentertime_dataframe["step"] <= step)]))
            if vehsize >= 1:
                # action_name="current"
                action_name = Phaselabel
            else:
                # action_name="next"
                action_name = (
                    inter_parameter_configuration_class.
                    intersectionid_programeid_stage_nextstage_dict)[interid][programeid][Phaselabel][0]
        except:
            action_name = (
                inter_parameter_configuration_class.
                intersectionid_programeid_stage_nextstage_dict)[interid][programeid][Phaselabel][0]
        return action_name


class action_decision():

    def action_decision_phaseorder_change(inter_parameter_configuration_object, network_parameter_configuration_object):
        # 该函数的行为动作为相序可变，设置了最大绿、最小绿，单位绿，满足最小绿后判断是否切换至任意相位，
        # 当决策是当前相位时，延长单位绿灯时间，输出下一决策点时间
        # action_name 动作名称   Phaselabel当前运行的相位阶段号， stage_time #当前相位开始放行时间
        # intersection_phaseprograme_object 为intersection_phaseprograme实例化的一个对象
        # 包含stage_mingreentime、stage_maxgreentime、stage_yellowplusallredtime、stage_tonextstage_whetheryelloworred等属性
        # stage_mingreentime、stage_maxgreentime各个相位阶段的最小、最大放行时间   list序列
        # stage_yellowplusallredtime  各个相位阶段的黄灯+全红时间
        # stage_tonextstage_whetheryelloworred
        # 各个阶段切换至下相位的是否考虑黄灯阶段或者全红阶段（取值0，1，2；0表示没有黄灯和全红相位、1表示只有黄灯相位）
        # nextdecision_point_time  下一次的决策点时间    lastdecision_point_time 上一次的决策点时间

        intersectionid = inter_parameter_configuration_object.interid
        programeid = inter_parameter_configuration_object.programeid
        action_name = inter_parameter_configuration_object.next_stage_label
        current_Phaselabel = inter_parameter_configuration_object.current_stage_label
        ##nextgreenstage_starttime=inter_parameter_configuration_object.nextgreenstage_starttime   #变成了下面的
        nextcyclestart_nextgreenstagestart_time = (
            inter_parameter_configuration_object.nextcyclestart_nextgreenstagestart_time)
        intersection_phaseprograme_object = network_parameter_configuration_object
        nextdecision_point_time = inter_parameter_configuration_object.nextdecision_point_time
        lastdecision_point_time = inter_parameter_configuration_object.lastdecision_point_time

        # 修改函数  增加交叉口类型 intersection_object  intersectionid   programeid
        stage_mingreentime = (
            inter_parameter_configuration_object.intersectionid_programeid_stage_mingreentime_dict)[intersectionid][programeid]
        stage_maxgreentime = (
            inter_parameter_configuration_object.intersectionid_programeid_stage_maxgreentime_dict)[intersectionid][programeid]
        stage_tonextstage_whetheryelloworred = (
            inter_parameter_configuration_object.intersectionid_programeid_stage_nextstage_dict)[intersectionid][programeid]

        # 该相位切换至下一个绿灯相位经历的黄灯+全红时间
        yellowplusallredtime = stage_tonextstage_whetheryelloworred[current_Phaselabel][2]
        # 下一个相位阶段序号   绿灯放行
        next_stage_label = stage_tonextstage_whetheryelloworred[current_Phaselabel][0]

        next_label = 0 if current_Phaselabel == len(stage_maxgreentime) - 1 else current_Phaselabel + 1
        # 当无黄灯或者全红相位时，is_yelloworred_next=next_stage_label 当由黄灯或者全红相位是，next_stage_label>is_yelloworred_next
        is_yelloworred_next = (
            next_label) if stage_tonextstage_whetheryelloworred[current_Phaselabel][1] > 0 else next_stage_label

        nextphasegreentime_overmaxgreen = stage_mingreentime[next_stage_label]

        nextphasegreentime_actionselection = stage_mingreentime[action_name]

        # 行为动作决策得到的下一个执行的相位阶段和本阶段号一致，延长该阶段的绿灯时间
        if action_name == current_Phaselabel:
            # 如果延长单位绿灯时间导致超过最大绿灯时长，则应该进入下一相位阶段，不重新决策
            if (nextdecision_point_time - nextcyclestart_nextgreenstagestart_time +
                    inter_parameter_configuration_object.greenlength_extension > stage_maxgreentime[current_Phaselabel]):
                # print("达到最大绿灯时间，切换至下一相位")
                # print("当前阶段运行阶段：",current_Phaselabel,"  运行时间：",step+1-nextcyclestart_nextgreenstagestart_time)
                currentcarphase_greentime = step + 1 - nextcyclestart_nextgreenstagestart_time
                # 切换至下一个阶段运行
                # 下一个阶段开始时间
                nextcyclestart_nextgreenstagestart_time = step + 1 + yellowplusallredtime
                # 切换至下一个阶段   正常的黄灯或者全红阶段运行，当无黄灯或者全红阶段，直接运行至下一绿灯相位阶段
                traci.trafficlight.setPhase(intersectionid, is_yelloworred_next)
                # traci.trafficlight.setPhase('0',next_stage_label)
                nextdecision_point_time = (
                        nextdecision_point_time + yellowplusallredtime + nextphasegreentime_overmaxgreen)  # 默认3s黄灯
                lastdecision_point_time = step + 1
                # 更新参数
                inter_parameter_configuration_object.nextdecision_point_time = nextdecision_point_time
                inter_parameter_configuration_object.lastdecision_point_time = lastdecision_point_time
                inter_parameter_configuration_object.nextcyclestart_nextgreenstagestart_time = (
                    nextcyclestart_nextgreenstagestart_time)
                inter_parameter_configuration_object.next_stage_label = next_stage_label
                inter_parameter_configuration_object.currentcarphase_greentime = currentcarphase_greentime
            else:
                # 设置当前相位的延长时间
                traci.trafficlight.setPhaseDuration(intersectionid, inter_parameter_configuration_object.greenlength_extension)
                # 更新下一个决策点时间
                nextdecision_point_time = nextdecision_point_time + inter_parameter_configuration_object.greenlength_extension
                lastdecision_point_time = step + 1
                next_stage_label = current_Phaselabel
                currentcarphase_greentime = step + 1 - nextcyclestart_nextgreenstagestart_time
                # print("当前阶段运行阶段：",current_Phaselabel,"  运行时间：",step+1-nextcyclestart_nextgreenstagestart_time)
                # 更新参数
                inter_parameter_configuration_object.nextdecision_point_time = nextdecision_point_time
                inter_parameter_configuration_object.lastdecision_point_time = lastdecision_point_time
                inter_parameter_configuration_object.next_stage_label = next_stage_label
                inter_parameter_configuration_object.currentcarphase_greentime = currentcarphase_greentime
        # 行为动作决策得到的下一个执行的相位阶段和本阶段号不一致，切换至对应的相位阶段号
        else:
            # 切换至下一个阶段运行
            ##切换至下一个阶段   正常的黄灯或者全红阶段运行，当无黄灯或者全红阶段，直接运行至下一绿灯相位阶段
            traci.trafficlight.setPhase(intersectionid, is_yelloworred_next)
            # print("当前阶段运行阶段：",current_Phaselabel,"  运行时间：",step+1-nextcyclestart_nextgreenstagestart_time)
            currentcarphase_greentime = step + 1 - nextcyclestart_nextgreenstagestart_time
            nextcyclestart_nextgreenstagestart_time = step + 1 + yellowplusallredtime  # 下一个阶段开始时间

            # traci.trafficlight.setPhase('0',action_name)
            next_stage_label = action_name
            # 更新决策点时间
            nextdecision_point_time = (
                    nextdecision_point_time + yellowplusallredtime + nextphasegreentime_actionselection)  # 3s黄灯
            lastdecision_point_time = step + 1

            # 更新参数
            inter_parameter_configuration_object.nextdecision_point_time = nextdecision_point_time
            inter_parameter_configuration_object.lastdecision_point_time = lastdecision_point_time
            inter_parameter_configuration_object.nextcyclestart_nextgreenstagestart_time = nextcyclestart_nextgreenstagestart_time
            inter_parameter_configuration_object.next_stage_label = next_stage_label
            inter_parameter_configuration_object.currentcarphase_greentime = currentcarphase_greentime


def actuatecontrol_foronestep(step, inter_parameter_configuration_object, network_parameter_configuration_object):
    # 每个仿真步执行，更新得到上一状态、下一决策时间点、上一决策时间点、下一绿灯相位或者周期开始时间、下一绿灯相位的序号
    # 等参数state_Last,nextdecision_point_time,lastdecision_point_time,nextcyclestart_nextgreenstagestart_time,next_stage_label
    traci.simulationStep(float(step + 1))

    state_represent.obtain_vehentertime(step, network_parameter_configuration_object, inter_parameter_configuration_object)

    # 需要进行决策
    if step + 1 == inter_parameter_configuration_object.nextdecision_point_time:
        # 得到路口“0”当前相位阶段编号  #这个要确认一下
        current_Phaselabel = traci.trafficlight.getPhase(inter_parameter_configuration_object.interid)
        # inter_parameter_configuration_class.pre_stage_label=inter_parameter_configuration_class.current_stage_label
        inter_parameter_configuration_object.current_stage_label = current_Phaselabel
        ###########################获取下一步动作的选择   ####################################################
        action_name = (choose_action.choose_action_induction(step,
                                                             network_parameter_configuration_object,
                                                             inter_parameter_configuration_object))
        inter_parameter_configuration_object.next_stage_label = action_name
        ##########################执行选择的动作，并得到下一步动作的决策时间点  里面的参数都要更新    认真检查###############
        action_decision.action_decision_phaseorder_change(inter_parameter_configuration_object,
                                                          network_parameter_configuration_object)

        # 说明一个周期内所有的绿灯相位都经历了，则周期结束
        if len((set(inter_parameter_configuration_object.greenstagelabellist).
                difference(set(inter_parameter_configuration_object.cycle_stage_passseq)))) == 0:
            # print("本周期开始时间：",inter_parameter_configuration_object.cycleendtime+1,
            # "本周期结束时间：",inter_parameter_configuration_object.nextcyclestart_nextgreenstagestart_time)
            inter_parameter_configuration_object.cycle_stage_passseq = []
            inter_parameter_configuration_object.cyclestarttime = (
                    inter_parameter_configuration_object.cycleendtime + 1)
            # 本周期结束时间点
            inter_parameter_configuration_object.cycleendtime = (
                inter_parameter_configuration_object.nextcyclestart_nextgreenstagestart_time)
        else:
            (inter_parameter_configuration_object.cycle_stage_passseq.
             append(inter_parameter_configuration_object.next_stage_label))

    if inter_parameter_configuration_object.nextcyclestart_nextgreenstagestart_time == step + 1:
        traci.trafficlight.setPhase(inter_parameter_configuration_object.interid,
                                    inter_parameter_configuration_object.next_stage_label)


pd_parameter3 = pd.DataFrame()
tripinfofile = "xxxx.xml"
sumocfgfile = "xxx.sumo.cfg"
loss_list1 = []  # 记录每次迭代的累计误差
singlesimulation_steplength = 3700
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print("正确")
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
# 表示不启用gui界面
traci.start(["sumo", '-c', sumocfgfile, "--tripinfo-output", tripinfofile, "--seed", str(4)])
# 路网参数的初始化
network_parameter_configuration_object = network_parameter_configuration()
# 单个路口，以后要是所有路口进行配置
inter_parameter_configuration_object = (
    inter_inductionpara_configuration("0", "0", network_parameter_configuration_object))
### 一些车检器参数的订阅
for inducid in traci.inductionloop.getIDList():
    feature_extract_fuc.subscribe_parameter_frominductionloop(inducid)
for detectorid in traci.lanearea.getIDList():
    feature_extract_fuc.subscribe_parameter_fromlaneareadet(detectorid)
# 仿真运行
for step in range(0, singlesimulation_steplength - 2):
    actuatecontrol_foronestep(step, inter_parameter_configuration_object, network_parameter_configuration_object)
