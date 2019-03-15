import os
import sys

v =  '''<flow id="flow_ns_c" route="route_ns" begin="bb" end="ee" vehsPerHour="200" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_nw_c" route="route_nw" begin="bb" end="ee" vehsPerHour="200" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ne_c" route="route_ne" begin="bb" end="ee" vehsPerHour="400" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_sw_c" route="route_sw" begin="bb" end="ee" vehsPerHour="400" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_sn_c" route="route_sn" begin="bb" end="ee" vehsPerHour="200" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_se_c" route="route_se" begin="bb" end="ee" vehsPerHour="200" departSpeed="max" departPos="base" departLane="best"/>

    <flow id="flow_en_c" route="route_en" begin="bb" end="ee" vehsPerHour="200" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ew_c" route="route_ew" begin="bb" end="ee" vehsPerHour="200" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_es_c" route="route_es" begin="bb" end="ee" vehsPerHour="400" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_wn_c" route="route_wn" begin="bb" end="ee" vehsPerHour="400" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_we_c" route="route_we" begin="bb" end="ee" vehsPerHour="200" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ws_c" route="route_ws" begin="bb" end="ee" vehsPerHour="200" departSpeed="max" departPos="base" departLane="best"/>'''

h = v

v2 =  '''<flow id="flow_ns_c" route="route_ns" begin="bb" end="ee" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_nw_c" route="route_nw" begin="bb" end="ee" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ne_c" route="route_ne" begin="bb" end="ee" probability="0.2" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_sw_c" route="route_sw" begin="bb" end="ee" probability="0.2" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_sn_c" route="route_sn" begin="bb" end="ee" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_se_c" route="route_se" begin="bb" end="ee" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>

    <flow id="flow_en_c" route="route_en" begin="bb" end="ee" probability="0.015" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ew_c" route="route_ew" begin="bb" end="ee" probability="0.015" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_es_c" route="route_es" begin="bb" end="ee" probability="0.030" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_wn_c" route="route_wn" begin="bb" end="ee" probability="0.030" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_we_c" route="route_we" begin="bb" end="ee" probability="0.015" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ws_c" route="route_ws" begin="bb" end="ee" probability="0.015" departSpeed="max" departPos="base" departLane="best"/>'''

h2 =  '''<flow id="flow_ns_c" route="route_ns" begin="bb" end="ee" probability="0.015" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_nw_c" route="route_nw" begin="bb" end="ee" probability="0.015" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ne_c" route="route_ne" begin="bb" end="ee" probability="0.030" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_sw_c" route="route_sw" begin="bb" end="ee" probability="0.030" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_sn_c" route="route_sn" begin="bb" end="ee" probability="0.015" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_se_c" route="route_se" begin="bb" end="ee" probability="0.015" departSpeed="max" departPos="base" departLane="best"/>

    <flow id="flow_en_c" route="route_en" begin="bb" end="ee" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ew_c" route="route_ew" begin="bb" end="ee" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_es_c" route="route_es" begin="bb" end="ee" probability="0.2" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_wn_c" route="route_wn" begin="bb" end="ee" probability="0.2" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_we_c" route="route_we" begin="bb" end="ee" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ws_c" route="route_ws" begin="bb" end="ee" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>'''


def get_context(begin, end, c):
    if c % 2 == 0:
        s = v
    else:
        s = h
    s = s.replace('c', str(c)).replace('bb', str(begin)).replace('ee', str(end))
    return s

def write_route_file(file=''):
    with open(file, 'w+') as f:
        f.write('''<routes>
                <route id="route_ns" edges="n_t t_s"/>
                <route id="route_nw" edges="n_t t_w"/>
                <route id="route_ne" edges="n_t t_e"/>
                <route id="route_we" edges="w_t t_e"/>
                <route id="route_wn" edges="w_t t_n"/>
                <route id="route_ws" edges="w_t t_s"/>
                <route id="route_ew" edges="e_t t_w"/>
                <route id="route_en" edges="e_t t_n"/>
                <route id="route_es" edges="e_t t_s"/>
                <route id="route_sn" edges="s_t t_n"/>
                <route id="route_se" edges="s_t t_e"/>
                <route id="route_sw" edges="s_t t_w"/>''')

        c = 0
        for i in range(0, 100000, 20000):
            f.write(get_context(i, i+20000, c))
            c += 1
        
        f.write('''</routes>''')

if __name__ == '__main__':
    write_route_file('nets/2way-single-intersection/single-intersection-gen.rou.xml')
