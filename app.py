from pytisfactory import *

Objects = deserilize(readlines('data.txt'))
Machines = MachineTable().load(Objects)

def mprint(machine: Machine):
    print(machine)
    print(format_architect(machine.architect()))
    print("") #endl

def 强化铁板10(smelter: bool = True):
    a = multiple(Machines["构筑站_螺丝"], 3)
    b = multiple(Machines["构筑站_铁棒"], 2)
    c = Serial("螺丝120", [b, a])
    d = multiple(Machines["构筑站_铁板"], 3)
    e = Parallel("强化铁板原料", [c, d])
    f = multiple(Machines["装配站_强化铁板"], 2)
    if smelter:
        g = multiple(Machines["冶炼站_铁锭"], 4)
        return Serial("强化铁板10", [g, e, f])
    else:
        return Serial("强化铁板10", [e, f])

def 转子8(smelter: bool = True):
    a = multiple(Machines["构筑站_螺丝"], 5)
    b = multiple(Machines["构筑站_铁棒"], 6)
    c = Buffer(Item("铁棒", 40))
    d = Parallel("螺丝及旁路", [a,c])
    e = multiple(Machines["转子"], 2)
    if smelter:
        f = multiple(Machines["冶炼站_铁锭"], 3)
        return Serial("转子8", [f, b, d, e])
    else:
        return Serial("转子8", [b, d, e])

def 模块化框架20(smelter: bool = True):
    a = multiple(强化铁板10(smelter=False), 3)
    b = multiple(Machines["铁棒"], 8)
    c = Parallel("模块化框架原料", [a,b])
    d = multiple(Machines["模块化框架"], 10)
    if smelter:
        e = multiple(Machines["冶炼站_铁锭"], 16)
        return Serial("模块化框架20", [e, c, d])
    else:
        return Serial("模块化框架20", [c, d])
    
def 油田小循环():
    a = multiple(Machines["精炼站_聚合树脂"], 3)
    x1 = Machines["精炼站_残渣燃料油"]
    x2 = multiple(Machines["燃油发电机"], 2)
    x = Serial("副产物发电", [x1, x2])
    y1 = Buffer(Item("聚合树脂", 390, ""))
    y2 = Machines["抽水站"]
    y = Parallel("塑料输入", [y1, y2])
    b = multiple(Machines["精炼站_聚合塑料"], 6)
    c = Serial("塑料产出", [y,b])
    d = Parallel("精炼产物消耗", [x, c])
    return Serial("油田小循环", [a, d])
    
    
if __name__ == "__main__":
    a = 油田小循环()
    mprint(a)
    save_text(mermaid(a), "out/a.mmd")
