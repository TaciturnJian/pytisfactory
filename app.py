from pytisfactory import *

Objects = deserilize(readlines('data.txt'))
Machines = MachineTable().load(Objects)

def mprint(machine: Machine):
    print(machine)
    print(machine.architect())
    print("") #endl

def 强化铁板10(smelter: bool = True):
    a = Machines["构筑站_螺丝"].multiply(3)
    b = Machines["构筑站_铁棒"].multiply(2)
    c = Serial("螺丝120", [b, a])
    d = Machines["构筑站_铁板"].multiply(3)
    e = Parallel("强化铁板原料", [c, d])
    f = Machines["装配站_强化铁板"].multiply(2)
    if smelter:
        g = Machines["冶炼站_铁锭"].multiply(4)
        return Serial("强化铁板10", [g, e, f])
    else:
        return Serial("强化铁板10", [e, f])

def 转子8(smelter: bool = True):
    a = Machines["构筑站_螺丝"].multiply(5)
    b = Machines("铁棒").multiply(6)
    c = Serial("转子原料", [b, a])
    d = Machines("转子").multiply(2)
    if smelter:
        e = Machines["冶炼站_铁锭"].multiply(3)
        return Serial("转子8", [e, c, d])
    else:
        return Serial("转子8", [c, d])

def 模块化框架20(smelter: bool = True):
    a = 强化铁板10(smelter=False).multiply(3)
    b = Machines("铁棒").multiply(8)
    c = Parallel("模块化框架原料", [a,b])
    d = Machines("模块化框架").multiply(10)
    if smelter:
        e = Machines["冶炼站_铁锭"].multiply(16)
        return Serial("模块化框架20", [e, c, d])
    else:
        return Serial("模块化框架20", [c, d])
    
def 油田小循环():
    a = Machines["精炼站_聚合树脂"].multiply(3)
    b = Machines["抽水站"]
    c = Parallel("塑料输入", [a, b])
    d = Machines["聚合塑料"].multiply(6)
    e = Machines("燃油发电机").multiply(2)
    return Serial("油田小循环", [c, d, e])

def test():
    a = Machines["构筑站_螺丝"].multiply(3)
    b = Machines("铁棒").multiply(2)
    return Serial("螺丝120", [b, a])
    
    
if __name__ == "__main__":
    a = 模块化框架20()
    mprint(a)
    save_text(mermaid(a), "out/a.mmd")
