# pytisfactory - python Tool for satisfactory

[Satisfactory](https://www.satisfactorygame.com/) is a game of building factories.

However, I'm not satisfied with the blueprint in the game.

I wrote this [python module](./pytisfactory.py) to help me:

- Calculate the input/output of a group of factories
- Save the topological graph of factory group and visualize it by mermaid code

You can explore [app.py](./app.py) for the usages.

幸福工厂拓扑工具，提供一种额外的蓝图方式(不能帮你一键在游戏中建造)。  
可以用来计算工厂消耗和产出，保存工厂的拓扑结构以及用于 mermaid 绘制的指令。  
仓库没有集成 mermaid 包，你需要自己将生成的 mmd 文件渲染成图片。

## example

With these code:

```python
from pytisfactory import *

Objects = deserilize(readlines('data.txt'))
Machines = MachineTable().load(Objects)

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

a = 油田小循环()
print(a)
print(format_architect(a.architect()))
save_text(mermaid(a), "out/a.mmd")
```

[The console output](./example/console.txt)
[The file output](./example/油田小循环.mmd)
